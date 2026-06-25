"""Full-test evaluation of a Phase-1 checkpoint over ALL 9 conditions (clean + 8 noisy),
on Modal GPU, using the WebDataset feature shards on the volume. Computes BOTH greedy and
beam-search CTC decode (one shared forward pass → greedy is nearly free), with tone-preserving
Vietnamese WER/CER → Table 5.1. Saves results JSON to the volume.

Why a Modal script (not a local notebook): ~10.5k samples × beam = a GPU job over data that
lives on the volume. The companion notebook (notebooks/eval_phase1.ipynb) LOADS the JSON to
present the tables/plot.

  modal run scripts/modal/eval_phase1.py --limit 50                # quick smoke (50/condition) first!
  modal run --detach scripts/modal/eval_phase1.py                  # full test, greedy + beam(10)
  modal run scripts/modal/eval_phase1.py --decode greedy           # greedy only (fast, skip beam)
  modal run scripts/modal/eval_phase1.py --checkpoint /mnt/checkpoints/phase1/epoch_25.pt

  # Phase-2 (MQOT) checkpoint → MUST pass phase2.yaml so the architecture matches (use_mqot=true);
  # load_state_dict is strict, so a mismatched config crashes on the mqot.* keys.
  modal run --detach scripts/modal/eval_phase1.py \
      --checkpoint /mnt/checkpoints/phase2/epoch_2.pt \
      --config-path /root/configs/phase2.yaml --decode greedy
"""
import json
import os
import sys
from pathlib import Path

import modal

if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
else:
    sys.path.insert(0, "/root")
from src.infra.modal_image import ML_TRAIN_IMAGE, get_volume

APP_NAME = "avsr-eval-phase1"
FEAT = "/mnt/vicocktail_features"

# 9 test conditions → shard globs (clean uses "test-000000", noisy uses "test_snr_*").
CONDITIONS = {
    "clean":      f"{FEAT}/vicocktail-avvn-test-000000-*.tar",
    "snr-5_int1": f"{FEAT}/vicocktail-avvn-test_snr_-5_interferer_1-000000-*.tar",
    "snr-5_int2": f"{FEAT}/vicocktail-avvn-test_snr_-5_interferer_2-000000-*.tar",
    "snr0_int1":  f"{FEAT}/vicocktail-avvn-test_snr_0_interferer_1-000000-*.tar",
    "snr0_int2":  f"{FEAT}/vicocktail-avvn-test_snr_0_interferer_2-000000-*.tar",
    "snr5_int1":  f"{FEAT}/vicocktail-avvn-test_snr_5_interferer_1-000000-*.tar",
    "snr5_int2":  f"{FEAT}/vicocktail-avvn-test_snr_5_interferer_2-000000-*.tar",
    "snr10_int1": f"{FEAT}/vicocktail-avvn-test_snr_10_interferer_1-000000-*.tar",
    "snr10_int2": f"{FEAT}/vicocktail-avvn-test_snr_10_interferer_2-000000-*.tar",
}

SAMPLES_PER_CONDITION = 1167  # ~ViCocktail test samples / condition (for the progress-bar total)

app = modal.App(APP_NAME)
volume = get_volume()


@app.function(image=ML_TRAIN_IMAGE, volumes={"/mnt": volume}, gpu="A10G", cpu=8.0, timeout=10800)
def evaluate_all(
    checkpoint: str,
    decode: str = "both",      # "both" | "greedy" | "beam"
    beam_width: int = 10,
    limit: int = 0,            # 0 = full set; >0 = cap samples/condition (smoke)
    batch_size: int = 16,
    config_path: str = "/root/configs/phase1.yaml",  # MUST match the ckpt's arch (phase2.yaml for MQOT)
    drop: str = "none",        # "none" | "visual" (→ audio-only) | "audio" (→ visual-only): modality ablation
    out_path: str = "/mnt/eval/phase1_results.json",
):
    import math

    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from src.data.collate import Collator
    from src.data.shards import build_webdataset
    from src.data.tokenizers import build_tokenizer
    from src.evaluation.decoding import CTCDecoder
    from src.evaluation.metrics import MetricCalculator
    from src.models.mota import create_model
    from src.utils.common import load_checkpoint
    from src.utils.config_utils import load_config

    do_greedy = decode in ("both", "greedy")
    do_beam = decode in ("both", "beam")
    methods = [m for m, on in (("greedy", do_greedy), ("beam", do_beam)) if on]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(config_path)
    print(f"model config: {config_path} (use_mqot={config['model'].get('use_mqot')})")
    tokenizer = build_tokenizer(config)
    model = create_model(
        {**config["model"], "mqot": config.get("mqot", {}), "rgf": config.get("rgf", {})}
    ).to(device)
    ck = load_checkpoint(checkpoint, model, device=device)
    model.eval()
    print(f"loaded {checkpoint}: epoch={ck.get('epoch')} best_metric(val WER)={ck.get('best_metric')}")

    blank_id = getattr(tokenizer, "eot_token_id", config["model"].get("blank_id", 4))
    decoder = CTCDecoder(tokenizer, blank_id=blank_id)
    metric = MetricCalculator()
    collate = Collator(pad_id=blank_id)

    def _wc(preds, refs):
        return {"wer": round(float(metric.compute_wer(preds, refs)), 2),
                "cer": round(float(metric.compute_cer(preds, refs)), 2)}

    results = {}
    for name, glob in CONDITIONS.items():
        ds = build_webdataset(glob, tokenizer, train=False, augment=False, shuffle_buffer=0)
        loader = DataLoader(
            ds, batch_size=batch_size, collate_fn=collate, num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )
        refs, gp, bp = [], [], []
        n_est = min(limit, SAMPLES_PER_CONDITION) if limit else SAMPLES_PER_CONDITION
        for batch in tqdm(loader, desc=name, total=math.ceil(n_est / batch_size)):
            audio_in = batch["audio"].to(device)
            visual_in = batch["visual"].to(device)
            if drop == "visual":      # audio-only: is the frozen visual stream actually used?
                visual_in = torch.zeros_like(visual_in)
            elif drop == "audio":     # visual-only: how much lip info does the model decode alone?
                audio_in = torch.zeros_like(audio_in)
            with torch.no_grad():
                logits = model(audio_in, visual_in, target=None)["ctc_logits"]
            if do_greedy:
                gp += decoder.decode_batch(logits, method="greedy")
            if do_beam:
                bp += decoder.decode_batch(logits, method="beam", beam_width=beam_width)
            refs += decoder.decode_targets(batch["target"])
            if limit and len(refs) >= limit:
                break
        entry = {"n": len(refs)}
        if do_greedy:
            entry["greedy"] = _wc(gp, refs)
        if do_beam:
            entry["beam"] = _wc(bp, refs)
        results[name] = entry
        msg = f"  {name:12s} n={len(refs):5d}" + "".join(
            f"  {m} WER={entry[m]['wer']:6.2f}" for m in methods
        )
        print(msg)

    # ---- Tables (one per decode method) ----
    noisy = {k: v for k, v in results.items() if k != "clean"}
    for m in methods:
        print(f"\n=========  TABLE 5.1 — Phase-1 baseline ({m})  =========")
        print(f"{'condition':14s}{'N':>7}{'WER%':>9}{'CER%':>9}")
        for name, r in results.items():
            print(f"{name:14s}{r['n']:>7}{r[m]['wer']:>9.2f}{r[m]['cer']:>9.2f}")
        if noisy:
            aw = sum(v[m]["wer"] for v in noisy.values()) / len(noisy)
            ac = sum(v[m]["cer"] for v in noisy.values()) / len(noisy)
            print(f"{'noisy-avg':14s}{'':>7}{aw:>9.2f}{ac:>9.2f}")
    print("\n(ViCocktail AV-HuBERT ref: ~9.4% clean — pretrained SOTA, not from-scratch)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": checkpoint, "decode": decode, "beam_width": beam_width,
                "epoch": ck.get("epoch"), "best_metric": ck.get("best_metric"),
                "results": results,
            },
            f, ensure_ascii=False, indent=2,
        )
    volume.commit()
    print(f"\nsaved → {out_path}")


@app.local_entrypoint()
def main(
    checkpoint: str = "/mnt/checkpoints/phase1/best_model.pt",
    decode: str = "both",
    beam_width: int = 10,
    limit: int = 0,
    batch_size: int = 16,
    config_path: str = "/root/configs/phase1.yaml",
    drop: str = "none",
    out_path: str = "/mnt/eval/phase1_results.json",
):
    evaluate_all.remote(
        checkpoint=checkpoint, decode=decode, beam_width=beam_width,
        limit=limit, batch_size=batch_size, config_path=config_path, drop=drop, out_path=out_path,
    )
