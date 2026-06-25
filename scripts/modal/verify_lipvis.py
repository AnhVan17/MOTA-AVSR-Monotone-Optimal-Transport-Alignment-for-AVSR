"""Mid-training verify của checkpoint lipvis trên DEV (KHÔNG chạm test → giữ test-chạm-1-lần).

Báo: epoch + best_metric của checkpoint, vài câu pred mẫu, và MODALITY ABLATION
(AV vs audio-only vs visual-only) trên dev NOISY (deterministic) với modality-dropout TẮT
(để cú drop thủ công là phép cô lập duy nhất). Trả lời: train có khoẻ + visual có đóng góp chưa?

  modal run scripts/modal/verify_lipvis.py                 # full dev (~5.9k/mode)
  modal run scripts/modal/verify_lipvis.py --limit 1000    # nhanh: cap 1000 sample/mode
  modal run scripts/modal/verify_lipvis.py --checkpoint /mnt/checkpoints/lipvis/epoch_14.pt
"""
import sys
from pathlib import Path

import modal

if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
else:
    sys.path.insert(0, "/root")
from src.infra.modal_image import ML_TRAIN_IMAGE, get_volume

APP_NAME = "avsr-verify-lipvis"
app = modal.App(APP_NAME)
volume = get_volume()


@app.function(image=ML_TRAIN_IMAGE, volumes={"/mnt": volume}, gpu="A10G", cpu=8.0, timeout=3600)
def verify(
    checkpoint: str = "/mnt/checkpoints/lipvis/best_model.pt",
    config_path: str = "/root/configs/phase_lipvis.yaml",
    limit: int = 0,
    batch_size: int = 16,
    shards: str = "",
):
    import copy

    import torch
    from torch.utils.data import DataLoader

    from src.data.collate import Collator
    from src.data.shards import build_webdataset
    from src.data.tokenizers import build_tokenizer
    from src.evaluation.decoding import CTCDecoder
    from src.evaluation.metrics import MetricCalculator
    from src.models.mota import create_model
    from src.utils.common import load_checkpoint
    from src.utils.config_utils import load_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(config_path)
    tokenizer = build_tokenizer(config)
    model = create_model(
        {**config["model"], "mqot": config.get("mqot", {}), "rgf": config.get("rgf", {})}
    ).to(device)
    ck = load_checkpoint(checkpoint, model, device=device)
    model.eval()
    print(f"checkpoint={checkpoint}")
    print(f"epoch={ck.get('epoch')}  best_metric(dev WER khi train)={ck.get('best_metric')}")
    qg = getattr(model, "quality_gate", None)
    if qg is not None and hasattr(qg, "residual_gate"):
        print(f"qg_residual_gate={torch.sigmoid(qg.residual_gate.detach()).item():.4f}")
    if hasattr(model, "fine_align_gate"):
        print(f"mqot_fine_align_gate={torch.sigmoid(model.fine_align_gate.detach()).item():.4f}")

    blank_id = getattr(tokenizer, "eot_token_id", config["model"].get("blank_id", 4))
    decoder = CTCDecoder(tokenizer, blank_id=blank_id)
    metric = MetricCalculator()
    collate = Collator(pad_id=blank_id)

    # Nguồn eval: mặc định = DEV (val_shards, proxy-noise aug, modality-dropout TẮT). Nếu truyền
    # --shards (glob test_snr_*) → eval TRÊN TEST: nhiễu babble đã baked vào feature nên KHÔNG aug
    # thêm; chỉ drop thủ công. Visual frame của test là sạch (chỉ audio bị nhiễu).
    eval_shards = shards or config["data"]["val_shards"]
    do_aug = (shards == "")
    print(f"eval on: {'DEV (proxy-noise)' if do_aug else f'TEST baked-noise: {eval_shards}'}")
    aug = copy.deepcopy(config.get("augmentation", {}))
    aug["modality_dropout_prob"] = 0.0

    def dev_loader():
        ds = build_webdataset(eval_shards, tokenizer, train=False, augment=do_aug,
                              aug_cfg=(aug if do_aug else None), shuffle_buffer=0, deterministic=do_aug)
        return DataLoader(ds, batch_size=batch_size, collate_fn=collate, num_workers=4)

    results, sample_printed = {}, False
    for drop in ("none", "visual", "audio"):
        refs, preds = [], []
        diag = {
            "gate_audio": 0.0,
            "gate_visual": 0.0,
            "q_audio": 0.0,
            "q_visual": 0.0,
            "batches": 0,
        }
        for batch in dev_loader():
            a = batch["audio"].to(device)
            v = batch["visual"].to(device)
            am = batch["audio_mask"].to(device)
            vm = batch["visual_mask"].to(device)
            if drop == "visual":
                v = torch.zeros_like(v)
            elif drop == "audio":
                a = torch.zeros_like(a)
            with torch.no_grad():
                out = model(a, v, target=None, audio_mask=am, visual_mask=vm)
                logits = out["ctc_logits"]
                gate = out.get("gate_weights")
                if gate is not None:
                    diag["gate_audio"] += gate[..., 0].float().mean().item()
                    diag["gate_visual"] += gate[..., 1].float().mean().item()
                if out.get("q_audio") is not None:
                    diag["q_audio"] += out["q_audio"].float().mean().item()
                if out.get("q_visual") is not None:
                    diag["q_visual"] += out["q_visual"].float().mean().item()
                diag["batches"] += 1
            preds += decoder.decode_batch(logits, method="greedy")
            refs += decoder.decode_targets(batch["target"])
            if limit and len(refs) >= limit:
                break
        wer = round(float(metric.compute_wer(preds, refs)), 2)
        cer = round(float(metric.compute_cer(preds, refs)), 2)
        results[drop] = (len(refs), wer, cer)
        if diag["batches"]:
            denom = diag["batches"]
            diag_str = (
                f"  gateA={diag['gate_audio'] / denom:.3f}"
                f" gateV={diag['gate_visual'] / denom:.3f}"
                f" qA={diag['q_audio'] / denom:.3f}"
                f" qV={diag['q_visual'] / denom:.3f}"
            )
        else:
            diag_str = ""
        print(f"  drop={drop:7s} n={len(refs):5d}  WER={wer:6.2f}  CER={cer:6.2f}{diag_str}")
        if drop == "none" and not sample_printed:
            for i in range(min(3, len(refs))):
                print(f"    REF : {refs[i]}")
                print(f"    PRED: {preds[i]}")
            sample_printed = True

    if getattr(model, "use_visual_ctc_aux", False):
        refs, preds = [], []
        for batch in dev_loader():
            a = batch["audio"].to(device)
            v = batch["visual"].to(device)
            am = batch["audio_mask"].to(device)
            vm = batch["visual_mask"].to(device)
            with torch.no_grad():
                out = model(a, v, target=None, audio_mask=am, visual_mask=vm)
                logits = out.get("visual_ctc_logits")
            if logits is None:
                break
            preds += decoder.decode_batch(logits, method="greedy")
            refs += decoder.decode_targets(batch["target"])
            if limit and len(refs) >= limit:
                break
        if refs:
            wer = round(float(metric.compute_wer(preds, refs)), 2)
            cer = round(float(metric.compute_cer(preds, refs)), 2)
            results["visual_ctc_direct"] = (len(refs), wer, cer)
            print(f"  visual-ctc-direct n={len(refs):5d}  WER={wer:6.2f}  CER={cer:6.2f}")

    av, ao, vo = results["none"][1], results["visual"][1], results["audio"][1]
    print(f"\n===== MODALITY ABLATION ({'noisy dev' if do_aug else 'TEST baked-noise'}) =====")
    print(f"  AV={av:.2f}   audio-only={ao:.2f}   visual-only={vo:.2f}")
    print(f"  → visual contribution (audio-only − AV) = {ao - av:+.2f} WER")
    print("    >0 = visual GIÚP (audio-only tệ hơn AV) · ~0 = chưa đóng góp · visual-only<100 = lip path đọc được")
    return results


@app.local_entrypoint()
def main(
    checkpoint: str = "/mnt/checkpoints/lipvis/best_model.pt",
    config_path: str = "/root/configs/phase_lipvis.yaml",
    limit: int = 0,
    shards: str = "",
):
    verify.remote(checkpoint=checkpoint, config_path=config_path, limit=limit, shards=shards)
