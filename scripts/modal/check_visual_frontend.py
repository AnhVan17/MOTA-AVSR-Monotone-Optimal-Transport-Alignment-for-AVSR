"""Phase-B PRE-FLIGHT (FREE, no training run): sanity-check the pretrained lip-reading frontend on
REAL ViCocktail mouth-crop frames BEFORE spending a GPU training run.

It loads the frozen frontend from the volume and runs it on a few dozen clips, then asserts the
features are non-degenerate and the checkpoint actually matched. It CANNOT prove the features are
good for Vietnamese lip-reading (that needs training) — it catches gross integration failures:
weights not loaded / wrong relu_type, NaN/constant/zero features, wrong shape, no temporal response.

  modal run scripts/modal/check_visual_frontend.py                       # defaults (swish, test shards)
  modal run scripts/modal/check_visual_frontend.py --relu-type relu      # if the ckpt is relu/swish
  modal run scripts/modal/check_visual_frontend.py --n-clips 80
"""
import sys
from pathlib import Path

import modal

if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
else:
    sys.path.insert(0, "/root")
from src.infra.modal_image import ML_TRAIN_IMAGE, get_volume

APP_NAME = "avsr-check-visual"
FEAT = "/mnt/vicocktail_features"

app = modal.App(APP_NAME)
volume = get_volume()


@app.function(image=ML_TRAIN_IMAGE, volumes={"/mnt": volume}, gpu="A10G", timeout=1800)
def check(
    weights: str = "/mnt/pretrained/lrw_resnet18_frontend.pth",
    shards: str = f"{FEAT}/vicocktail-avvn-test-000000-*.tar",
    relu_type: str = "swish",
    n_clips: int = 50,
):
    import os

    import torch
    from torch.utils.data import DataLoader

    from src.data.collate import Collator
    from src.data.shards import build_webdataset
    from src.data.tokenizers import build_tokenizer
    from src.models.visual.lipreading_frontend import LipReadingFrontend
    from src.utils.config_utils import load_config

    assert os.path.exists(weights), f"weights missing on volume: {weights} (run Phase A7 upload first)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fe = LipReadingFrontend(weights=None, relu_type=relu_type).to(device).eval()
    stats = fe.load_pretrained(weights)  # explicit → capture match stats
    print(f"checkpoint match: loaded={stats['loaded']} missing={len(stats['missing'])} "
          f"unexpected={len(stats['unexpected'])} (relu_type={relu_type})")

    cfg = load_config("/root/configs/phase_lipvis.yaml")
    tok = build_tokenizer(cfg)
    ds = build_webdataset(shards, tok, train=False, augment=False, shuffle_buffer=0)
    loader = DataLoader(
        ds, batch_size=4, collate_fn=Collator(getattr(tok, "eot_token_id", 4)), num_workers=4
    )

    feats = []  # list of per-clip [T, 512]
    with torch.no_grad():
        for batch in loader:
            out = fe(batch["visual"].to(device))  # [B,T,512]
            for i in range(out.shape[0]):
                feats.append(out[i].float().cpu())
            if len(feats) >= n_clips:
                break

    allf = torch.cat([f.reshape(-1) for f in feats])
    finite = bool(torch.isfinite(allf).all())
    magnitude = float(allf.abs().mean())
    # temporal response: per-clip variance across time (averaged over feature dims), then over clips
    temporal_var = float(
        torch.stack([f.var(dim=0).mean() for f in feats if f.shape[0] > 1]).mean()
    )
    # cross-clip distinctness: spread of per-clip mean feature vectors
    clip_means = torch.stack([f.mean(dim=0) for f in feats])  # [N,512]
    cross_clip_std = float(clip_means.std(dim=0).mean())

    print(
        f"\nn_clips={len(feats)} shape_ok={feats[0].shape[1] == 512} finite={finite} "
        f"magnitude={magnitude:.4f} temporal_var={temporal_var:.5f} cross_clip_std={cross_clip_std:.5f}"
    )
    checks = {
        "checkpoint_fully_matched (missing==0)": len(stats["missing"]) == 0,
        "shape == [T,512]": feats[0].shape[1] == 512,
        "features finite": finite,
        "magnitude > 0.01": magnitude > 0.01,
        "temporal_var > 1e-4 (responds to lip motion)": temporal_var > 1e-4,
        "cross_clip_std > 1e-3 (clips distinguishable)": cross_clip_std > 1e-3,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    verdict = all(checks.values())
    print(
        "\nPRE-FLIGHT "
        + ("PASS — integration OK, safe to launch the training run."
           if verdict
           else "FAIL — fix (relu_type / channel-order / normalize / crop) BEFORE burning a run.")
    )
    return verdict


@app.local_entrypoint()
def main(
    weights: str = "/mnt/pretrained/lrw_resnet18_frontend.pth",
    shards: str = f"{FEAT}/vicocktail-avvn-test-000000-*.tar",
    relu_type: str = "swish",
    n_clips: int = 50,
):
    ok = check.remote(weights=weights, shards=shards, relu_type=relu_type, n_clips=n_clips)
    print("RESULT:", "PASS" if ok else "FAIL")
