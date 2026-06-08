# Scripts

Split into **two parts** by where they run:

## `modal/` — runs on the cloud (Modal GPU)

Thin Modal wrappers: they define `image` + `volume` + `@app.function(gpu=...)`, then call the
pure training logic in [`src/training/run.py`](../src/training/run.py). Training logic is not duplicated.

```text
modal/
├── train_phase1.py, train_phase2.py   # training (calls src/training/run.py)
├── data_prep/                         # preprocessing: crop, feature extraction
├── inference/                         # inference_phase1
└── utils/                             # Modal Volume management, debug, vocab verification
```

```bash
modal run scripts/modal/train_phase1.py
modal run scripts/modal/train_phase2.py
modal run scripts/modal/data_prep/prep_features_gpu.py
modal run scripts/modal/utils/check_volume.py
```

## `local/` — runs directly on your machine, auto-detects device

Auto-detects **cuda → mps (Apple) → cpu** via [`src/utils/device.py`](../src/utils/device.py).
For fast testing, no Modal/cloud required.

```text
local/
├── smoke_test.py     # fast smoke test (no real data needed)
├── lr_finder.py      # LR range test
└── data/             # pure local data utilities (split/merge manifest, verify vocab, download)
```

```bash
# Smoke test: verify the pipeline (model->loss->backward) runs, NO real data needed
python scripts/local/smoke_test.py                 # auto-detect device
python scripts/local/smoke_test.py --device cpu     # force CPU
python scripts/local/smoke_test.py --no-mqot        # disable MQOT

# LR range test
python scripts/local/lr_finder.py --config configs/phase1_base.yaml

# Data utilities (run from repo root)
python scripts/local/data/split_manifest.py
python scripts/local/data/verify_vocab_vi.py
```

### Apple Silicon (MPS) notes

- **AMP** (mixed precision) only runs on CUDA → auto-disabled on MPS/CPU.
- **CTC loss** (`aten::_ctc_loss`) is not implemented on MPS → smoke_test sets
  `PYTORCH_ENABLE_MPS_FALLBACK=1` so this op falls back to CPU. Real training on a Mac is
  therefore slow at the CTC step; prefer Modal/CUDA for real training, use MPS only for
  smoke/debug.

## DRY principle

Training logic is written ONCE in `src/training/run.py::run_training(config)`. Both `modal/`
and `local/` only build a `config` (with different paths) and call into it. The training loop
is never copied.
