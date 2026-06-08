# Scripts

Chia rõ **2 phần** theo nơi chạy:

## `modal/` — chạy trên cloud (Modal GPU)

Vỏ Modal mỏng: định nghĩa `image` + `volume` + `@app.function(gpu=...)`, rồi gọi logic
train THUẦN ở [`src/training/run.py`](../src/training/run.py). Không lặp lại logic train.

```text
modal/
├── train_phase1.py, train_phase2.py   # training (gọi src/training/run.py)
├── data_prep/                         # tiền xử lý: crop, extract feature
├── inference/                         # inference_phase1
└── utils/                             # quản lý Modal Volume, debug, verify vocab
```

```bash
modal run scripts/modal/train_phase1.py
modal run scripts/modal/train_phase2.py
modal run scripts/modal/data_prep/prep_features_gpu.py
modal run scripts/modal/utils/check_volume.py
```

## `local/` — chạy thẳng trên máy, tự dò device

Tự detect **cuda → mps (Apple) → cpu** qua [`src/utils/device.py`](../src/utils/device.py).
Phục vụ test nhanh, không cần Modal/cloud.

```text
local/
├── smoke_test.py     # smoke test nhanh (không cần data thật)
├── lr_finder.py      # LR range test
└── data/             # tiện ích data thuần local (split/merge manifest, verify vocab, download)
```

```bash
# Smoke test: kiểm pipeline (model→loss→backward) chạy được, KHÔNG cần data thật
python scripts/local/smoke_test.py                 # tự dò device
python scripts/local/smoke_test.py --device cpu     # ép CPU
python scripts/local/smoke_test.py --no-mqot        # tắt MQOT

# LR range test
python scripts/local/lr_finder.py --config configs/phase1_base.yaml

# Tiện ích data (chạy từ repo root)
python scripts/local/data/split_manifest.py
python scripts/local/data/verify_vocab_vi.py
```

### Lưu ý Apple Silicon (MPS)

- **AMP** (mixed precision) chỉ chạy trên CUDA → tự tắt trên MPS/CPU.
- **CTC loss** (`aten::_ctc_loss`) chưa có trên MPS → smoke_test bật sẵn
  `PYTORCH_ENABLE_MPS_FALLBACK=1` để op này fallback CPU. Train thật trên Mac vì vậy
  sẽ chậm ở bước CTC; ưu tiên dùng Modal/CUDA cho train thật, MPS chỉ để smoke/debug.

## Nguyên tắc DRY

Logic train viết MỘT lần ở `src/training/run.py::run_training(config)`. Cả `modal/` và
`local/` chỉ dựng `config` (đường dẫn khác nhau) rồi gọi vào đó. Không sao chép vòng train.
