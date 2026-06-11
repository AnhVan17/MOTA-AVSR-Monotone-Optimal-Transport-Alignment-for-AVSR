# ViCocktail frame-shard verification

Nơi review/verify output **frame-schema** (video frames uint8 + Whisper audio fp16) trước khi
chạy preprocess lớn. Data nặng (`.tar`, `.png`, `_meta.json`) **không commit** (xem `.gitignore`).

```
verification/vicocktail_frames/
├── shards/      ← tải .tar (+ _meta.json) từ Modal volume về đây
└── montages/    ← script xuất ảnh montage mouth-crop ra đây để soi mắt thường
```

## Quy trình

1. Tải shard mẫu từ Modal volume `avsr-volume`:

   ```bash
   modal volume get avsr-volume \
     /vicocktail_features/vicocktail-avvn-train-000000-b000-000000.tar \
     verification/vicocktail_frames/shards/
   ```

2. Chạy verify (kiểm schema 2 tầng + xuất montage):

   ```bash
   python scripts/local/verify_frame_shard.py \
     "verification/vicocktail_frames/shards/*.tar" \
     verification/vicocktail_frames/montages \
     --n 6
   ```

3. Mở `montages/*.png` — kiểm crop có đúng **vùng miệng** không (dấu hiệu face-align OK),
   frame H×W đồng nhất, text tiếng Việt khớp.
