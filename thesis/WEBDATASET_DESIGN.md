# WebDataset Sharding — Design Note (MOTA-v2, Phase A)

**Ngày:** 2026-06-10 · **Branch:** `feat/phase-a-data` · **Trạng thái:** design (chưa implement)
**Phiên bản pin:** `webdataset==0.2.79` (trong `PREPROC_IMAGE`).
**Bối cảnh:** giải giới hạn **500,000 inodes/Volume** + **262,144 files/dir** của Modal (đã verify, [docs](https://modal.com/docs/guide/volumes)). Bổ trợ [ENGINEERING_PLAN.md](ENGINEERING_PLAN.md) §A.2–A.3.

---

## 0. Vấn đề & giải pháp

- **Loose `.pt`:** mỗi sample = 2–3 file (`audio.pt`, `visual.pt`, manifest). Train ViCocktail ≈ **99 raw shard × ~1,829 ≈ 181k sample** → **>360k file** → vỡ cap 500k inode + attach Volume chậm (latency tuyến tính theo số file).
- **WebDataset:** gói ~2,000 sample vào **1 `.tar` shard** → file count giảm **~2000×** (≈ 90 file train). Raw ViCocktail vốn đã là `.tar` → vào khuôn tự nhiên; chỉ cần bước EXTRACT **xuất `.tar`** thay vì `.pt` rời.

---

## 1. Khái niệm cốt lõi

1. **Shard** = 1 file `.tar` chứa nhiều sample, đọc tuần tự.
2. **Sample** = nhóm file **cùng basename** (`__key__`), phân biệt bằng **đuôi**. VD `__key__=0000001933` → `0000001933.audio.pth`, `0000001933.visual.pth`, `0000001933.txt` = 1 sample.
3. **Streaming (IterableDataset)** — **KHÔNG random-access**. Hệ quả: shuffle qua shard-shuffle + buffer; không có `__len__`; chia data theo **shard** (không theo sample).

---

## 2. Layout 1 shard

| File trong tar     | Nội dung                       | Encode (auto theo đuôi) |
| ------------------ | ------------------------------ | ----------------------- |
| `{key}.audio.pth`  | tensor `[T_a, 768]` (Whisper)  | `.pth` → `torch.save`   |
| `{key}.visual.pth` | tensor `[T_v, 512]` (ResNet18) | `.pth` → `torch.save`   |
| `{key}.txt`        | transcript tiếng Việt          | `.txt` → utf-8          |

- **Tên shard:** `vicocktail-{split}-%06d.tar` (split = `train` / `test_snr_0_interferer_1` / …).
- **maxcount ≈ 2000 sample/shard** (qua config, KHÔNG hardcode) → ~90 shard train. Chọn maxcount đủ nhỏ để **#shard ≫ #DataLoader workers** (xem §6).
- **`_meta.json`** kèm theo: `{"num_samples": N, "num_shards": M}` → để biết `steps/epoch`.

---

## 3. Ghi — `ShardWriter` (sửa preprocessor)

```python
import webdataset as wds

with wds.ShardWriter(f"{out_dir}/vicocktail-{split}-%06d.tar", maxcount=2000) as sink:
    for s in processed_samples:          # mỗi sample sau crop (face-alignment) + extract
        sink.write({
            "__key__":    s.id,          # basename duy nhất
            "audio.pth":  s.audio_feat,  # torch.Tensor → tự torch.save
            "visual.pth": s.visual_feat,
            "txt":        s.text,        # str → utf-8
        })
# đạt maxcount (hoặc maxsize) → ShardWriter tự xoay sang shard kế
```

- `maxcount` (số sample) và/hoặc `maxsize` (bytes) — đạt cái nào trước thì xoay shard.
- `.pth` auto `torch.save`; `.txt` auto utf-8. `__key__` là basename, các key còn lại thành file theo đuôi.

---

## 4. Đọc — pipeline (loader mới `src/data/datasets/webdataset_loader.py`)

```python
import webdataset as wds, io, torch

def make_decode(tokenizer):
    def decode(sample):
        text = sample["txt"].decode("utf-8")
        return {
            "audio":    torch.load(io.BytesIO(sample["audio.pth"])),   # decode .pth tường minh
            "visual":   torch.load(io.BytesIO(sample["visual.pth"])),
            "text":     text,
            "target":   tokenizer.encode(text),                        # tokenize tại đây
            "rel_path": sample["__key__"],
        }
    return decode

ds = (
    wds.WebDataset(shard_glob, shardshuffle=is_train, handler=wds.warn_and_continue)
       .shuffle(1000 if is_train else 0)     # buffer sample-level (train mới shuffle)
       .map(make_decode(tokenizer))
)
loader = DataLoader(ds, batch_size=B, num_workers=4, collate_fn=Collator(pad_id))
```

- **Decode `.pth` tường minh** bằng `torch.load(io.BytesIO(...))` — chắc hơn dựa vào auto-decoder.
- **Tokenize** text→target trong `.map` (dùng `WhisperTokenizer` sẵn có).
- **`Collator` cũ giữ nguyên** — vẫn pad variable-length + drop None. WDS yield sample lẻ; DataLoader gom batch qua Collator.

---

## 5. Tích hợp code

| File                                           | Thay đổi                                                                                                         |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `src/data/preprocessors/base.py` `run()`       | thay vòng `torch.save` rời bằng `ShardWriter`; ghi `_meta.json`                                                  |
| `src/data/datasets/webdataset_loader.py` (MỚI) | dựng `wds.WebDataset(...).shuffle(...).map(decode)`                                                              |
| `src/data/loader.py`                           | nhánh theo `data.format`: `webdataset` (mới) vs `jsonl` (giữ cho smoke local)                                    |
| `configs/*.yaml`                               | `data.format`, `data.shards` (glob, vd `/mnt/vicocktail_features/vicocktail-train-*.tar`), `data.shard_maxcount` |

> Giữ đường `.pt`/manifest cũ làm fallback cho smoke-test local nhỏ — chọn qua config, không xoá.

---

## 6. Gotcha quan trọng (vì IterableDataset)

1. **Shuffle 2 tầng:**
   - Shard-level: `shardshuffle=True` (đảo thứ tự shard mỗi epoch).
   - Sample-level: `.shuffle(N)` (buffer reservoir N sample).
   - **Train bật cả hai; Val/Test TẮT** (`shardshuffle=False`, `.shuffle(0)`) → thứ tự cố định → **WER lặp lại được**.
2. **Chia theo SHARD, không theo sample:**
   - `num_workers>0`: WDS phân shard cho từng worker (`split_by_worker`) → tránh worker đọc trùng.
   - Đa-GPU (DDP): thêm `split_by_node` để chia shard theo node.
   - ⚠️ **#shard ≫ #workers (× #nodes)**, nếu không có worker rỗng → giảm `maxcount` để nhiều shard hơn.
3. **Không có `__len__`:** IterableDataset không báo độ dài → dùng `.with_epoch(N)` hoặc đọc `_meta.json["num_samples"]` để tính `steps/epoch` cho training loop + scheduler.

---

## 7. Edge cases & quyết định thiết kế

- **Sample lỗi/corrupt trong tar:** `handler=wds.warn_and_continue` (bỏ qua, log) — cộng với Collator đã drop None.
- **Shard cuối lẻ (< maxcount):** ShardWriter tự xử, không cần can thiệp.
- **Variable length (T_a, T_v khác nhau):** vẫn để `Collator` pad — KHÔNG pad lúc ghi shard (giữ shard gọn).
- **Determinism Val:** không shuffle, không `with_epoch`.
- **Reproducibility:** seed cho shuffle buffer + shard order (qua config) để chạy lại giống nhau.

---

## 8. Kế hoạch implement (TDD)

1. **Test trước:** `tests/unit/test_webdataset.py` — round-trip: ghi 5 sample synthetic → đọc lại → assert audio/visual/text khớp + shape giữ nguyên.
2. `ShardWriter` trong `base.py` `run()` + `_meta.json`.
3. `webdataset_loader.py` (decode + tokenize + map) + test decode đúng dict cho Collator.
4. `loader.py` branch `data.format`.
5. Config keys.
6. **Modal smoke:** process 1 raw shard → feature `.tar` → đọc lại → train 1 epoch (loss hữu hạn, WER sane).

**Nghiệm thu:** round-trip test pass · 1 raw shard → feature-shard trên Modal không crash · train đọc shard chạy 1 epoch OK · file count Volume << 500k.

---

## Sources (verified)

- webdataset GitHub + FAQ: <https://github.com/webdataset/webdataset>
- ShardWriter `.pth`/`__key__` convention (SpeechBrain WDS tutorial): <https://speechbrain.readthedocs.io/en/v1.0.2/tutorials/advanced/data-loading-for-big-datasets-and-shared-filesystems.html>
- split_by_node / split_by_worker / with_epoch (WebDataset multinode docs): <https://rom1504.github.io/webdataset/multinode/>
- HF WebDataset format: <https://huggingface.co/docs/hub/datasets-webdataset>
- Modal Volume limits: <https://modal.com/docs/guide/volumes>
