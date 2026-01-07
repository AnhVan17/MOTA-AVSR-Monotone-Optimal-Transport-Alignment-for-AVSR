# 🚀 Hướng Dẫn Pipeline AVSR Tiếng Việt (ViCocktail) Toàn Diện

Tài liệu này hướng dẫn quy trình từ lúc có dữ liệu thô cho đến khi hoàn thành huấn luyện mô hình Audio-Visual Speech Recognition (AVSR) trên môi trường Modal.com.

---

## 📋 Mục lục
1. [Bước 1: Download (Tải dữ liệu)](#1-bước-1-download-tải-dữ-liệu)
2. [Bước 2: Unpack (Giải nén)](#2-bước-2-unpack-giải-nén)
3. [Bước 3: Preprocess (Mouth Cropping)](#3-bước-3-preprocess-mouth-cropping)
4. [Bước 4: Extract (Feature Extraction)](#4-bước-4-extract-feature-extraction)
5. [Bước 5: Training (Huấn luyện)](#5-bước-5-training-huấn-luyện)

---

## 1. Bước 1: Download (Tải dữ liệu)

Tải dữ liệu thô từ nguồn trực tiếp vào Modal Volume.

*   **Lệnh thực hiện:**
    ```bash
    modal run scripts/modal/download.py
    ```

---

## 2. Bước 2: Unpack (Giải nén)

Giải nén các file `.tar` để chuẩn bị cho quá trình xử lý hình ảnh và âm thanh.

*   **Lệnh thực hiện:**
    ```bash
    modal run scripts/modal/unpack_vicocktail_selective.py --stage unpack
    ```
    *(Lệnh này sẽ giải nén các file vào thư mục làm việc trên volume)*

---

## 3. Bước 3: Preprocess (Mouth Cropping)

Đây là bước quan trọng nhất của tiền xử lý hình ảnh: Phát hiện khuôn mặt và cắt lấy vùng miệng (Mouth Crop) để giảm nhiễu thông tin cho model.

*   **Lệnh thực hiện:**
    ```bash
    modal run scripts/modal/preprocess_vicocktail.py --stage crop
    ```
    *Kết quả là các video mouth crop được lưu vào volume `avsr-vicocktail-processed`.*

---

## 4. Bước 4: Extract (Feature Extraction)

Chuyển đổi các video mouth crop và file âm thanh thành các tensor đặc trưng (features) định dạng `.pt`.

*   **Lệnh thực hiện:**
    ```bash
    modal run scripts/modal/preprocess_vicocktail.py --stage extract
    ```
    *   **Audio Features:** Sử dụng Whisper để trích xuất Mel-spectrogram.
    *   **Visual Features:** Sử dụng ResNet-18/Conformer để trích xuất không gian.

---

## 5. Bước 5: Training (Huấn luyện)

Huấn luyện mô hình E2E với cơ chế **Pruned Vocabulary** (Bộ từ vựng rút gọn) để đạt tốc độ và độ chính xác cao nhất cho tiếng Việt.

*   **Lệnh huấn luyện:**
    ```bash
    modal run --detach scripts/modal/train_vicocktail.py
    ```

### 💡 Lưu ý về Training:
*   **Tự động Pruning:** Hệ thống sẽ tự động chạy `scripts/prune_whisper_vocab.py` khi build image trên Modal.
*   **Tiết kiệm bộ nhớ:** Model sử dụng vocab size ~8,000 (thay vì 51,000) giúp training nhanh hơn và tránh lỗi tràn bộ nhớ GPU.

---

## 📈 Theo dõi & Quản lý
*   **Logs:** Theo dõi tiến trình qua `modal logs <APP_ID>`.
*   **Checkpoints:** Tải model tốt nhất về máy local:
    ```bash
    modal volume get avsr-vicocktail-processed checkpoints/phase1/best_model.pt .
    ```

---
*Chúc bạn huấn luyện thành công!*
