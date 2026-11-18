# Báo cáo Day 09 — Nguyễn Viết Tiến, 03140

## 1. Kết luận trước
Pipeline ra quyết định **Hold** (mã thoát 10). Đầu vào có 224 ảnh, bước nhận dữ liệu loại 21 ảnh, còn 203 ảnh được xuất bản nháp ra submission/export/clean_v1/. Lý do chưa Release được là dù đã hết lỗi bắt buộc (0 lỗi), vẫn còn 38/203 ảnh (18.7%) có cảnh báo chưa được duyệt, cao hơn mức 10% cho phép.

## 2. Lỗi và phân việc
Số liệu lấy từ 02_error_list_v1.csv:

| Mã lỗi | Số dòng / ảnh | Ai sửa | Vì sao |
|---|---|---|---|
| CORRUPT_IMAGE | 2 / 2 ảnh | Máy (intake) | Ảnh lỗi không mở được |
| BAD_SIZE | 3 / 3 ảnh | Máy (intake) | Cạnh ngắn < 250px |
| DUP_IMAGE_EXACT | 8 / 8 ảnh | Máy (intake) | Trùng md5, giữ ảnh đầu |
| DUP_IMAGE_NEAR | 3 / 3 ảnh | Máy (intake) | Trùng gần đúng (pHash <= 8) |
| LEAK_TEST | 5 / 5 ảnh | Máy (intake) | Trùng với tập test |
| BAD_CLASS_NAME | 2 / data.yaml | Máy (autofix) | Sửa lại tên lớp đúng thứ tự |
| BAD_CLASS_ID | 6 / 6 ảnh | Máy (autofix) | Xoá dòng có ID ngoài 0..3 |
| BAD_FORMAT | 3 / 3 ảnh | Máy (autofix) | Xoá dòng thiếu cột hoặc nan |
| BOX_ZERO | 4 / 4 ảnh | Máy (autofix) | Xoá box có w hoặc h <= 0 |
| DUP_BOX | 7 / 7 ảnh | Máy (autofix) | Xoá box trùng lặp cùng lớp |
| BOX_OUT_OF_IMAGE | 8 / 8 ảnh | Máy (autofix) | Cắt box về trong [0, 1] |
| MISSING_LABEL_FILE | 6 / 6 ảnh | Người | Máy không tự gán nhãn được |
| EMPTY_LABEL | 12 / 12 ảnh | Người | Xem lại ảnh nền hay bị thiếu nhãn |
| BOX_TINY | 6 / 4 ảnh | Người | Box nhỏ hơn 12px |
| BOX_HUGE | 3 / 3 ảnh | Người | Box quá lớn (> 95% ảnh) |
| CLASS_RARE | 4 / dataset | Người | Lớp ít mẫu |
| POSSIBLE_MISSING | 53 / 52 ảnh | Người | Model thấy vật nhưng nhãn thiếu |
| POSSIBLE_FALSE_POS | 29 / 27 ảnh | Người | Box nhãn không khớp dự đoán nào |
| POSSIBLE_LOOSE_BOX | 28 / 24 ảnh | Người | Đúng lớp nhưng IoU < 0.5 |
| POSSIBLE_WRONG_CLASS | 18 / 18 ảnh | Người | Model đoán khác lớp với nhãn |

Hàng đợi 05_review_queue.csv có 125 ảnh, được xếp theo tổng trọng số lỗi. Những ảnh vừa bị máy xoá dòng vừa có lỗi ngữ nghĩa (như img_0020, điểm 11.524) và 6 ảnh thiếu file nhãn (NEEDS_LABEL, điểm 10.0) được đẩy lên trên cùng. Với ngân sách 80 ảnh, người duyệt xem được 6 ảnh thiếu file nhãn và 74 ảnh lỗi ngữ nghĩa.

## 3. Calibration (2 ngưỡng)
| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do |
|---|---|---|---|---|
| duplicates.phash_dist | 2 | 8 | DUP_IMAGE_NEAR: 1.00 / 0.50 → 1.00 / 0.67 | Bắt thêm được ảnh trùng gần đúng mà precision vẫn 1.00; không chọn 20 vì làm recall của DUP_IMAGE_EXACT tụt xuống 0.33. |
| second_opinion.conf | 0.25 | 0.6 | WRONG_CLASS: 0.56 / 0.83 → 0.62 / 0.83; MISSING: 0.50 / 1.00 → 0.57 / 1.00 | Giảm bớt báo động giả ở WRONG_CLASS và MISSING trong khi vẫn giữ recall của MISSING ở 1.00 (lên 0.7 thì recall bắt đầu giảm). |

Điểm dễ đánh lừa: bộ calib chỉ có 39 ảnh nên sai khác 1–2 ảnh là P/R nhảy khá mạnh; calib cũng không có ca LEAK_TEST và lỗi chèn chủ yếu là person ↔ car (khác với lỗi thực tế trên raw hay nhầm giữa car/truck/bus). Kiểm tra 01_intake.csv trên raw với phash_dist = 8 thì thấy loại đúng 3 ảnh trùng gần đúng (img_0076, img_0143 có dist=0; img_0223 có dist=6).

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
Thử chạy `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb`: vì không có người duyệt nên còn nguyên 6 lỗi MISSING_LABEL_FILE, gate trả về Rework, mã thoát 20 và không xuất dataset ra work/exp_nb.

Hai điểm mù mà pipeline hiện tại bỏ qua:
- Ảnh trùng nhưng bị cắt/xoay mạnh làm pHash lệch quá ngưỡng 8 thì bước intake không bắt được.
- Các cặp nhãn và dự đoán khác lớp có IoU nằm trong khoảng [0.3, 0.5) bị bỏ qua không cảnh báo; ngoài ra nếu chính yolo26l cũng nhận sai hoặc conf < 0.6 thì second_opinion cũng chịu.

## 5. Train: raw vs clean
Train trên Colab (T4 GPU) với yolo26n, 20 epoch, 2 seed (0 và 1), đo trên gold_val (40 ảnh) và test (100 ảnh):

| biến thể | #ảnh train | số seed | map50 (TB ± độ lệch) | map50_95 (TB ± độ lệch) | precision (TB ± độ lệch) | recall (TB ± độ lệch) |
|---|---|---|---|---|---|---|
| raw | 224 | 2 | 0.726 ± 0.000 | 0.512 ± 0.004 | 0.764 ± 0.061 | 0.649 ± 0.028 |
| clean | 203 | 2 | 0.699 ± 0.025 | 0.486 ± 0.013 | 0.796 ± 0.050 | 0.596 ± 0.049 |

Chênh mAP50 (clean − raw) = -0.027.

Độ chênh -0.027 này gần bằng độ lệch giữa 2 seed của tập clean (± 0.025), nên với 2 seed và 100 ảnh test thì chưa thể kết luận được bên nào tốt hơn bên nào (clean có precision nhỉnh hơn nhưng recall lại thấp hơn). Có 2 nguyên nhân hợp lý cần kiểm thêm:
1. Tập raw bị dính 5 ảnh trùng với tập test (LEAK_TEST) nên mAP của raw bị kéo lên giả tạo. Để kiểm tra thì có thể bỏ 5 ảnh này khỏi raw rồi train lại.
2. Tập clean mới ở mức Hold (còn 38 ảnh cảnh báo chưa duyệt) và ít hơn raw 21 ảnh. Có thể duyệt nốt số ảnh còn lại và chạy thêm vài seed để xem kết quả có ổn định hơn không.

## 6. Quyết định và ghi chú AI
Quyết định cuối là **Hold**. Muốn thành Release thì cần tăng ngân sách duyệt thêm ít nhất 19 ảnh nữa để đưa số ảnh chưa duyệt xuống dưới 20 ảnh (< 10% của 203 ảnh).

Ghi chú dùng AI:
- Prompt: đưa mô tả đầu vào/ra từ docstring của 4 module kèm các ca biên (dòng trống, nhiều lỗi trên cùng dòng) để viết code.
- Cách kiểm: chạy toàn bộ unit test `pytest` và chạy thử `pipeline.run` kiểm tra lại file csv xuất ra.
- Chỗ AI làm sai: lúc đầu autofix bị lệch số dòng khi gặp dòng trống và ghi đè cả clip_box lên dòng đã drop_line, sau đó tôi sửa lại và chạy test pass hết.
