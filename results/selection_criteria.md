# Tiêu chí chọn cấu hình và kế hoạch thử nghiệm (chốt trước bước 6)

Ghi ngày 2026-11-16, **trước khi chạy các cấu hình cải thiện** và trước mọi lần chạy test. Dữ liệu dùng để đặt giả thuyết là kết quả dev ở bước 5 ([dev_comparison.md](dev_comparison.md)).

## Tiêu chí chọn (chỉ dùng dev)

Xét lần lượt từng tiêu chí; chỉ chuyển sang tiêu chí sau khi hòa ở tiêu chí trước. **Chênh lệch ≤ 1 câu được coi là hòa**, vì với 21 câu có đáp án, 1 câu tương đương khoảng 5 điểm phần trăm, nằm trong sai số.

1. **Context Evidence Rate** cao hơn (đủ evidence trong context cuối).
2. **Grounded answer trung bình** cao hơn (nhãn thủ công 0/1/2).
3. **Abstention đúng** cao hơn (status `abstain` trên 3 câu thiếu bằng chứng), rồi **từ chối nhầm** thấp hơn.
4. **Chi phí** thấp hơn, rồi **độ trễ trung vị** thấp hơn.

Cấu hình baseline để so sánh trên test là `dense-B-k3` (dense retrieval Day 6). Sau khi chọn, cấu hình được khóa trong `starter/results/selection.json` và commit; test chỉ chạy **một lần** cho baseline và cấu hình đã chọn, không chỉnh sửa gì sau khi xem test.

## Thử nghiệm cải thiện (mỗi cấu hình chỉ đổi một yếu tố so với `hybrid-B-k3`)

| Cấu hình | Thay đổi | Giả thuyết | Lỗi dev nhắm tới | Đánh đổi dự kiến |
|---|---|---|---|---|
| `hybrid-B-k5` | top-k 3 → 5 | Evidence nằm ở hạng 4–5 sẽ vào context ngay ở lượt search đầu (retrieval với query gốc: EvHit@5 21/21, còn EvHit@3 19/21), giảm số lượt search và câu trả lời thiếu | d02 ($25 tỷ ở hạng 4–5), d21 (nhóm IFC) | Context dài hơn, nhiều token input hơn, nhiều đoạn gây nhiễu hơn |
| `hybrid-B-k3-think` | `thinking_budget` 0 → 1024 | Model có bước suy nghĩ sẽ đặt cờ `abstain` khớp với nội dung, và giữ đủ điều kiện trong câu trả lời | u-d02, u-d03 (nội dung từ chối nhưng `abstain=false`), d01 (bỏ điều kiện 60 ngày) | Token thinking tính giá output ($2,50/1M), trễ cao hơn |
| `hybrid-B-k5-think` | Kết hợp cả hai | **Chỉ chạy nếu cả hai thay đổi trên đều không làm giảm** chỉ số theo tiêu chí 1–3 | — | Cộng dồn chi phí của cả hai |

Các kết quả không cải thiện vẫn được báo cáo đầy đủ.
