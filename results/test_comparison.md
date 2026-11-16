# So sánh cấu hình — split `test`

| Cấu hình | Thiết lập | Hit@1 query gốc | EvHit@3 query gốc | Hit@1 query agent | EvHit@3 query agent | Context Evidence Rate | Grounded 0/1/2 · TB | Citation hợp lệ | Abstention đúng | Từ chối nhầm | Lượt search TB {phân bố} | Trễ trung vị (s) | Token | Chi phí | Lỗi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense-B-k3 | dense, index B, top-3, filter on, budget 6000 chars, ≤4 searches | 6/7 (86%) | 8/9 (89%) | 7/7 (100%) | 7/9 (78%) | 9/9 (100%) | 0/1/8 · 1.89 | 13/13 (100%) | 2/3 (67%) | 0/9 (0%) | 1.25 {'1': 10, '2': 1, '3': 1} | 2.78 | 29,734 | $0.0131 | 0 |
| hybrid-B-k5 | hybrid, index B, top-5, filter on, budget 6000 chars, ≤4 searches | 5/7 (71%) | 8/9 (89%) | 6/7 (86%) | 7/9 (78%) | 9/9 (100%) | 0/1/8 · 1.89 | 15/15 (100%) | 2/3 (67%) | 0/9 (0%) | 1.42 {'1': 9, '2': 2, '4': 1} | 2.76 | 44,620 | $0.0180 | 0 |

Hit@1 tính trên câu một evidence; EvHit@3 và Context Evidence Rate tính trên mọi câu có đáp án (multi-hop cần đủ mọi nhóm). Grounded, citation và abstention lấy từ nhãn thủ công (`labels/`), tổng hợp bằng `starter/agent_metrics.py`. Chi phí tính theo token × đơn giá trong config (Gemini 2.5 Flash $0.30/1M input, $2.50/1M output), không phải hóa đơn.
