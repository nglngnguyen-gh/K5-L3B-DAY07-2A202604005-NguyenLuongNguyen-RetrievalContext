# Phân tích lỗi dev và kết quả trước/sau

Số liệu lấy từ [dev_comparison.md](dev_comparison.md). Nhãn nằm trong `labels/dev/`, trace trong `starter/results/agent/<config>/dev/runs.jsonl`. Giả thuyết được ghi trước khi chạy, trong [selection_criteria.md](selection_criteria.md).

## Lỗi 1 — d02: bằng chứng $25 tỷ không vào context (lỗi retrieval và ranking)

- **Hiện tượng:** câu hỏi là "Nhóm WBG phát hành bao nhiêu bảo lãnh trong FY26 và so với mục tiêu 2030 thế nào?".
  - `dense-B-k3` tìm 3 lượt nhưng chỉ lấy được p18, p17, p38, p3, p4. Agent trả lời bằng **$17,9 tỷ** (của Guarantee Platform, p38) và nói "báo cáo không có mục tiêu 2030". Kết quả: **điểm 0**.
  - `bm25-B-k3` chỉ lấy được ý "vượt mục tiêu $20 tỷ" (p4), không có con số $25 tỷ: **điểm 1**.
- **Nguyên nhân:**
  - Có ba đoạn cùng chủ đề nhưng không chứa con số cần tìm: p4 ("vượt mục tiêu $20 tỷ"), p17 (infographic, con số $25B bị tách khỏi nhãn khi trích text), p18 (Platform tăng 46%). Chúng xếp trên p6.
  - Với query gốc, p6 đứng ở hạng 5 khi dùng dense và hạng 4 khi dùng hybrid ([trace retrieval](../starter/results/retrieval/dev/traces.jsonl)).
- **Giả thuyết:**
  - Hybrid kéo được p6 lên nhờ BM25 khớp cụm "2030 goal".
  - Top-5 đưa được p6 vào context ngay lượt đầu.
- **Trước và sau:**

  | Cấu hình | d02 vào context? | Điểm |
  |---|---|---|
  | dense-B-k3 (trước) | Không | 0 |
  | hybrid-B-k3 | Có (lượt 2) | 2 |
  | hybrid-B-k5 | Có (lượt 2) | 2 |
  | hybrid-B-k3-think | Có (lượt 2) | 2 |
  | hybrid-B-k5-think | **Không** (model viết query "total ... guarantees") | 1 |

- **Kết luận:** hybrid giúp được câu này, nhưng kết quả **không ổn định**: nó phụ thuộc vào cách model diễn đạt lại query. Tăng top-k lên 5 không đủ khi query viết lại kéo về đúng ba đoạn gây nhiễu.
  - Retrieval với query gốc cũng cho thấy điều này: khi agent tự viết lại query, Hit@1 của dense giảm từ 18/20 xuống 15/20.

## Lỗi 2 — u-d02 và u-d03: nội dung đã từ chối nhưng cờ `abstain=false` (lỗi điều khiển agent)

- **Hiện tượng:** với u-d02 (green bond của IBRD), cả 3 cấu hình ở bước 5 đều viết đúng ý "báo cáo không nêu; $2 tỷ là của IFC" nhưng gọi `final_answer` với `abstain=false`, nên status là `answered`. Hybrid cũng lặp lại lỗi này ở u-d03.
  - Nếu chấm theo nội dung, kết quả là 3/3; nếu chấm theo status thì chỉ được 1/3 đến 2/3.
- **Nguyên nhân:** khi tìm thấy thông tin **liên quan**, model coi việc nêu thông tin đó (kèm citation) là "trả lời". Prompt có quy tắc về câu trả lời một phần, và model đã áp dụng quy tắc này sai chỗ.
- **Giả thuyết:** cho model bước thinking (1024 token) sẽ giúp nó cân nhắc trước khi đặt cờ.
- **Trước và sau** (abstention đúng tính theo status, trên 3 câu):

  | Cấu hình | u-d01 | u-d02 | u-d03 | Tổng |
  |---|---|---|---|---|
  | hybrid-B-k3 (trước) | ✅ | ❌ | ❌ | 1/3 |
  | hybrid-B-k5 | ✅ | ❌ | ✅ | 2/3 |
  | hybrid-B-k3-think | ✅ | ❌ | ✅ | 2/3 |
  | hybrid-B-k5-think | ✅ | ✅ | ✅ | 3/3 |

- **Kết luận:** riêng thinking **không** sửa được u-d02, kết hợp mới sửa được. Tuy nhiên chỉ có 3 câu, và u-d03 đổi kết quả cả ở cấu hình không có thinking (k5), nên đây có thể là biến động ngẫu nhiên giữa các lần chạy. Bằng chứng còn yếu.
  - Hướng sửa chắc chắn hơn (chưa thử): thay cờ boolean bằng trạng thái `answered | partial | not_found` và kiểm tra bằng code, hoặc thêm một bước tự kiểm tra cờ trước khi trả lời.

## Lỗi 3 — d01: context có đủ evidence nhưng câu trả lời bỏ điều kiện

- **Hiện tượng:** ở **cả 6 cấu hình**, câu trả lời chỉ nói "từ 19 tháng xuống 9 tháng" mà bỏ ý "một số dự án rủi ro thấp chỉ mất 60 ngày", nên đều được **1/2**. Context luôn chứa p6:b002, đoạn có đủ câu này.
- **Nguyên nhân:** câu hỏi "giảm bao nhiêu" khiến model chỉ trả lời con số chính. Đoạn p5 (thư Chủ tịch, chỉ có "19 → 9") cũng thường đứng hạng 1.
- **Kết quả:** không cấu hình nào sửa được, kể cả có thinking. Đây là giới hạn ở khâu tạo câu trả lời, không phải retrieval.

## Đánh đổi chất lượng, thời gian và chi phí (dev, 24 câu)

| Cấu hình | Context Evidence Rate | Grounded TB | Abstention | Chi phí | Trễ trung vị | Token |
|---|---|---|---|---|---|---|
| dense-B-k3 | 20/21 | 1.86 | 2/3 | $0.0231 | 2.44 s | 51k |
| hybrid-B-k3 | 21/21 | 1.95 | 1/3 | $0.0226 | 2.46 s | 47k |
| hybrid-B-k5 | 21/21 | 1.95 | 2/3 | $0.0352 (+52%) | 2.34 s | 87k |
| hybrid-B-k3-think | 21/21 | 1.95 | 2/3 | $0.0394 (+70%) | 3.50 s | 56k |
| hybrid-B-k5-think | 20/21 | 1.90 | 3/3 | $0.0456 (+97%) | 3.50 s | 68k |

- **Top-5** làm gần như gấp đôi token input (context trung bình 3,5k so với 2,0k ký tự). Giả thuyết "ít lượt search hơn" **sai**: số lượt search trung bình tăng từ 1.17 lên 1.46.
- **Thinking** làm tăng độ trễ khoảng 1 s mỗi câu và chi phí khoảng 70%, nhưng giảm số lượt search (1.17 xuống 1.08).
- Mọi khác biệt về chất lượng đều chỉ 1–2 câu trên 21 hoặc 3 câu, nên **không đủ để khẳng định cải thiện**.
