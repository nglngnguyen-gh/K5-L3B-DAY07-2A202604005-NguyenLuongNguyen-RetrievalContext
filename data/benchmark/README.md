# Benchmark Day 7 — WBG Annual Report 2026 (`wbg-ar-2026-day7-v1`)

Corpus: *The World Bank Group Annual Report 2026* (World Bank Group, DOI 10.1596/AR2026, CC BY-NC-ND 3.0 IGO), 46 trang PDF, 41 trang được tính. Đây là cùng corpus, cùng chunk và cùng vector store với Day 6. Đơn vị trang là **trang PDF vật lý, đánh số từ 1**.

Benchmark được viết và kiểm chứng **trước mọi lần chạy retrieval hoặc agent của Day 7**. Hash SHA-256 của `questions.json` và `abstain.json` nằm trong `verification.json`; mọi thay đổi sau đó phải ghi lại ở đây.

## File

| File | Nội dung |
|---|---|
| `questions.json` | 30 câu có đáp án, gồm `id, split, type, origin, question, expected_answer, gold` hoặc `evidence_groups`, và `filter` nếu có |
| `abstain.json` | 6 câu thiếu bằng chứng, gồm `answerable: false`, `expected_behavior: "search_and_abstain"`, `absence_reason`, `absence_check` (từ khóa đã quét) và `near_topic_distractors` |
| `cases.json` | Danh sách `id, split, answerable` dùng cho `starter/agent_metrics.py`, sinh tự động |
| `verification.json` | Kết quả của `python -m src.verify_benchmark`, sinh tự động |

## Phân chia

| | Có đáp án | Thiếu bằng chứng | Tổng |
|---|---:|---:|---:|
| dev | 21 | 3 | 24 |
| test | 9 | 3 | 12 |

- **dev d01–d15** là dev Day 6, giữ nguyên câu hỏi và evidence. Riêng d04 có thêm evidence thay thế ở p5.
- **dev d16–d20** là **test Day 6 (t01–t05) chuyển sang dev**. Bộ test này đã bị xem khi phân tích Day 6, nên không còn là đánh giá độc lập; theo RULES, nó được dùng làm dữ liệu để chỉnh.
- **d21** là câu multi-hop mới, dùng để thử truy xuất nhiều bước trên dev.
- **test t01–t09** được **viết mới cho Day 7**, từ những trang không làm gold chính cho câu dev nào (p16, p17, p21, p22, p25, p28, p37, p41, p44). Hai trang trùng với dev chỉ là evidence thay thế, chứa thông tin khác: p27 là dòng tổng bảng của t01 (dev d17 dùng chú thích vùng của bảng này), p29 là bảng allocable income của t03 (dev d17 dùng chú thích ở đây).
- Loại câu test: 2 câu bảng, 2 câu có điều kiện, 1 câu có filter, 1 câu trực tiếp, 1 câu diễn đạt lại, 2 câu multi-hop. Test được thiết kế khó hơn test Day 6, vốn có điểm grounding tối đa ở mọi cấu hình (hiệu ứng trần).

## Định nghĩa hit và cách chấm

- **Khớp evidence:** `squash(evidence) in squash(chunk.text)`, trong đó `squash` gộp mọi khoảng trắng và xuống dòng thành một dấu cách, và chunk phải thuộc đúng `source_id` và `page`. Lý do: chunk của index A đã gộp khoảng trắng, còn index B giữ xuống dòng; Day 6 dùng cùng quy tắc.
- **Câu một evidence:** các phần tử trong `gold` là evidence **thay thế**, khớp một phần tử là đủ.
- **Câu multi-hop** (`evidence_groups`, gồm d21, t08, t09): **mọi nhóm đều bắt buộc**, trong mỗi nhóm khớp một phần tử là đủ.
  - Evidence Hit@3 và Context Evidence Rate chỉ tính là hit khi đủ mọi nhóm.
  - Hit@1 không áp dụng được, vì một chunk không thể chứa evidence từ hai trang. Vì vậy Hit@1 được báo **trên các câu một evidence**, còn câu multi-hop báo riêng mức "số nhóm tìm được / tổng số nhóm".
- **Câu thiếu bằng chứng** không vào mẫu số của các chỉ số retrieval. Chúng chỉ dùng cho chỉ số abstention: agent phải tìm kiếm rồi từ chối, và không được đưa ra khẳng định không có căn cứ.
- **Filter** (`{"section": ...}`) là ngữ cảnh người dùng cung cấp trước khi tìm, ví dụ "đang đọc chương X", được ghi trong `filter_context`. Filter không được suy ra từ gold hay đáp án.

## Kiểm chứng (`python -m src.verify_benchmark`)

1. Evidence nằm nguyên văn trên trang gold (bỏ qua khác biệt khoảng trắng), và trang đó là trang được tính.
2. Evidence có nằm trọn trong ít nhất một chunk của index A và index B không. Hiện có **1 cảnh báo**: evidence p28 của t03 bị cắt qua hai chunk ở index A (cửa sổ cố định 800 ký tự). Với index A, t03 chỉ hit được qua evidence thay thế ở p29. Đây là giới hạn thật của cách chunk, nên được giữ lại.
3. Giá trị filter có trong `sections` của trang gold.
4. Không trùng ID và không trùng câu hỏi.
5. Với câu thiếu bằng chứng, script quét các từ khóa `absence_check` trên toàn bộ 41 trang và in đoạn trích để **đọc tay**. Đã đọc và xác nhận cả 6 câu:
   - u-d01: không trang nào nêu ngân sách của CAO.
   - u-d02: chỉ có green bond của **IFC** ($2 tỷ, p19), không có của IBRD.
   - u-d03 và u-t01: các bảng chỉ có từ FY22 đến FY26.
   - u-t02: chỉ có số nhân sự toàn Nhóm (20,706) và ICSID (76).
   - u-t03: không nêu lãi suất hay spread cho vay.

## Rủi ro và giới hạn

- Câu hỏi, gold và nhãn đều do một người làm, có AI hỗ trợ soạn nháp, và đã đối chiếu với text trích từ PDF.
- Text trích bằng `pypdf` có lỗi thứ tự đọc trên 8/41 trang (xem báo cáo Day 6). Evidence được chọn từ các đoạn liền mạch, nhưng một số bảng như p15 Scorecard bị tách nhãn khỏi giá trị nên không dùng làm câu hỏi.
- Corpus chỉ có một tài liệu nên các chương lặp lại giữa dev và test (IBRD, MIGA, ICSID). Test chỉ có 12 câu: chênh lệch 1 câu tương đương khoảng 8–11 điểm phần trăm.
