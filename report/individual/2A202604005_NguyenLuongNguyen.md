# Báo cáo Day 7 — Agentic RAG

**Học viên:** Nguyễn Lương Nguyên · 2A202604005 · L3B
**Corpus:** The World Bank Group Annual Report 2026 (tiếp nối Day 6) · **Ngày chạy:** 2026-11-16

**Tóm tắt.**
- Tôi xây agent dùng `gemini-2.5-flash` có function calling, tìm bằng chứng trong vector store Day 6 (PostgreSQL + pgvector, `gemini-embedding-001` 768 chiều).
- So sánh 3 phương pháp retrieval: dense, BM25 và hybrid RRF. Thử thêm 3 cấu hình cải thiện trên dev.
- Quy tắc chọn công bố trước đã chọn lại **baseline dense**, vì không cải thiện nào hơn baseline quá 1 câu, trong khi baseline rẻ nhất.
- Trên test (12 câu mới, chạy một lần): Context Evidence Rate 9/9, grounded 1,89/2, citation hợp lệ 13/13, abstention 2/3, từ chối nhầm 0/9.
- Lỗi còn lại nằm ở **khâu sinh câu trả lời** (bỏ điều kiện, đặt sai cờ abstain), không phải ở retrieval.

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - *The World Bank Group Annual Report 2026* (DOI 10.1596/AR2026), giấy phép **CC BY-NC-ND 3.0 IGO**. Chi tiết nguồn, SHA-256 của PDF và quy tắc tính trang nằm trong `data/manifest.json` của Day 6.
  - PDF có 46 trang, trong đó 41 trang nội dung được tính (khoảng 183k ký tự).
  - Dùng lại chunk Day 6: index **A** (cửa sổ cố định 800/100 ký tự, 276 chunk) và index **B** (theo đoạn văn, tối đa 800 ký tự, 331 chunk). Các thí nghiệm agent dùng index B.
  - Text, chunk và trace có chứa text đều **không commit**, vì giấy phép ND không cho phát hành tác phẩm phái sinh. Người chấm tái tạo bằng `python -m src.export_day6` từ DB Day 6.
- **Vector store, embedding model, phiên bản và số chiều:**
  - PostgreSQL 16 + **pgvector 0.8.7** chạy trong Docker của Day 6. Tìm cosine exact, không dùng ANN.
  - **`gemini-embedding-001`**, cắt còn 768 chiều rồi chuẩn hóa L2. Tài liệu được embed với `RETRIEVAL_DOCUMENT` (ở Day 6), câu hỏi với `RETRIEVAL_QUERY`. Cấu hình này giống hệt Day 6, và code kiểm tra model và số chiều của index khi khởi tạo.
  - Vector câu hỏi được cache trong bảng `embedding_cache` của Day 6. Toàn bộ Day 7 chỉ tốn khoảng 20 lần gọi embedding, chi phí không đáng kể (dưới $0,001).
- **Agent model, cách chạy và phân chia dev/test:**
  - **`gemini-2.5-flash`** gọi qua API `v1beta`, temperature 0; thinking tắt, trừ các cấu hình `-think`.
  - Lệnh chạy: `python -m src.run_agent --config configs/<cfg>.json --split dev|test`.
  - Split (chi tiết trong [data/benchmark/README.md](../../data/benchmark/README.md)):
    - **Dev:** 21 câu có đáp án và 3 câu thiếu bằng chứng. Gồm 15 câu dev Day 6, **5 câu test Day 6 chuyển sang dev** (đã bị xem khi phân tích Day 6 nên không còn độc lập), và 1 câu multi-hop mới.
    - **Test:** **viết mới**, 9 câu có đáp án và 3 câu thiếu bằng chứng, chỉ lấy từ các trang không làm gold chính cho câu dev nào. Gồm 2 câu bảng, 2 câu có điều kiện, 1 câu có filter, 1 câu trực tiếp, 1 câu diễn đạt lại, 2 câu multi-hop.
  - Benchmark được commit kèm hash (`5c787f5`) **trước khi có code retrieval**.
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - `src.verify_benchmark` kiểm tra evidence nằm nguyên văn trên trang gold (bỏ qua khác biệt khoảng trắng), có nằm trọn trong một chunk A/B không, filter có khớp không, và có trùng câu hỏi không.
  - Với câu thiếu bằng chứng, script quét từ khóa trên toàn bộ 41 trang và in đoạn trích; tôi đọc tay từng đoạn để xác nhận thông tin thật sự không có.
    - Ví dụ u-d02 hỏi về green bond của IBRD: chỉ có green bond của **IFC** ($2 tỷ, p19). Đây là bẫy nhầm tổ chức.
  - **Hit** = chunk đúng trang và chứa evidence. Câu multi-hop (`evidence_groups`) phải **đủ mọi nhóm**, nên Hit@1 chỉ tính trên câu một evidence. Câu thiếu bằng chứng không vào mẫu số của các chỉ số retrieval.

## Retrieval và thử nghiệm

Ba phương pháp nằm trong [src/retrieval.py](../../src/retrieval.py):
- **Dense:** pgvector cosine.
- **BM25:** Okapi, k1=1,5, b=0,75. Tokenizer giữ nguyên số (`50,177` và thêm dạng `50177`).
- **Hybrid:** RRF, gộp 20 candidate dense và 20 candidate BM25 với k=60, chỉ dùng rank nên không cộng trực tiếp hai loại score.

Mọi cấu hình dùng cùng corpus, index B, filter `section` do người dùng cung cấp (4 câu dev, 1 câu test), context budget 6000 ký tự và tối đa 4 lượt search.

Cách đọc các cột:
- **Hit@1 và Evidence Hit@3** tính trên **query gốc** (`src.eval_retrieval`).
- **Context Evidence Rate** tính trên context cuối mà agent gửi cho model, gồm mọi lượt search.
- **Thời gian** là trung vị mỗi câu của agent; **chi phí** là tổng của cả split, tính bằng token × đơn giá niêm yết $0,30/1M input và $2,50/1M output (gồm cả thinking). Đây là ước tính, không phải hóa đơn.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline `dense-B-k3` | dense, top-3, filter on, không rerank, 6000 ký tự | Dense trên vector store Day 6 (giống strategy C của Day 6) | dev | 18/20 (90%) | 20/21 (95%) | 20/21 (95%) | 2,44 s · $0,0231 |
| Phương pháp 2 `bm25-B-k3` | BM25, top-3, filter on | Khớp chính xác số và tên riêng | dev | 18/20 (90%) | 19/21 (90%) | 19/21 (90%) | 2,23 s · $0,0242 |
| Phương pháp 3 `hybrid-B-k3` | Hybrid RRF (dense 20 + BM25 20), top-3, filter on | Kết hợp ngữ nghĩa và khớp từ khóa | dev | **19/20 (95%)** | 19/21 (90%) | **21/21 (100%)** | 2,46 s · $0,0226 |
| Cải thiện 1 `hybrid-B-k5` | Hybrid, **top-5** | Evidence ở hạng 4–5 vào context ngay lượt đầu (với query gốc, EvHit@5 = 21/21) | dev | 19/20 | 19/21 (EvHit@5 21/21) | 21/21 | 2,34 s · $0,0352 |
| Cải thiện 2 `hybrid-B-k3-think` | Hybrid, top-3, **thinking 1024** | Đặt cờ abstain khớp với nội dung, giữ đủ điều kiện | dev | 19/20 | 19/21 | 21/21 | 3,50 s · $0,0394 |
| Cải thiện 3 `hybrid-B-k5-think` | Hybrid, top-5 kết hợp thinking (đã khai báo chạy nếu 1 và 2 không làm giảm chỉ số) | Cộng dồn hai hiệu ứng | dev | 19/20 | 19/21 | 20/21 | 3,50 s · $0,0456 |
| **Cấu hình đã chọn** `dense-B-k3` | như baseline | Quy tắc chọn trên dev, xem bên dưới | test | 6/7 (86%) | 8/9 (89%) | 9/9 (100%) | 2,78 s · $0,0131 |
| Baseline `dense-B-k3` | trùng với cấu hình đã chọn | — | test | 6/7 | 8/9 | 9/9 | 2,78 s · $0,0131 |
| Nhánh đối chứng `hybrid-B-k5` | khai báo trước khi chạy test, không ảnh hưởng lựa chọn | — | test | 5/7 (71%) | 8/9 (EvHit@5 9/9) | 9/9 (100%) | 2,76 s · $0,0180 |

Ghi chú thêm:
- **Phương pháp mới đồng thời là cải thiện:** hybrid là phương pháp 3, đồng thời là nền cho cả ba cải thiện. Mỗi cải thiện chỉ đổi một yếu tố so với `hybrid-B-k3`.
- **Index A và B:** retrieval với query gốc trên index A cho dense/BM25/hybrid đều đạt 18/20 và 20/21; B gần như ngang (bảng 12 cấu hình trong `starter/results/retrieval/dev/summary.json`). Tôi chọn B vì chunk liền mạch hơn (coherence 1,8 so với 1,0 ở Day 6), quan trọng khi model đọc context. Ở index A, evidence của t03 bị cắt qua hai chunk.
- **Query viết lại:** agent tự viết lại câu hỏi thành chuỗi từ khóa. Kết quả lượt search đầu tiên được báo riêng trong [results/dev_comparison.md](../../results/dev_comparison.md):
  - Dense: Hit@1 giảm từ **18/20 xuống 15/20**.
  - BM25: Hit@1 tăng từ 18/20 lên 19/20.

  Vậy việc viết lại query có lợi cho BM25 nhưng có hại cho dense.
- **BM25 trên test** được chạy thêm để tham khảo, chỉ ở mức retrieval, không nằm trong protocol test: Hit@1 6/7, Evidence Hit@3 7/9.

**Hai lỗi dev, nguyên nhân và cách thử** (chi tiết, kèm lỗi thứ ba d01, trong [results/dev_error_analysis.md](../../results/dev_error_analysis.md)):

1. **d02** (bảo lãnh $25 tỷ, p6):
   - Dense đưa lên trước ba đoạn cùng chủ đề nhưng không có con số cần tìm: p4 ("vượt mục tiêu $20 tỷ"), p17 (infographic, con số bị tách khỏi nhãn khi trích text), p18. Agent trả lời bằng $17,9 tỷ của Guarantee Platform: **điểm 0**.
   - Hybrid đưa được p6 vào context, **điểm 2** ở hybrid-k3, k5 và k3-think.
   - Nhưng ở k5-think, model viết query "total … guarantees", nên p6 lại trượt (điểm 1).
   - Kết luận: hybrid giúp được câu này nhưng phụ thuộc vào cách model viết query, nên **chưa ổn định**.
2. **Cờ abstain** (u-d02, u-d03):
   - Nội dung câu trả lời từ chối đúng ("không nêu; $2 tỷ là của IFC"), nhưng model gọi `final_answer` với `abstain=false`.
   - Abstention của hybrid-k3 tính theo status chỉ được 1/3, dù tính theo nội dung là 3/3.
   - Thinking (k3-think) cho 2/3; kết hợp (k5-think) cho 3/3. Tuy nhiên riêng thinking không sửa được u-d02, và chỉ có 3 câu, nên bằng chứng yếu.

**Lý do chọn cấu hình** (quy tắc trong [results/selection_criteria.md](../../results/selection_criteria.md), commit `50da184` trước khi chạy cải thiện; lựa chọn khóa trong `starter/results/selection.json`, commit `4c6a109` trước khi chạy test). Quy tắc xét lần lượt các tiêu chí, với chênh lệch ≤ 1 câu so với cấu hình tốt nhất được coi là hòa:
1. **Context Evidence Rate:** loại BM25 (19/21, kém cấu hình tốt nhất 2 câu).
2. **Grounded:** tất cả còn lại hòa.
3. **Abstention:** loại hybrid-k3 (1/3, kém k5-think 3/3 hai câu).
4. **Chi phí:** dense $0,0231 rẻ nhất, so với k5 $0,0352, k3-think $0,0394 và k5-think $0,0456.

→ **Chọn `dense-B-k3`**. Các cải thiện không được chọn vì trên 24 câu dev chúng chỉ hơn baseline tối đa 1 câu ở mỗi tiêu chí, trong khi chi phí cao hơn 52–97%. Nhánh `hybrid-B-k5` (cấu hình quy tắc sẽ chọn nếu loại baseline ra) được chạy trên test để đối chứng.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi** (xem [src/agent.py](../../src/agent.py)):
  - **Tool:** `search_documents(query)` chạy phương pháp và top-k trong config. `read_neighbors(chunk_id)` có sẵn nhưng tắt trong mọi cấu hình đã đo. `final_answer(answer, citations, abstain)` là cách duy nhất để kết thúc.
  - **Gọi tool:** function calling chế độ `mode=ANY`, nên mỗi lượt model bắt buộc gọi một tool. Filter section do hệ thống truyền vào từ ngữ cảnh người dùng; model không tự đặt filter. Gold không bao giờ đi vào agent.
  - **Context:** bỏ trùng theo chunk id (chunk đã gửi chỉ báo "already shown"). Budget 6000 ký tự, chunk vượt budget bị giữ lại và được log. Context cuối là mọi chunk đã gửi cho model.
  - **Citation:** mỗi id phải nằm trong context và được ghi dạng `[id]` trong câu trả lời (chấp nhận `[a, b]`). Sai thì model có **1 lượt sửa**; vẫn sai thì tính là abstain.
  - **Điều kiện dừng:** model gọi `final_answer`; đạt tối đa 4 lượt search (model được nhắc trả lời ngay); tối đa 10 lượt gọi model.
  - **Lỗi:** 429/5xx và lỗi mất kết nối được retry. Lỗi khác ghi `status: error`, không có kết quả giả.
  - **Lỗi đã gặp và cách xử lý trong smoke test:**
    - SDK gọi API `v1`, nơi không hỗ trợ function calling (lỗi 400). Đã chuyển sang `v1beta`.
    - `WinError 10054` (mất kết nối) chưa được retry. Đã thêm retry cho lỗi transport.
    - Model viết citation dạng `[id1, id2]`. Validator nay chấp nhận nhiều id trong một cặp ngoặc.
  - Prompt quy định nội dung tài liệu là dữ liệu, không phải chỉ dẫn; agent phải kiểm tra đúng tổ chức, năm và đơn vị; không đủ bằng chứng thì từ chối.

Các chỉ số dưới đây tổng hợp bằng `starter/agent_metrics.py` từ nhãn thủ công trong [labels/](../../labels/):

| Cấu hình / split | Tool use | Số lượt gọi (search TB {phân bố}) | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| dense-B-k3 / dev | 24/24 | 1.25 {1:20, 2:2, 3:2} | 1/1/19 · 1.86 | 28/28 | 22/22 | 2/3 | 0/21 |
| bm25-B-k3 / dev | 24/24 | 1.29 {1:20, 2:1, 3:3} | 0/3/18 · 1.86 | 30/33 | 22/22 | 2/3 | 0/21 |
| hybrid-B-k3 / dev | 24/24 | 1.17 {1:21, 2:2, 3:1} | 0/1/20 · 1.95 | 34/34 | 23/23 | 1/3 | 0/21 |
| hybrid-B-k5 / dev | 24/24 | 1.46 {1:18, 2:2, 3:3, 4:1} | 0/1/20 · 1.95 | 35/35 | 22/22 | 2/3 | 0/21 |
| hybrid-B-k3-think / dev | 24/24 | 1.08 {1:22, 2:2} | 0/1/20 · 1.95 | 35/35 | 22/22 | 2/3 | 0/21 |
| hybrid-B-k5-think / dev | 24/24 | 1.08 {1:23, 3:1} | 0/2/19 · 1.90 | 34/34 | 21/21 | 3/3 | 0/21 |
| **dense-B-k3 / test** | 12/12 | 1.25 {1:10, 2:1, 3:1} | 0/1/8 · **1.89** | 13/13 | 10/10 | 2/3 | 0/9 |
| hybrid-B-k5 / test | 12/12 | 1.42 {1:9, 2:2, 4:1} | 0/1/8 · 1.89 | 15/15 | 10/10 | 2/3 | 0/9 |

Cách đọc bảng:
- "Câu có citation" = số câu có status `answered` và có citation / số câu có status `answered`. Mỗi câu gọi model từ 2 đến 7 lần (`model_calls` trong trace).
- 3 citation không hợp lệ của BM25 trên dev: một citation ở d21 gắn vào khẳng định sai "không có số của IFC", và hai citation ở u-d03 gắn vào nhận xét sai "tài liệu chỉ có FY25–26".
- **Điểm context trên test** (đọc tay cả 9 câu có đáp án; 0/1/2 = không có / có nhưng thiếu điều kiện hoặc nguồn / đủ):
  - dense-B-k3: **0/1/8**. t01 chỉ có bảng ngành p27 với cùng số liệu, không có Table 1 mà câu hỏi nêu tên.
  - hybrid-B-k5: **0/0/9**.

Ví dụ trên test (bảng đầy đủ 24 dòng: [results/test_results.md](../../results/test_results.md)):

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| t08 (dense) | multi-hop, câu có bằng chứng | 2 lượt: `IBRD commitments to Ukraine fiscal 2026` → p10, **p28**, p27; `IFC concessional finance Ukraine fiscal 2026` → **p25**, p25, p10 | "IBRD committed $4,210 million to Ukraine [p028:b002]. IFC trust funds have received over $1 billion in concessional finance commitments for Ukraine [p025:b006]." | Context 2, answer 2, citation 2/2. Agent tự tách câu thành hai lượt search, mỗi lượt lấy một nhóm evidence: dòng bảng Table 7 (p28) và câu ở p25 |
| u-t02 (dense) | thiếu bằng chứng | 1 lượt: `MIGA full-time staff` → p38, p25, p40 | "the document does not contain information about the number of full-time staff employed by MIGA." (abstain, không citation) | Abstention ✅. Báo cáo chỉ có tổng nhân sự toàn Nhóm (20,706) và ICSID (76) |
| u-t03 (dense) | thiếu bằng chứng, **lỗi** | 1 lượt: `IBRD interest rate lending spread fiscal 2026` → p28, p29, p28 | "…does not specify the exact interest rate or lending spread… [p028:b004]" với `abstain=false` | Abstention ❌. Nội dung đúng nhưng cờ sai, cùng kiểu lỗi với u-d02 trên dev |
| t03 (cả hai nhánh) | có filter, **lỗi một phần** | dense 1 lượt (p28, p29, p29); hybrid 4 lượt lặp gần như cùng một query | Liệt kê đúng $2,3 tỷ, $1,1 tỷ, $750 triệu, $500 triệu [p028:b005] | Answer 1. Bỏ điều kiện: chỉ $1,1 tỷ là "đã duyệt", còn IDA và Surplus là "được đề xuất". Cùng kiểu lỗi với d01 trên dev |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - **Trace đầy đủ:** `starter/results/agent/<config>/<split>/runs.jsonl` (tool call, tham số, kết quả, context, câu trả lời, citation, token, độ trễ, lỗi), cùng `traces.json` theo định dạng của `agent_metrics.py`.
  - Trace có chứa text tài liệu nên bị gitignore; tái tạo bằng các lệnh trong [README](../../README.md#bài-nộp--nguyễn-lương-nguyên-2a202604005).
  - **Đã commit:** nhãn trong `labels/`, bảng tổng hợp trong `results/`, và `starter/results/selection.json`.
  - **Thời gian:** trung vị khoảng 2,3–3,5 s mỗi câu. Thỉnh thoảng có câu mất 20–70 s do phía API; con số này được giữ nguyên trong trace.
  - **Tổng chi phí agent cho toàn bộ Day 7:** 376 lần gọi model, khoảng 454k token, **khoảng $0,23**, gồm cả smoke test. Embedding câu hỏi dưới $0,001.
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - **Trước/sau:** xem phần Retrieval và thử nghiệm ở trên, cùng [results/dev_error_analysis.md](../../results/dev_error_analysis.md).
  - **Đánh đổi:**
    - Top-5 gần gấp đôi lượng token input (context trung bình 3,5k so với 2,0k ký tự). Giả thuyết "ít lượt search hơn" **sai**: số lượt search tăng từ 1,17 lên 1,46.
    - Thinking làm tăng độ trễ khoảng 1 s mỗi câu và chi phí khoảng 70%.
    - Trên test, hybrid-k5 có context tốt hơn ở 1 câu nhưng câu trả lời không tốt hơn, và tốn thêm 38% chi phí.
  - **Kết luận:** retrieval không còn là nút thắt. Context có đủ evidence ở 9/9 câu test, kể cả câu bảng và multi-hop. Hai kiểu lỗi lặp lại giữa dev và test là **bỏ điều kiện khi tóm tắt** (d01, t03) và **cờ abstain không khớp với nội dung** (u-d02, u-t03). Muốn cải thiện tiếp nên sửa ở khâu sinh câu trả lời:
    - Thay cờ boolean bằng trạng thái `answered | partial | not_found` và kiểm tra bằng code.
    - Thêm bước tự kiểm tra rằng mọi điều kiện và vế của câu hỏi đều có trong câu trả lời.
  - **Giới hạn:**
    - Chỉ có một tài liệu, 24 câu dev và 12 câu test: 1 câu tương đương 5–11 điểm phần trăm, nên mọi chênh lệch giữa các cấu hình đều nằm trong mức nhiễu.
    - Chỉ có 3 câu thiếu bằng chứng mỗi split.
    - Mỗi cấu hình chỉ chạy **một lần**. Temperature 0 nhưng kết quả vẫn dao động giữa các lần chạy (ví dụ d02 ở k5-think), và tôi chưa đo độ dao động này.
    - Câu hỏi, gold và nhãn do một người làm, có AI chấm nháp; người chấm biết câu trả lời thuộc cấu hình nào.
    - Lỗi thứ tự đọc của `pypdf` (8/41 trang, ghi nhận từ Day 6) vẫn còn, và infographic (p17) mất liên kết giữa con số và nhãn.
    - Chi phí là ước tính từ token và đơn giá niêm yết, chưa đối chiếu hóa đơn.

### Công cụ AI và cách kiểm tra

Tôi dùng Claude Code (Claude Opus 5.5) để viết code, soạn nháp benchmark test mới và chấm nháp toàn bộ nhãn (cột `labeler`). Cách tôi kiểm tra:
- Evidence được kiểm chứng tự động trên trang gold. Câu thiếu bằng chứng được quét từ khóa trên toàn corpus, và tôi đọc tay từng đoạn trích.
- Benchmark, tiêu chí chọn và lựa chọn đều được commit **trước** bước dùng đến chúng: `5c787f5`, `50da184`, `4c6a109`.
- Mọi nhãn dev và test do tôi duyệt lại (cột `reviewed_by`), đối chiếu với toàn văn chunk được cite qua `src.review_sheet`.
- Unit test offline: 22 test (`PYTHONUTF8=1 python -m unittest discover -s tests`), không thay cho benchmark thật.

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện. Tài liệu chỉ có 46 trang PDF (87 trang in), chưa đạt yêu cầu từ 100 trang.
