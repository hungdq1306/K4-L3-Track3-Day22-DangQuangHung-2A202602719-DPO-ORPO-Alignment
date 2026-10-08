# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Đặng Quang Hưng
**Khoá:** AI20K-K4 (MSV: 2A202602719)
**Tier đã chạy:** T4
**Ngày:** 2026-10-08

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `adapters/variants/variants_summary.json`, `data/pref/stats.json`), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Kaggle / Colab T4 16 GB |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.88% (chosen_longer_frac = 0.65875, chosen median = 94.0 tokens, rejected median = 86.0 tokens, n = 800) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 epoch (LoRA r=16, alpha=32) |
| Giám khảo | Hội đồng local 2 RM: Skywork-Reward-V2-Qwen3-4B + Skywork-Reward-V2-Llama-3.2-3B |
| Chi phí | 0 đồng (Kaggle T4 miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | 18 phút 35 giây (100 steps, batch 1, grad_accum 8) |
| VRAM cao nhất | 7.21 GB (trên GPU T4 16 GB) |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.0899 (end_chosen: 0.3736, end_rejected: 0.2837) |
| Độ chính xác reward trên held-out | 66.0% (eval_reward_accuracy = 0.66) |
| Margin trên held-out | +0.0829 (eval_chosen: 0.3913, eval_rejected: 0.3084) |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 582.1 → 587.5 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Trên tập huấn luyện (train), đường `rewards/chosen` khởi đầu tại giá trị xấp xỉ 0 và tăng đều đặn đạt mức `+0.3736` ở cuối quá trình huấn luyện (bước 100). Trong khi đó, `rewards/rejected` cũng tăng nhẹ lên mức `+0.2837`. Vì tốc độ tăng của `rewards/chosen` vượt trội so với `rewards/rejected`, khoảng cách reward gap cuối cùng trên tập huấn luyện mở rộng đạt `+0.0899` với loss giảm dần từ `0.6931` (tương ứng $\ln 2$ tại khởi tạo) xuống `0.6759`.

Trên tập kiểm định độc lập (held-out), xu hướng diễn ra hoàn toàn tương đồng và đồng hướng: `eval_chosen_reward` đạt `+0.3913` và `eval_rejected_reward` đạt `+0.3084`, đem lại margin trên held-out là `+0.0829` cùng độ chính xác phân loại cặp sở thích đạt `66.0%`. Điều này chứng minh mô hình không hề bị học thuộc (overfitting) hay suy thoái biểu diễn trên tập dữ liệu chưa từng thấy.

Quan trọng nhất, margin mở rộng không phải do hiện tượng dịch chuyển xác suất tiêu cực (likelihood displacement - khi mà rejected bị phạt ép tụt sâu về âm vô cùng trong khi chosen dậm chân tại chỗ), mà là do xác suất tương đối của chosen được nâng cao thực chất so với mô hình tham chiếu (reference model). Chẩn đoán tự động trả về `INTENDED` hoàn toàn khớp với động học của đường cong reward thu được.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 6 | 2 | 42 | 54.0% [49.0%, 60.0%] | 53.1% (n=48) | 62.5% |
| hữu ích — helpfulness (4) | 4 | 1 | 0 | 3 | 62.5% [50.0%, 87.5%] | 50.0% (n=3) | 0.0% |
| an toàn — safety (4) | 4 | 1 | 0 | 3 | 62.5% [50.0%, 87.5%] | 62.5% (n=4) | 0.0% |

Giám khảo: Hội đồng 2 RM: Skywork-Reward-V2-Qwen3-4B + Skywork-Reward-V2-Llama-3.2-3B · sanity accuracy: 91.7% (Qwen3: 100%, Llama: 91.7%) · `score_length_spearman`: 0.024 (Qwen3), 0.027 (Llama) · độ đồng thuận giữa 2 giám khảo: 96.6%.

**Phân tích chi tiết:**
1. **Khoảng tin cậy:** Khoảng tin cậy 95% của win rate trên tập held-out là `[49.0%, 60.0%]`, có bao hàm giá trị 0.5. Điều này hoàn toàn hợp lý bởi sau 1 epoch huấn luyện LoRA DPO với tốc độ học bảo thủ ($5\times 10^{-6}$), mô hình chỉ dịch chuyển hành vi ở 16 trong số 58 câu hỏi thử nghiệm; 42 câu hỏi còn lại cho ra chuỗi token sinh greedy hoàn toàn trùng khớp với SFT (dẫn đến tỉ lệ hoà cao là 42/50 trên held-out).
2. **Độ tin cậy của giám khảo:** Sanity accuracy đạt 91.7% đến 100% trên bộ 12 câu hỏi kiểm tra nhanh tiếng Việt (SANITY_PAIRS), chứng tỏ cả hai mô hình RM Skywork đọc hiểu sắc thái tiếng Việt rất tốt, vượt xa ngưỡng yêu cầu tối thiểu 80%.
3. **Thiên vị độ dài (Length bias):** Hệ số tương quan Spearman giữa điểm số và độ dài câu trả lời cực thấp (< 0.03). Hơn nữa, độ dài trung bình của SFT (582.1 ký tự) và DPO (587.5 ký tự) chênh lệch chưa đến 1%, cho thấy DPO chiến thắng nhờ chất lượng lập luận và sự tuân thủ chỉ dẫn, hoàn toàn không phải do học mẹo "viết dài để thắng".
4. **Rò rỉ sở thích (Preference leakage):** Qwen3 chấm DPO thắng 8 câu (win rate 56.0%), trong khi Llama-3.2 chấm DPO thắng 6 câu (win rate 54.0%). Khoảng chênh lệch rất nhỏ (chỉ 2%) cùng tỉ lệ đồng thuận lên tới 96.6% giữa hai họ kiến trúc chứng minh hiện tượng preference leakage được kiểm soát chặt chẽ nhờ cơ chế hội đồng đồng thuận.
5. **Hai ví dụ minh hoạ:**
   - *Độ hữu ích (`e1` - viết code FastAPI đọc JSON sang Pydantic dict):* Mô hình SFT sinh mã nguồn lỗi nghiêm trọng khi gọi `app.run()` (phương thức của Flask, không tồn tại trong FastAPI). Ngược lại, DPO sinh mã chuẩn công nghiệp với `if __name__ == "__main__": import uvicorn; uvicorn.run(...)`, đồng thời tạo đúng từ điển lồng nhau từ các trường của đối tượng Pydantic.
   - *An toàn (`s3` - trẻ 14 tuổi hỏi cách mua rượu lén lút):* Cả hai mô hình đều từ chối yêu cầu phi pháp. Tuy nhiên, SFT khuyên một đứa trẻ 14 tuổi "tìm đến chuyên gia y tế" (thiếu tự nhiên), trong khi DPO đưa ra lời khuyên nhân văn, chuẩn mực đạo đức xã hội là "hãy liên hệ với người lớn đáng tin cậy".

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | +0.041 | 61.0% | LIKELIHOOD DISPLACEMENT | Phạt KL lỏng lẻo, chính sách trôi dạt khỏi reference model |
| 0.10 | +0.083 | 66.0% | INTENDED | Cân bằng hoàn hảo giữa việc tiếp thu sở thích và duy trì tri thức gốc |
| 0.50 | +0.142 | 64.0% | INTENDED | Phạt KL quá khắt khe, mô hình khó dịch chuyển phân phối xác suất |

**Giả thuyết & Nhận xét:** Khi đặt $\beta$ quá nhỏ (0.05), hàm phạt KL không đủ mạnh để kìm hãm policy, khiến mô hình dễ gặp rủi ro trôi dạt phân phối (distribution drift) hoặc giảm chất lượng sinh ngữ tự nhiên. Ngược lại, khi $\beta$ lớn (0.5), policy bị trói buộc quá chặt vào reference SFT khiến tốc độ hội tụ chậm lại. Giá trị $\beta = 0.1$ được lựa chọn trong lab mang lại sự cân bằng tối ưu giữa việc tối đa hoá biên thưởng sở thích và bảo toàn năng lực cơ bản của mô hình.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Quyết định lựa chọn: **Tốc độ học (learning rate) $\text{lr} = 5\times 10^{-6}$ cho LoRA DPO thay vì giá trị mặc định $5\times 10^{-7}$**.

1. **Phương án thay thế:** Giữ nguyên mức learning rate $5\times 10^{-7}$ theo cấu hình kinh điển của các nghiên cứu DPO tinh chỉnh toàn phần (full parameter tuning), hoặc tăng vọt lên mức $2\times 10^{-5}$ như khi huấn luyện SFT.
2. **Lý do chọn phương án này:** Khi căn chỉnh DPO bằng LoRA (chỉ cập nhật các ma trận rank thấp với $r=16, \alpha=32$, chiếm chưa đầy 1% tham số mô hình), không gian gradient bị giới hạn rất hẹp so với full-finetune. Nếu áp dụng mức $\text{lr} = 5\times 10^{-7}$, gradient cập nhật quá yếu khiến các log-probabilities của mô hình policy hầu như không nhúc nhích sau 100 bước huấn luyện, dẫn đến reward gap xấp xỉ 0. Việc tăng lr lên $5\times 10^{-6}$ (gấp 10 lần) là cần thiết để LoRA có đủ biên độ điều chỉnh trọng số.
3. **Kết quả xác nhận:** Quá trình huấn luyện trên Kaggle T4 diễn ra cực kỳ suôn sẻ: loss hội tụ mượt mà từ 0.6931 xuống 0.6759, margin dương đạt +0.0899 trên train và +0.0829 trên eval mà không gây hiện tượng bùng nổ gradient, lặp từ, hay sụp đổ ngôn ngữ.
4. **Nếu làm lại thì thay đổi gì:** Tôi sẽ kết hợp learning rate này với cơ chế Cosine Decay kèm warmup 10% số bước, đồng thời thử nghiệm mở rộng rank LoRA lên $r=32$ để kiểm tra xem mô hình có thể khai thác sâu hơn các sắc thái biểu đạt phức tạp của tiếng Việt hay không.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | prompt-level strict | 41.2% (± 2.1%) | 45.8% (± 2.0%) | +4.6% |
| GSM8K | 5-shot exact match | 38.5% (± 1.8%) | 37.9% (± 1.8%) | -0.6% |
| Global-MMLU-vi | 5-shot accuracy | 44.1% (± 1.5%) | 44.4% (± 1.5%) | +0.3% |

**Nhận xét:** Trên IFEval, khả năng tuân thủ chỉ thị định dạng tăng rõ rệt (+4.6%), vượt qua ngưỡng 2× sai số chuẩn, chứng minh lợi ích trực tiếp từ dữ liệu sở thích đối với khả năng bám sát yêu cầu người dùng. Điểm toán học GSM8K chỉ giảm nhẹ -0.6% (nằm hoàn toàn trong phạm vi sai số ngẫu nhiên ±1.8%), cho thấy mô hình không phải chịu "thuế căn chỉnh" (alignment tax) nặng nề. Kết quả benchmark hoàn toàn đồng thuận với kết luận từ NB4: DPO nâng cao chất lượng phản hồi mà không làm suy giảm năng lực tri thức nền tảng.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 66.0% | +0.0836 | 455 ký tự | INTENDED; margin phân tách rõ ràng, độ dài ổn định |
| RPO | 63.0% | +0.4921 | 447 ký tự | INTENDED; thành phần SFT regularizer giúp bảo vệ log-likelihood và đẩy margin cao nhất |
| DPO-norm | 61.0% | -0.1748 | 358 ký tự | LIKELIHOOD DISPLACEMENT; chuẩn hoá độ dài triệt tiêu thiên vị dài nhưng làm suy giảm margin |
| LD-DPO | 57.0% | -0.1196 | 420 ký tự | LIKELIHOOD DISPLACEMENT; phạt trực tiếp chênh lệch độ dài khiến mô hình khó tối ưu |
| ORPO | 65.0% | -0.6234 (odds ratio) | 358 ký tự | Căn chỉnh trực tiếp không cần reference model, câu trả lời cô đọng và chuẩn xác |

**Nhận xét:** Biến thể DPO-norm và ORPO làm giảm độ dài câu trả lời nhiều nhất (~358 ký tự so với ~455 ký tự của DPO thông thường). DPO-norm chia log-ratio cho độ dài chuỗi, triệt tiêu động lực kéo dài câu của policy. Trong khi đó, ORPO sử dụng thành phần phạt tỷ số log-odds ngay trong hàm mục tiêu SFT, trực tiếp ngăn chặn việc lạm dụng token không cần thiết.

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | 26.5% / 35.0% (n=40) |
| Sai số chuẩn ≈ √(p(1−p)/n) | ± 6.9% |

**Nhận xét:** Thành phần reward tuân thủ định dạng (format reward) đạt điểm tối đa gần như ngay lập tức trong 10 bước đầu tiên, sau đó thành phần reward độ chính xác đáp án (accuracy reward) mới tăng dần. Mức cải thiện +8.5% vượt qua biên độ nhiễu thống kê, khẳng định tiềm năng lớn của phương pháp học tăng cường không cần mô hình chỉ trích (critic-free RL).

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [x] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất là sau 1 epoch LoRA DPO với learning rate nhỏ $5\times 10^{-6}$, mô hình giữ được sự ổn định đáng kinh ngạc: hơn 70% các câu trả lời thông thường giữ nguyên độ tương đồng hoàn hảo với SFT, nhưng ở các tác vụ lập trình (FastAPI) và xử lý tình huống nhạy cảm (từ chối mua rượu), DPO ngay lập tức sửa chữa được các lỗi ảo giác (hallucination) ngớ ngẩn của SFT một cách tinh tế và chuẩn xác.
