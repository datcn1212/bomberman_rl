# Báo cáo thử nghiệm: Mở rộng Agent Linear Q-Learning sang Task 2 (crate + bomb)

> Bản thử nghiệm nội bộ (không phải report nộp bài cuối kỳ). Ghi lại đầy đủ quá trình — kể cả các hướng **thất bại** — theo đúng tinh thần đề bài mục 9 (Methods): *"You may also describe approaches you tried and abandoned later, including the reasons."* Tiếp nối `bao_cao_q_learning.md` (Task 1). Branch: `dev/datcn/2`.

## 1. Mục tiêu

Mở rộng agent `q_linear` (đã giải tốt Task 1, coin-heaven, ~48/50 coin ổn định qua 3 seed) sang Task 2: bàn có crate, chưa có đối thủ, agent phải học đặt bomb phá crate để lộ coin, đồng thời không tự giết mình. Theo kế hoạch đã duyệt: dùng `loot-crate` (crate 0.75, 50 coin) làm bước đệm trước khi xác nhận trên `classic` (scenario chính thức của Task 2).

**Kết quả tổng quát: chưa thành công.** Toàn bộ các cấu hình thử trên `loot-crate` đều dừng ở **0.00 coin trung bình, luôn chạm trần 400 bước**, kể cả sau 7 lần chẩn đoán và sửa liên tiếp, kể cả khi train tới 40.000 episode. Phần dưới ghi lại đầy đủ quá trình, vì bản thân quá trình chẩn đoán có giá trị khoa học thật.

## 2. Thiết kế ban đầu (trước khi debug)

**Feature 15 chiều**: 4 chiều one-hot hướng BFS tới target chung (coin nếu có, ngược lại ô kề crate gần nhất), 4 valid-move, 4 "nguy hiểm nếu di chuyển hướng này", 1 "đang nguy hiểm", 1 "đặt bomb ở đây có đáng không", 1 bias.

**Reward ban đầu**: `COIN_COLLECTED=+1`, `INVALID_ACTION=-1`, `CRATE_DESTROYED=+0.3`, `KILLED_SELF/GOT_KILLED=-5`, `SURVIVED_ROUND=+0.5`, cộng potential-based shaping theo target chung.

**Replay buffer**: deque 2000, sample batch 32/bước (thay vì update online từng bước như Task 1) — theo thiết kế đã duyệt để tăng ổn định.

**Kết quả 3-seed đầu tiên** (α=0.01, γ=0.95, 15.000 episode/seed): **0.00 / 0.00 / 0.00 coin**, cả 3 đều chạm đúng 400 bước, 0% tự sát.

## 3. Bảy lần chẩn đoán và sửa — toàn bộ đều không giải quyết được vấn đề gốc

| # | Giả thuyết | Thay đổi | Kết quả (loot-crate, coins TB / steps TB) |
|---|---|---|---|
| 1 | Feature hướng dùng chung cho coin và crate khiến model không phân biệt được 2 ngữ cảnh | Tách 4 chiều hướng-coin và 4 chiều hướng-crate riêng (15→19 chiều) | 0.00 / 400.0 |
| 2 | Potential shaping nhảy vọt khi chuyển target crate→coin lúc lộ coin | Bỏ hẳn shaping theo crate, chỉ shaping theo coin | 0.00 / 400.0 |
| 3 | Bỏ shaping crate làm mất tín hiệu dày khi chưa có coin, "đứng yên" hoá hấp dẫn | Potential **cộng dồn**: 0.3×(−dist crate) + 1.0×(−dist coin nếu có) | 0.00 / 400.0 |
| 4 | `CRATE_DESTROYED` quá nhỏ so với rủi ro chết (-5); `SURVIVED_ROUND` thưởng vô điều kiện cho thụ động | Bỏ `SURVIVED_ROUND`, tăng `CRATE_DESTROYED` 0.3→2.0 | 0.00 / 400.0 |
| 5 | Q(WAIT) tự sinh reward dương từ chính công thức shaping khi đứng yên (Φ(s)·(γ−1) > 0 vì Φ≤0) | Phạt `WAITED=-0.1` | 0.00 / 400.0 *(xem mục 4 — dựa trên bằng chứng debug sai)* |
| 6 | Agent dao động tất định vô hạn giữa đúng 2 ô (không phải "đứng yên") vì policy không có trí nhớ | Thêm feature "vừa ở ô này trong 8 bước gần đây" (19→20 chiều), agent tự lưu lịch sử vị trí trong `self` | 0.00 / 400.0 (4.000 episode) |
| 7 | Feature #6 đúng hướng nhưng 4.000 episode chưa đủ để trọng số học cách dùng nó | Giữ nguyên thiết kế #6, train 40.000 episode | 0.00 / 400.0 |

## 4. Một khúc quanh quan trọng: tự phát hiện và sửa sai lầm debug của chính mình

Sau fix #4, dùng log debug (`self.logger.info` mỗi bước) để xem trực tiếp hành vi thay vì chỉ nhìn số liệu tổng hợp — phát hiện `Q(WAIT)=0.0297 > Q(BOMB)=-0.025` ngay tại ô mà feature báo "đặt bomb ở đây rất đáng". Từ đó suy luận (đúng về mặt toán học): với công thức potential-shaping, đứng yên tại state có Φ(s)<0 luôn sinh ra `reward = Φ(s)·(γ−1) > 0` — một khoản thưởng miễn phí — dẫn tới fix #5 (phạt WAITED).

Fix #5 test xong vẫn 0.00. Khi đào sâu thêm để hiểu tại sao, phát hiện **các lệnh debug trace đó dùng `Q_LINEAR_MODEL_PATH` dạng đường dẫn tương đối gõ tay trên dòng lệnh** (`agent_code/q_linear/weights_X.npy`) — trong khi framework tự `chdir` vào đúng thư mục `agent_code/q_linear/` trước khi gọi `callbacks.py`, khiến chuỗi tương đối đó bị hiểu sai thành đường dẫn lồng không tồn tại. Kết quả: `os.path.isfile(...)` trả về `False`, agent **âm thầm dùng trọng số khởi tạo ngẫu nhiên** thay vì model đã train — toàn bộ bằng chứng "Q(WAIT) thắng" ở trên là quan sát trên **model chưa học gì**, không phải model thật.

Điểm quan trọng: đây là lỗi trong **lệnh debug tôi tự gõ tay**, không phải lỗi trong pipeline train/eval chính (`run_task2_config.py`), vì script đó dùng `Path(__file__).parent` — tự động phân giải thành đường dẫn tuyệt đối khi script thực sự chạy, nên không bị ảnh hưởng. Đã xác minh lại bằng cách in `cwd`/`isfile` trực tiếp trong `setup()` và so sánh 2 cách gọi. Toàn bộ kết quả huấn luyện/đánh giá (bảng mục 3) **không bị ảnh hưởng bởi lỗi này** — chỉ có phần suy luận dẫn tới fix #5 là dựa trên bằng chứng sai.

Debug lại đúng cách (đường dẫn tuyệt đối) cho thấy sự thật: agent dao động **tất định giữa đúng 2 ô** (ví dụ (15,15)↔(15,14)) suốt 400 bước — tại ô có `bomb-here-good=1.0`, `Q(BOMB)=-2.28` vẫn thua xa `Q(DOWN)=0.535` dẫn nó quay lại ô kia. Đây chính là hiện tượng "kẹt loop" đã thấy ở Task 1, nhưng nghiêm trọng hơn nhiều vì chặn đứng toàn bộ chuỗi hành vi cần thiết, dẫn tới fix #6 và #7.

## 5. Đánh giá cuối: vấn đề có thể không nằm ở feature engineering

Feature #6 (`recently_visited`) được tính đúng, xác nhận qua log (`True` từ bước 3), nhưng **không thay đổi được quyết định**: so sánh cùng vị trí trước/sau khi biết "đã từng ở đây", `Q(DOWN)` thực ra **tăng** (0.298 → 0.499) thay vì giảm. Train thêm 10 lần (40.000 episode) không sửa được điều này.

Giả thuyết hợp lý nhất hiện tại: nút thắt không phải "thiếu thông tin trong feature" mà là **giới hạn năng lực biểu diễn của mô hình tuyến tính kết hợp động lực học semi-gradient TD**. Hành vi cần học ở Task 2 (tiếp cận → đặt bomb → né 4 bước → chờ nổ → quay lại → nhặt coin) là một chuỗi nhiều pha, phụ thuộc ngữ cảnh dài — khác hẳn Task 1 (chỉ cần "đi thẳng tới coin đang thấy", gần như đơn pha). Một hàm Q **tuyến tính** trên feature tức thời khó biểu diễn tốt giá trị của một chuỗi hành vi dài như vậy, và transition "dao động 2 ô" (rất phổ biến trong buffer một khi đã kẹt) có thể lấn át tín hiệu học từ các transition hiếm nhưng quan trọng (chuỗi đặt-bomb-thành-công). Đây cũng là biểu hiện cùng họ với vấn đề "deadly triad" (function approximation + bootstrapping) đã quan sát được ở Task 1 (γ=0.8 bất ổn qua 3 seed) — chỉ là nghiêm trọng hơn nhiều trong không gian hành vi phức tạp hơn của Task 2.

## 6. Quyết định dừng lại

Theo thảo luận với người dùng: dừng đào sâu hướng crate-bombing bằng linear Q-learning ở đây, ghi lại toàn bộ làm tư liệu khoa học, không tiếp tục đoán-sửa thêm. Model mặc định nộp bài (`q_linear_weights.npy`) được khôi phục về bản **Task 1 (coin-heaven) đã train lại bằng đúng code hiện tại (20 chiều)** để đảm bảo tương thích — xem mục 7.

## 7. Khôi phục model mặc định tương thích

Trong quá trình debug, `q_linear_weights.npy` (đường dẫn mặc định `callbacks.py` load) bị các lệnh smoke-test ghi đè bằng trọng số ngẫu nhiên/chưa train đủ. Đồng thời, model Task 1 tốt nhất đã lưu trước đó (`weights_alpha_low.npy`, 9 chiều) không còn tương thích với `features.py` hiện tại (đã tiến hoá lên 20 chiều qua các fix Task 2). Đã train lại nhanh trên `coin-heaven` bằng chính code 20 chiều hiện tại (các feature liên quan crate/bomb tự nhiên bằng 0 trong môi trường không crate/không bomb), α=0.01, γ=0.95, 20.000 episode:

| coins TB | steps TB | suicide_rate |
|---|---|---|
| 19.81 / 50 | 343.4 | 7.2% |

Kết quả này **kém hơn hẳn** bản Task 1 gốc (9 chiều, 48.00±0.57 qua 3 seed, 0% chết — xem `bao_cao_q_learning.md`). Hai khác biệt đáng chú ý:
- **Điểm số giảm mạnh**: đây chỉ là 1 lần train nhanh (không kiểm chứng 3-seed) để khôi phục tính tương thích, không phải một nỗ lực tối ưu lại — không nên so sánh trực tiếp với con số đã kiểm chứng kỹ của Task 1.
- **Xuất hiện tự sát (7.2%)** dù `coin-heaven` không có crate: các feature/reward liên quan tới bomb được thêm cho Task 2 (`bomb-here-good`, phạt `KILLED_SELF/GOT_KILLED=-5`, phạt `WAITED=-0.1`) vẫn hoạt động trong scenario này và rõ ràng đã thay đổi hành vi agent theo hướng xấu đi ngay cả ở bài toán mà Task 1 từng giải rất tốt — bằng chứng nữa cho thấy các thay đổi thiết kế ở Task 2 có tác dụng phụ ngoài ý muốn, chưa được kiểm soát tốt.

**Kết luận thực dụng**: model mặc định hiện tại (`q_linear_weights.npy`) ưu tiên tính *tương thích và có thể chạy được* hơn là *hiệu năng tối ưu* — nếu cần nộp bài thật, nên cân nhắc quay lại đúng code+trọng số 9 chiều của Task 1 (giữ 2 phiên bản `callbacks.py` riêng cho 2 task) thay vì dùng chung 1 bộ code đang tiến hoá dở dang.

## 8. Hạn chế và hướng tiếp theo

- **Chưa giải được Task 2 bằng kiến trúc hiện tại** (linear Q, online semi-gradient, replay buffer nhỏ). Đây là hạn chế chính cần nêu rõ trong report chính thức.
- Hướng nên thử tiếp (không thực hiện trong phiên này): batch fitted-Q (thu thập dữ liệu theo đợt rồi fit lại toàn bộ, thay vì cập nhật liên tục — giảm vấn đề "moving target" của bootstrapping); hoặc chuyển hẳn sang Model B (Random Forest, đã định hướng trong `plan.md`) cho task này, vì cây quyết định có thể biểu diễn cấu trúc giá trị nhiều pha tốt hơn linear model.
- Nếu tiếp tục hướng linear: cân nhắc cơ chế phá loop tường minh hơn (không chỉ dựa vào feature mềm) làm lưới an toàn kỹ thuật, chấp nhận đây là lựa chọn thiết kế có chủ đích thay vì kỳ vọng thuần "học ra".
- Nên kéo dài đáng kể giai đoạn epsilon cao (khám phá ngẫu nhiên nhiều hơn) trước khi giảm dần, để buffer tích luỹ đủ trải nghiệm đặt-bomb-thành-công thật trước khi chuyển sang khai thác tham lam.
- Bài học quy trình: khi debug bằng cách gõ lệnh tay có truyền đường dẫn liên quan tới thư mục agent, luôn dùng đường dẫn tuyệt đối — framework tự đổi thư mục làm việc trước khi gọi callback, dễ gây hiểu sai nếu không để ý.

## 9. Chi tiết tái lập

Code: `agent_code/q_linear/{features,callbacks,train}.py` (trạng thái hiện tại = sau fix #7, 20 chiều feature). Công cụ thí nghiệm: `run_task2_config.py`. Log/trọng số từng lần thử: `agent_code/q_linear/{log,weights}_<tên>.csv|npy` (ví dụ `lootcrate_a001_seed1..3`, `lootcrate_fixcheck2..6`, `lootcrate_40k`). Kết quả eval thô: `results/eval_<tên>.json`.
