# Báo cáo thử nghiệm: Agent Random Forest Fitted-Q - Task 1 đến Task 4

> Bản thử nghiệm nội bộ (không phải report nộp bài cuối kỳ). Model thứ ba, tiếp nối `bao_cao_q_learning.md` (Model A: Linear Q, Task 1), `bao_cao_task2.md` (Linear Q ở Task 2 - thất bại) và `bao_cao_tabularQ.md` (Model B: Tabular Q, Task 1-4). Branch: `dev/datcn/RF_fittedQ`.

## 1. Mục tiêu và động cơ

Xây agent thứ ba dùng **Random Forest làm hàm xấp xỉ Q**, huấn luyện bằng **Fitted-Q Iteration** (phương pháp batch), để trả lời hai câu hỏi mà hai model trước chưa trả lời được:

1. **Về mặt thuật toán**: hai model trước đều cập nhật online từng bước (semi-gradient cho linear, Bellman update cho tabular). Fitted-Q Iteration thu thập dữ liệu theo đợt rồi fit lại toàn bộ hàm Q - về lý thuyết tránh được vấn đề "moving target" của bootstrapping online, vốn là nguyên nhân bất ổn đã quan sát trực tiếp ở Model A (`bao_cao_q_learning.md` mục 5).
2. **Về mặt biểu diễn**: Random Forest học được ranh giới quyết định phi tuyến mà linear không biểu diễn nổi, đồng thời nhận đầu vào **liên tục** (khoảng cách thật) thay vì phải rời rạc hoá như tabular - không mất thông tin định lượng.

Random Forest cũng là kỹ thuật đã dạy trên lớp (Decision Trees & Random Forest), nên hợp lệ cho yêu cầu "ít nhất 1 model dùng kỹ thuật trên lớp".

## 2. Thiết kế

### 2.1 Feature: liên tục, và né vùng nguy hiểm NGAY TỪ ĐẦU

22 chiều liên tục (`agent_code/q_rf/features.py`):

| Chiều | Ý nghĩa |
|---|---|
| 0-3 | one-hot hướng BFS tới coin gần nhất |
| 4 | khoảng cách tới coin đó (50.0 nếu không tới được) |
| 5-8 | one-hot hướng BFS tới ô kề crate gần nhất |
| 9 | khoảng cách tới ô đó |
| 10-13 | mỗi hướng: "đi hướng này có bước vào vùng nổ không" |
| 14-17 | mỗi hướng: "có đi được không" |
| 18 | đang đứng trong vùng nguy hiểm |
| 19 | đặt bomb ở đây có phá được crate và hiện an toàn |
| 20 | đặt bomb ở đây có trúng đối thủ |
| 21 | stuck_ratio (độ lặp vị trí gần đây) |

**Khác biệt then chốt so với tabular**: giữ nguyên khoảng cách dạng số (chiều 4, 9) thay vì bucket hoá. Cây quyết định tự tìm ngưỡng chia (vd. "coin cách dưới 3 ô thì hành xử khác cách 9 ô") - thông tin mà bản rời rạc của tabular vứt đi, còn linear chỉ biểu diễn được bằng một hệ số dốc cố định.

**Áp dụng ngay bài học đắt giá nhất từ Model B**: mọi lệnh BFS tìm mục tiêu đều **né vùng nguy hiểm** (`avoid=danger`) ngay từ phiên bản đầu tiên. Ở tabular, thiếu điều này gây ra lỗi chết người chỉ phát hiện được sau bốn hướng thử tham số thất bại và một lần debug trace theo trình tự chết (xem `bao_cao_tabularQ.md` mục 3.7): sau khi thoát bomb thành công, hướng-tới-mục-tiêu kéo agent ngược lại đúng vùng khói còn hiệu lực. Riêng đường thoát hiểm (`escape_dir` bên tabular) thì **không** né - vùng nổ là hành lang thẳng nên thoát ra thường phải băng qua thêm 1-2 ô còn nguy hiểm; ở đây thông tin tương đương được cung cấp qua chiều 10-13 để model tự học.

### 2.2 Thuật toán: Fitted-Q Iteration

Thay vì cập nhật từng bước, gom transition vào buffer (tối đa 60.000) và cứ `REFIT_EVERY` episode thì fit lại toàn bộ:

```
targets = r + gamma * max_a' Q_k(s', a')     (r nếu s' là terminal)
Q_{k+1} = fit(states, targets)               (một RandomForestRegressor cho mỗi action)
```

Sáu regressor riêng biệt (một cho mỗi action) thay vì một model nhận action làm feature - giống cách `q_linear` dùng một vector trọng số cho mỗi action, giúp mỗi cây chuyên biệt hoá cho một hành động.

Tham số: `n_estimators=20`, `max_depth=12`, `min_samples_leaf=5`, `n_jobs=1` (luật cấm multiprocessing lúc thi đấu; giữ nguyên cả lúc train cho trung thực).

### 2.3 Reward: giữ nguyên hoàn toàn từ hai model trước

`REWARDS` dict, potential-based shaping, phạt stuck/danger trực tiếp, thưởng đặt bomb gần đối thủ - **copy nguyên** từ `q_tabular/train.py`. Đây là quyết định ở tầng MDP, độc lập với cách biểu diễn Q, và đã tốn rất nhiều công để đi tới (xem `bao_cao_task2.md`). Thay đổi ở đây sẽ khiến ba model không còn so sánh được với nhau.

## 3. Kết quả và các lần chẩn đoán

### 3.1 Ràng buộc thi đấu: latency (kiểm tra TRƯỚC khi tối ưu hiệu năng)

Random Forest nặng hơn hẳn một phép nhân vector (linear) hay một phép tra bảng (tabular), nên rủi ro vi phạm giới hạn 0.5s/bước là có thật và phải kiểm tra sớm. Benchmark trên bàn `classic` mật độ crate đầy đủ, có bom đang đếm ngược (worst case cho BFS), gồm **toàn bộ** chi phí một bước quyết định (trích xuất feature + 6 lần predict):

| mean | p99 | max | % ngân sách 0.5s |
|---|---|---|---|
| 9.18 ms | 9.46 ms | 9.51 ms | **1.9%** |

**Kết luận: an toàn tuyệt đối** - còn dư hơn 50 lần ngân sách. Ràng buộc thi đấu không phải lý do để loại hướng Random Forest.

### 3.2 Phiên bản đầu (1 vòng FQI mỗi lần refit): Task 1 tốt, Task 2 sụp

| Scenario | coins TB | suicide_rate | Ghi chú |
|---|---|---|---|
| `coin-heaven` (Task 1) | **47.54 / 50** | 12.0% | 1500 episode |
| `loot-crate` (Task 2) | **0.05 / 50** | 64.0% | 1500 episode |

Task 1 gần bằng linear (48.00) và tabular (50.00) - xác nhận cơ chế hoạt động đúng, không có bug nền tảng. Nhưng Task 2 gần như bằng không.

**Chẩn đoán (dựa trên bản chất thuật toán, không đoán mò)**: mỗi lần refit, giá trị chỉ lan truyền **đúng một bước bootstrap** về phía sau, vì target của lần fit này được xây từ forest của lần fit trước. Với `REFIT_EVERY=100` trên 1500 episode, tổng cộng chỉ có **15 lần lan truyền**. Task 1 chỉ cần chuỗi quyết định rất ngắn ("đi thẳng tới coin đang nhìn thấy") nên 15 bước là thừa đủ; Task 2 cần chuỗi dài (tới gần crate -> đặt bomb -> né 4 bước -> chờ nổ -> quay lại nhặt coin) nên tín hiệu phần thưởng không bao giờ lan ngược tới được quyết định đặt bomb ban đầu.

Bằng chứng từ learning curve khớp với chẩn đoán: coin/episode bò từ 0.00 lên chỉ 0.58 rồi tụt lại, crate phá được tăng chậm 2.66 -> 4.56 rồi giảm.

### 3.3 Sửa: nhiều vòng FQI trên cùng buffer

Chạy `FQI_PASSES=6` vòng lặp fitted-Q trên cùng một buffer mỗi lần refit - đây chính là dạng chuẩn của Fitted-Q Iteration trong tài liệu, và **không tốn thêm một bước tương tác môi trường nào**. Đồng thời giảm `n_estimators` 40 -> 20 để bù chi phí tính toán.

| Phiên bản | coins TB (`loot-crate`) | thời gian train |
|---|---|---|
| v1 (1 vòng/refit) | 0.05 | 2420s |
| v2 (6 vòng/refit) | **0.31** | **464s** |

Tốt hơn 6 lần và nhanh hơn 5 lần cùng lúc. Learning curve tại cùng mốc episode 1200: coins 1.07 so với 0.58 của v1, crates 5.75 so với 4.56 - cải thiện nhất quán chứ không phải nhiễu một điểm.

### 3.4 Tăng số episode: cải thiện tiếp, nhưng lộ ra vấn đề nghiêm trọng hơn

Learning curve của v2 vẫn đang đi lên tại điểm cuối (chưa bão hoà), nên thiếu dữ liệu là biến rõ ràng tiếp theo cần thử. Chạy 3 seed với 4000 episode:

| Seed | coins TB (`loot-crate`) |
|---|---|
| 1 | 0.95 |
| 2 | 0.03 |
| 3 | 0.00 |
| **Trung bình** | **0.33 +/- 0.44** |

Seed tốt nhất cải thiện tiếp gần 3 lần so với 1500 episode (0.31 -> 0.95), nhưng hai seed còn lại gần như bằng không - **chênh nhau tới 30 lần giữa các seed cùng cấu hình**.

### 3.5 Phát hiện: thoái hoá đúng thời điểm epsilon giảm hết

Nhìn learning curve cuối kỳ của cả ba seed cho thấy một pattern **giống hệt nhau**, không phải nhiễu:

| Episode | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| 3000 | 1.09 | 1.14 | 1.00 |
| 3500 | 0.67 | 0.60 | 0.63 |
| 4000 | 0.51 | 0.46 | 0.63 |

Cả ba đều đạt đỉnh quanh episode 3000 rồi **thoái hoá đều đặn**. Với `EPS_DECAY_FRAC=0.8` và 4000 episode, epsilon chạm đáy (0.05) tại đúng **episode 3200** - khớp chính xác thời điểm bắt đầu tụt.

**Diễn giải**: khi exploration gần như tắt, buffer chỉ còn chứa các quỹ đạo do chính policy hiện tại sinh ra - một phân phối rất hẹp. Random Forest fit lại trên phân phối hẹp đó sẽ khớp rất tốt vùng dữ liệu quen thuộc nhưng ngoại suy tệ ở những state mà policy mới cần ghé qua, và chính sách tiếp theo lại càng hẹp hơn - một vòng xoáy tự củng cố. Đây là biểu hiện của phân phối dữ liệu lệch (distribution shift) trong phương pháp batch off-policy, không phải bug cài đặt.

**Kiểm chứng trực tiếp**: giữ `EPS_END=0.25` thay vì 0.05, mọi thứ khác giữ nguyên.

| Episode | EPS_END=0.05 (TB 3 seed) | EPS_END=0.25 |
|---|---|---|
| 3000 | 1.08 | 0.65 |
| 3500 | 0.63 | **0.98** |
| 4000 | 0.53 | **0.89** |

Đường cong **không còn thoái hoá** - vẫn giữ mức cao ở cuối thay vì tụt. Chạy đủ 3 seed để xác nhận:

| Seed | EPS_END=0.05 (v3) | EPS_END=0.25 (v4) |
|---|---|---|
| 1 | 0.95 | 1.57 |
| 2 | 0.03 | 1.26 |
| 3 | 0.00 | 1.27 |
| **Trung bình** | **0.33 +/- 0.44** | **1.37 +/- 0.14** |

Giả thuyết được xác nhận trên **cả hai trục**: điểm trung bình tăng hơn 4 lần, và độ lệch chuẩn giữa các seed giảm 3 lần (0.44 -> 0.14). Cấu hình cũ có seed ăn may (0.95) lẫn seed hỏng hoàn toàn (0.00); cấu hình mới cho ba kết quả gần như trùng nhau. Đây là bằng chứng độc lập thứ hai cho cùng chẩn đoán: chính việc mất đa dạng dữ liệu (chứ không phải sự ngẫu nhiên khi khởi tạo rừng) là thứ khiến kết quả trồi sụt.

### 3.6 So sánh với hai model trước trên `classic` (scenario chính thức của Task 2)

3 seed, curriculum 2 pha (`loot-crate` -> `classic`), so sánh hai cấu hình epsilon:

| Seed | v3 (EPS_END=0.05) | v4 (EPS_END=0.25) |
|---|---|---|
| 1 | 0.06 | 0.15 |
| 2 | 0.09 | 0.17 |
| 3 | 0.03 | 0.12 |
| **Trung bình** | **0.06** | **0.15 +/- 0.02** |

Cấu hình v4 lại cải thiện 2.5 lần và cực kỳ ổn định giữa các seed (độ lệch chuẩn chỉ 0.02), nhất quán với phát hiện ở mục 3.5. Suicide_rate cũng giảm (66.4% -> 36.0%).

Nhưng so với Model B (tabular, **cùng reward, cùng curriculum, cùng feature ý tưởng**): **5.52/9 coin** - RF vẫn kém hơn khoảng **37 lần**.

### 3.7 Task 3 và Task 4 (có đối thủ)

Curriculum 3 pha (`loot-crate` -> `classic` một mình -> `classic` có đối thủ), 3000 episode mỗi pha, `EPS_END=0.25`, 3 seed:

| Đối thủ | coins TB | kills TB | suicide_rate TB |
|---|---|---|---|
| `peaceful_agent` (Task 3) | 0.10 +/- 0.04 | 0.000 | **23.6% +/- 0.8%** |
| `rule_based_agent` (Task 4) | 0.33 +/- 0.06 | 0.005 | **55.1% +/- 4.4%** |

So sánh trực tiếp với Model B (tabular) trên cùng hai bài toán:

| Chỉ số | Task 3 tabular | Task 3 RF | Task 4 tabular | Task 4 RF |
|---|---|---|---|---|
| coins | **3.25** | 0.10 | **0.59** | 0.33 |
| kills | **0.227** | 0.000 | 0.003 | 0.005 |
| suicide_rate | 39.0% | **23.6%** | 88.6% | **55.1%** |

**Một kết quả đảo chiều đáng chú ý**: RF **sống sót tốt hơn hẳn** tabular ở cả hai task (suicide 23.6% so với 39.0%, và 55.1% so với 88.6%) dù nhặt được ít coin hơn nhiều. Kết hợp với `steps_mean` cao (307-316 bước ở Task 3, so với trần 400), bức tranh khá rõ: RF học được một chính sách **thận trọng quá mức** - tránh nguy hiểm rất tốt nhưng hiếm khi dám đặt bomb để mở đường tới coin. Đây không phải cùng một thất bại với tabular (tabular chết nhiều vì mạo hiểm), mà là thất bại ở đầu ngược lại của cùng một sự đánh đổi.

## 4. Nhận xét

**Bốn lần chẩn đoán, mỗi lần đều truy được về một nguyên nhân cụ thể**: hiệu năng trên `loot-crate` đi từ 0.05 lên 1.37 coin (gấp 27 lần) qua bốn thay đổi, và **không thay đổi nào là đoán mò**: (1) số vòng FQI - suy ra từ cơ chế lan truyền giá trị của thuật toán, kiểm chứng bằng việc Task 1 (chuỗi ngắn) vẫn tốt trong khi Task 2 (chuỗi dài) sụp; (2) số episode - suy ra từ learning curve chưa bão hoà; (3) epsilon cuối - suy ra từ việc cả ba seed thoái hoá đúng thời điểm epsilon chạm đáy, xác nhận trên cả điểm trung bình lẫn độ ổn định. Bản thân quy trình này là phần có giá trị khoa học nhất của model, kể cả khi kết quả cuối không cạnh tranh được.

**Nhưng RF fitted-Q thua rõ rệt tabular Q trên mọi task có crate**, dù dùng **chung reward, chung ý tưởng feature, chung curriculum**: `loot-crate` 1.37 so với 24.86; `classic` 0.15 so với 5.52; Task 3 0.10 so với 3.25. Vì mọi thứ khác đã được giữ giống nhau có chủ đích, khác biệt phải nằm ở **cách học giá trị**, và ba yếu tố đều chỉ về cùng một hướng:

- **Số lần cập nhật chênh lệch hàng nghìn lần**. Tabular cập nhật mỗi bước: 8000 episode x ~250 bước = khoảng 2 triệu lần cập nhật. RF chỉ fit lại 40 lần (4000/100) x 6 vòng = 240 lần fit. Mỗi lần fit tuy "nhìn" toàn bộ buffer nhưng vẫn là 240 điểm cải thiện, so với 2 triệu.
- **Mỗi lần fit là học lại từ đầu**. Rừng mới không kế thừa gì từ rừng cũ ngoài các giá trị target - trái ngược với Q-table hay vector trọng số vốn tích luỹ dần qua từng cập nhật.
- **Tổng quát hoá làm mờ đúng những ranh giới quan trọng nhất**. Cây quyết định chia không gian thành các vùng và gán một giá trị chung cho cả vùng. Nhưng trong Bomberman, hai state chỉ khác nhau một ô (đứng trong vùng nổ hay ngoài nó) có giá trị thật chênh lệch cực lớn. Chính khả năng tổng quát hoá - thứ giúp RF mạnh ở bài toán học có giám sát - lại làm nhoè ranh giới sinh-tử này, trong khi Q-table biểu diễn chúng hoàn toàn tách biệt.

**Điểm mạnh thật sự của RF nằm ở chỗ khác**: chính sách nó học ra **an toàn hơn hẳn** (suicide thấp hơn tabular ở cả Task 3 và Task 4), và **ổn định giữa các seed hơn nhiều** sau khi sửa epsilon (độ lệch chuẩn 0.02-0.14, so với 12.49 của tabular trên `loot-crate`). Nếu tiêu chí là "không tự sát và hành xử nhất quán" thì RF thắng; nếu tiêu chí là điểm số thì thua đậm.

**Ràng buộc thi đấu không phải vấn đề**: 9.5ms/bước là 1.9% ngân sách 0.5s. Đây là điều đáng ghi nhận vì trực giác ban đầu ("random forest chắc quá nặng cho real-time") hoá ra sai, và nếu không đo thì rất dễ loại nhầm hướng này ngay từ đầu vì lý do không có thật.

## 5. Hạn chế và hướng tiếp theo

- **Ngân sách compute không tương đương giữa ba model** - đây là hạn chế quan trọng nhất khi đọc các con số so sánh. Tabular chạy 8000 episode trong khoảng 4 phút; RF chạy 4000 episode mất 30-40 phút (chậm hơn khoảng 60 lần cho mỗi episode). Với cùng **thời gian thực**, tabular có thể chạy nhiều hơn RF hàng chục lần. Nói cách khác, so sánh hiện tại công bằng theo *số episode* nhưng RF bị thiệt nặng theo *thời gian*, mà thời gian mới là ràng buộc thật của đồ án.
- Chưa thử `REFIT_EVERY` nhỏ hơn (refit dày hơn -> nhiều điểm cải thiện hơn), vốn là biến nhắm thẳng vào nguyên nhân "240 lần fit so với 2 triệu lần cập nhật" nêu ở mục 4. Chi phí tăng tuyến tính nên cần cân nhắc với ràng buộc thời gian.
- Chưa quét `n_estimators`/`max_depth`/`min_samples_leaf` - toàn bộ kết quả dùng một bộ tham số rừng duy nhất, chọn theo cân bằng tốc độ/chất lượng chứ chưa tối ưu.
- Chưa thử ý tưởng nhắm vào điểm yếu cụ thể đã chẩn đoán ở mục 4: bổ sung feature nhị phân đánh dấu rõ ranh giới sinh-tử (thay vì để cây tự tìm ngưỡng trên các chiều khoảng cách liên tục), hoặc tăng trọng số mẫu cho các transition gần cái chết.
- Task 3/Task 4 dùng 3000 episode mỗi pha thay vì 4000 như Task 2 (giới hạn thời gian) - các con số của hai task này vì vậy hơi bất lợi so với phần còn lại của báo cáo, không nên so sánh trực tiếp tuyệt đối.
- Kills gần như bằng 0 ở cả Task 3 và Task 4, giống hệt Model B - nhất quán với chẩn đoán ở `bao_cao_tabularQ.md` mục 3.5 rằng bắn trúng mục tiêu di chuyển bằng vũ khí trễ 4 bước là bài toán khó tự thân, không phụ thuộc cách biểu diễn Q.

## 6. Chi tiết tái lập

Code: `agent_code/q_rf/{features,callbacks,train}.py` (tự chứa hoàn toàn, không import từ `q_linear` hay `q_tabular`). Cờ `Q_RF_CONTINUE=1` cho phép train tiếp từ model có sẵn (dùng cho curriculum learning). Công cụ thí nghiệm (không nộp bài): `run_rf_config.py` - hỗ trợ 1 pha (1 scenario), 2 pha (curriculum loot-crate -> classic) và 3 pha (thêm pha có đối thủ cho Task 3/4) qua biến `PHASES`. Log/model từng lần chạy: `agent_code/q_rf/{log,model}_<tên>.csv|pkl`. Kết quả eval thô: `results/eval_<tên>.json`.

**Cấu hình tốt nhất tìm được** (dùng cho mọi kết quả v4 ở trên): `FQI_PASSES=6`, `EPS_END=0.25`, `n_estimators=20`, `max_depth=12`, `REFIT_EVERY=100`, 4000 episode mỗi pha (3000 cho Task 3/4). Lệnh tái lập:

```
CFG_NAME=<ten> PHASES=2 EPISODES=4000 EVAL_ROUNDS=150 Q_RF_EPS_END=0.25 python3 run_rf_config.py
```

**Tên các lần chạy trong repo**: `rf_task1_sanity` (Task 1); `rf_v3_*` (EPS_END=0.05, cấu hình bị thoái hoá); `rf_v4_*` (EPS_END=0.25, cấu hình tốt nhất); `rf_task3_run*`/`rf_task4_run*` (Task 3/4). Benchmark latency ở mục 3.1 chạy trực tiếp trên `model_smoketest.pkl` với một `game_state` dựng thủ công ở mật độ crate của `classic` kèm hai quả bom đang đếm ngược.

**Model mặc định** (`q_rf_model.pkl`, đường dẫn `callbacks.py` tự load): bản `rf_v4_classic_run2` - chọn theo điểm cao nhất trên `classic` (scenario chính thức của Task 2, 0.17/9 coin) chứ không phải bản cao điểm nhất trên `loot-crate` (`rf_v4_eps_lootcrate`, 1.57/50). Cả hai đều còn xa mức nộp bài được; đây chỉ là mốc tham chiếu để tái lập, không phải ứng viên thi đấu - xem mục 4 và 5.
