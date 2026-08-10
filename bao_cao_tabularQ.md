# Báo cáo thử nghiệm: Agent Tabular Q-Learning - Task 1 và Task 2

> Bản thử nghiệm nội bộ (không phải report nộp bài cuối kỳ). Tiếp nối `bao_cao_q_learning.md` (Task 1, Linear Q) và `bao_cao_task2.md` (Task 2, Linear Q - dừng lại ở 0.00 coin trên `classic`, không giải được). Agent này ra đời trực tiếp từ giả thuyết đặt ra ở cuối `bao_cao_task2.md` mục 5: nút thắt của Linear Q có thể là **giới hạn biểu diễn của hàm xấp xỉ tuyến tính**, không phải thiếu feature - nếu đúng, một Q-table rời rạc (không xấp xỉ, không "aliasing" giữa các state khác nhau trông giống nhau qua ống kính linear) phải học tốt hơn trên chính bài toán mà Linear Q thất bại.

## 1. Mục tiêu và phạm vi

Xây một agent **thứ hai**, độc lập về mặt biểu diễn (Model B: tabular Q-learning, kỹ thuật dạy trên lớp - lecture 35-36), xử lý **cả Task 1 và Task 2** bằng chung một thiết kế state/reward, để so sánh trực tiếp và công bằng với `q_linear` ở cùng ngân sách episode. Câu hỏi cần trả lời: bỏ hàm xấp xỉ tuyến tính đi, giữ nguyên reward/MDP đã kiểm chứng, kết quả Task 2 có đổi khác về chất không?

## 2. Thiết kế

### 2.1 State - rời rạc hoá riêng, không copy tài liệu tham khảo

Trong lúc tìm hiểu `ex2-bomberman` (repo nộp bài thật của nhóm khác, chỉ tham khảo phương pháp, không copy code), thấy agent `classic_1` của họ dùng tabular Q với 1920 state (8 đặc trưng categorical, enumerate bằng `itertools.product`). Không dùng lại state space đó - tự thiết kế state gọn hơn (6 chiều, mã hoá dạng mixed-radix thành 1 số nguyên, tra bảng O(1) thay vì quét tuyến tính qua danh sách tổ hợp):

| Chiều | Số giá trị | Ý nghĩa |
|---|---|---|
| `target_dir` | 5 | Hướng bước đầu trên đường BFS ngắn nhất tới coin (nếu có), ngược lại tới ô kề crate gần nhất |
| `target_is_coin` | 2 | Target hiện tại là coin hay ô-kề-crate |
| `danger_now` | 2 | Đang đứng trong vùng nổ bom nào đó |
| `escape_dir` | 5 | Hướng BFS tới ô an toàn gần nhất (chỉ có ý nghĩa khi `danger_now=1`) |
| `can_bomb_here` | 2 | `has_bomb & cạnh crate & không đang nguy hiểm` - đúng semantic đã kiểm chứng ở feature linear |
| `stuck_bucket` | 3 | Bucket hoá `stuck_ratio` liên tục (LOW/MED/HIGH) |

Tổng **600 state x 6 action = 3600 ô bảng** - nhỏ hơn nhiều so với 1920 state của `ex2-bomberman`, cố tình để tabular hội tụ được trong vài nghìn episode thay vì cần rất nhiều dữ liệu. Không cần feature "đi được hướng này không" riêng như bản linear: `target_dir`/`escape_dir` luôn trỏ tới ô đi được (đảm bảo bởi chính thuật toán BFS), trong khi bản linear cần 4 cờ valid-move vì linear không tự chạy BFS lúc suy luận được - đây là chỗ tabular đơn giản hoá được so với linear, không phải bỏ sót thông tin.

### 2.2 Reward - tái sử dụng nguyên vẹn từ Linear Q, không thiết kế lại

`REWARDS` dict, potential-based shaping (đã fix lỗi thứ tự tính `best_crate_distance`), phạt `stuck_ratio` trực tiếp, phạt `in_danger` trực tiếp - copy nguyên từ `q_linear/train.py` (bản đã kiểm chứng qua toàn bộ hành trình debug ở `bao_cao_task2.md`). Đây là quyết định ở tầng MDP/reward, độc lập với việc dùng linear hay tabular để biểu diễn Q - không có lý do gì thiết kế lại từ đầu.

**Không dùng n-step TD** (khác với thử nghiệm gần nhất trên linear model): thí nghiệm n-step (N=3) trên `q_linear` cho thấy nó khuếch đại phạt `danger_penalty` lan ngược về 2-3 hành động trước đó, làm chính sách sụp về "không bao giờ đặt bomb" - bài học áp dụng thẳng ở đây: giữ update tabular Q-learning 1-step chuẩn, tránh rủi ro tương tự.

### 2.3 Hyperparameter

`alpha=0.2` (cao hơn hẳn `alpha=0.01` của linear - hợp lý vì tabular không có nhiễu chéo giữa các state như linear (cập nhật 1 ô không ảnh hưởng ô khác), nên chịu được bước cập nhật mạnh hơn mà không mất ổn định); `gamma=0.95` (giữ nguyên - thuộc về task, không phụ thuộc cách biểu diễn); epsilon-greedy giảm tuyến tính 1.0 -> 0.05 qua 80% episode (giống linear). Không cập nhật replay buffer/batch (tabular không cần - buffer batch của linear vốn để đối phó bất ổn từ hàm xấp xỉ + bootstrap, tabular không có vấn đề đó).

## 3. Kết quả

### 3.1 Task 1 (coin-heaven) - kiểm tra đúng cơ chế trước khi đụng Task 2

5.000 episode train, 500 round eval (ε=0), **1 lần chạy** (chưa kiểm 3-seed - mục đích ở đây là gate đúng/sai cơ chế, không phải chọn hyperparameter tối ưu):

| coins TB | steps TB | suicide_rate |
|---|---|---|
| **50.00 / 50** | 190.5 | 0.000 |

Tuyệt đối - mọi round trong 500 round eval đều nhặt hết coin, không chết. Tốt hơn cả kết quả tốt nhất của linear (48.00+/-0.57/50, mục 4.1 `bao_cao_q_learning.md`). Hợp lý: trong `coin-heaven` state space thực chất rất nhỏ (chủ yếu chỉ `target_dir`x`stuck_bucket` biến thiên), và tabular không có sai số xấp xỉ nên hội tụ đúng chính sách tối ưu cho từng state một khi đã thăm đủ. Xác nhận cơ chế encode/update hoạt động đúng - đủ điều kiện chuyển sang Task 2.

### 3.2 Task 2 - loot-crate (bước đệm, crate 0.75, 50 coin)

3 lần chạy độc lập, 8.000 episode train / 500 round eval - **cùng ngân sách episode với baseline linear đã kiểm chứng** (`bao_cao_task2.md`, `stuck_penalty=0.5`, 0.85+/-0.14 coin, 53%+/-16% suicide) để so sánh công bằng:

| Lần | coins | steps TB | suicide_rate |
|---|---|---|---|
| 1 | 14.93 | 343.2 | 0.220 |
| 2 | 10.91 | 182.6 | 0.852 |
| 3 | 3.92 | 162.6 | 0.692 |
| **Trung bình** | **9.92 +/- 4.55** | - | **0.588 +/- 0.268** |

So với linear (0.85 +/- 0.14 coin): **ngay cả lần tệ nhất (3.92) cũng gấp ~4.6 lần** kết quả trung bình của linear; trung bình chung gấp **~11.7 lần**. Đổi lại, độ lệch chuẩn giữa 3 seed lớn hơn nhiều (4.55 so với 0.14 của linear) - tabular ở đây kém ổn định giữa các lần chạy hơn hẳn, và suicide_rate trung bình (58.8%) cũng cao hơn linear (53%), dao động mạnh theo seed (22% -> 85%).

### 3.3 Task 2 - classic (scenario chính thức, crate 0.75, chỉ 9 coin)

3 lần chạy độc lập, cùng cấu hình, cùng 8.000 episode - **đây là scenario mà linear thất bại tuyệt đối (0.00/0.00/0.00 coin cả 3 seed, xem `bao_cao_task2.md` mục 7)**:

| Lần | coins | steps TB | suicide_rate |
|---|---|---|---|
| 1 | 2.11 | 104.5 | 1.000 |
| 2 | 0.54 | 241.1 | 0.422 |
| 3 | 1.73 | 206.1 | 0.580 |
| **Trung bình** | **1.46 +/- 0.67** (/9) | - | **0.667 +/- 0.244** |

**Khác biệt định tính, không chỉ định lượng**: linear không chạm được coin nào trên `classic` ở bất kỳ seed nào trong số đã thử; tabular thu được trung bình 1.46/9 (~16% trần điểm) ở cả 3 lần chạy độc lập, không phải may mắn 1 lần. Đây là bằng chứng trực tiếp ủng hộ giả thuyết đặt ra ở `bao_cao_task2.md` mục 5: hàm xấp xỉ tuyến tính (không phải thiếu feature hay reward sai) là nút thắt chính khiến linear không transfer được từ `loot-crate` sang `classic`.

### 3.4 Bốn hướng cải thiện suicide_rate trên classic - ba thất bại, một thành công

Từ kết quả mục 3.3 (1.46/9 coin, suicide 66.7%), thử 4 hướng độc lập để giảm suicide mà không giảm coin, theo đúng phương pháp "một biến một lần, đo 3-seed":

**(a) Quét `DANGER_PENALTY_WEIGHT` (1.0 baseline -> 1.5 -> 2.0 -> 3.0), 3-seed mỗi giá trị**, mọi thứ khác giữ nguyên:

| DANGER_PENALTY_WEIGHT | coins TB | suicide_rate TB |
|---|---|---|
| 1.0 (baseline) | 1.46 +/- 0.67 | 66.7% +/- 24.4% |
| 1.5 | 0.92 +/- 0.76 | 89.6% +/- 4.3% |
| 2.0 | 0.00 +/- 0.00 | 0.0% +/- 0.0% |
| 3.0 | 0.00 +/- 0.00 | 0.0% +/- 0.0% |

**Thất bại, không đơn điệu**: không giá trị nào tốt hơn baseline. 1.5 làm suicide TỆ HƠN (89.6% so với 66.7%) thay vì tốt hơn; từ 2.0 trở lên chính sách sụp hoàn toàn về "không bao giờ đặt bomb" (0 coin, 0% suicide, luôn chạm trần 400 bước) - đúng kiểu sụp đã thấy khi thử n-step TD trên linear (mục 2.2). Đây là lần thứ 3 gặp đúng dạng ngưỡng chuyển pha này (lần 1: sweep danger_penalty trên linear; lần 2: n-step TD trên linear; lần 3: sweep này) - bắt đầu trông giống đặc điểm cấu trúc chung của cách `danger_penalty` tương tác với chuỗi hành vi "đặt bomb rồi thoát", không phải lỗi riêng của một model.

**(b) Debug trace để kiểm tra xem escape có phải nút thắt không**: chạy 15 round eval với log chi tiết từng bước (`Q_TABULAR_DEBUG=1`), giải mã ngược 319 state có `danger_now=1` xuất hiện trong log, so khớp `escape_dir` (hướng BFS gợi ý) với hành động Q-value cao nhất thực tế:

```
escape_dir khop voi argmax: 316 / 319 (99.1%)
```

**Bác bỏ giả thuyết "agent không theo tín hiệu thoát"**: chính sách đã học gần như luôn làm đúng theo hướng thoát được gợi ý. Vấn đề không nằm ở việc CHỌN sai hướng.

**(c) Dựa trên (b), giả thuyết mới: hướng đúng nhưng đường thoát không đủ nhanh so với BOMB_TIMER (4 bước)**. Sửa `can_bomb_here`: trước khi coi 1 vị trí là "tốt để đặt bomb", mô phỏng đặt bomb tại đó, cộng vùng nổ giả định vào vùng nguy hiểm hiện tại, rồi BFS kiểm tra có ô an toàn nào trong bán kính `BOMB_TIMER-1=3` bước không - chỉ giữ `can_bomb_here=1` nếu có. Test lại đầy đủ (loot-crate + classic, 3-seed mỗi cái, 8.000 episode, `DANGER_PENALTY_WEIGHT` về lại 1.0):

| Scenario | coins TB (v1 -> v2) | suicide_rate TB (v1 -> v2) |
|---|---|---|
| loot-crate | 9.92 +/- 4.55 -> **2.15 +/- 0.76** | 58.8% +/- 26.8% -> **74.7% +/- 20.8%** |
| classic | 1.46 +/- 0.67 -> **1.23 +/- 0.99** | 66.7% +/- 24.4% -> **97.8% +/- 1.8%** |

**Thất bại rõ ràng, cả 6/6 lần chạy đều tệ hơn**: coin giảm, suicide tăng mạnh ở cả 2 scenario, không phải nhiễu seed. Giả thuyết đúng về mặt lý luận (đã verify (b) trước khi sửa) nhưng sai về thực nghiệm. Cách diễn giải hợp lý nhất: làm `can_bomb_here=1` khắt khe hơn khiến nó xuất hiện HIẾM hơn hẳn - việc này làm loãng tín hiệu học "đặt bomb ở đây là tốt" (vốn đúng trong đa số trường hợp thực tế, kể cả những trường hợp bị lọc ra vì lý do timing) nhiều hơn là loại bỏ được tín hiệu xấu. Đã **revert về bản đơn giản ban đầu** (mục 2.1), xoá 2 hàm phụ trợ không còn dùng.

**(d) Tăng số episode train (8.000 -> 20.000) trên classic, 3-seed**, mọi cấu hình khác giữ nguyên (baseline đã revert):

| coins TB | suicide_rate TB |
|---|---|
| **1.47 +/- 0.32** | **82.1% +/- 13.3%** |

**Không cải thiện**: coin gần như y hệt baseline 8.000 episode (1.46), suicide không tốt hơn (có phần cao hơn, dù trong khoảng chồng lấn với baseline). Khác với dự đoán ban đầu (tabular có đảm bảo hội tụ lý thuyết tốt hơn linear nên train lâu hơn có thể giúp) - thực nghiệm cho thấy 8.000 episode đã gần mức bão hoà của thiết kế hiện tại, không phải do thiếu dữ liệu.

**(e) Curriculum learning: train trước trên `loot-crate` (coin dày, dễ học đặt bomb), rồi TIẾP TỤC cùng 1 Q-table trên `classic`** (thêm cờ `Q_TABULAR_CONTINUE=1` vào `callbacks.py` để nạp bảng đã có thay vì luôn khởi tạo lại khi `self.train=True`). Phase 1: 8.000 episode `loot-crate`. Phase 2: 8.000 episode `classic`, tiếp tục cùng bảng. Eval trên `classic`, 3-seed:

| Lần | coins | suicide_rate |
|---|---|---|
| 1 | 1.68 | 0.892 |
| 2 | 2.46 | 0.936 |
| 3 | 3.19 | 0.380 |
| **Trung bình** | **2.44 +/- 0.62** | **73.6% +/- 25.2%** |

**Thành công một phần - kết quả tốt nhất tìm được**: coin tăng ~67% so với baseline (1.46 -> 2.44), suicide không tệ hơn đáng kể (khoảng dao động chồng lấn baseline). Đây là hướng DUY NHẤT trong 4 hướng thử cải thiện được coin_mean. Trực giác hợp lý: `classic` có ít crate-dense-area hơn và chỉ 9 coin (thay vì 50) nên tín hiệu "đặt bomb đúng chỗ" tới rất thưa nếu học từ đầu trên chính `classic` - bắt đầu từ 1 Q-table đã có sẵn kinh nghiệm đặt-bomb-và-thoát tích luỹ trên `loot-crate` (nơi tín hiệu này dày hơn nhiều lần) giúp `classic` không phải học lại từ số 0, chỉ cần tinh chỉnh cho mật độ crate/coin khác.

## 4. Nhận xét

**Xác nhận giả thuyết state-aliasing**: toàn bộ hành trình debug Linear Q ở Task 2 (`bao_cao_task2.md`) xoay quanh việc phát hiện và vá lỗi bẫy dao động tất định giữa 2 vị trí có feature vector giống hệt nhau qua lăng kính linear - một dạng aliasing đặc thù của hàm xấp xỉ. Chuyển sang tabular (mỗi state có Q-value độc lập, không chia sẻ trọng số) loại bỏ tận gốc khả năng xảy ra kiểu bẫy đó, và kết quả xác nhận trực tiếp: cùng reward, cùng ngân sách episode, tabular vừa học nhanh hơn (Task 1: 50.00 tuyệt đối so với 48.00+/-0.57) vừa transfer được sang scenario mà linear hoàn toàn bó tay (`classic`: 1.46+/-0.67 so với 0.000).

**Suicide_rate là bài toán khó hơn dự kiến, không phải "chỉ cần tune 1 tham số"**: 4 hướng thử độc lập ở mục 3.4 cho thấy vấn đề không nằm ở (a) trọng số phạt nguy hiểm quá thấp - tăng nó chỉ làm sụp toàn bộ hành vi đặt bomb; (b) chọn sai hướng thoát - đã verify bằng debug trace, hướng thoát đúng 99% số lần; (c) `can_bomb_here` quá lỏng lẻo về mặt thời gian - sửa chặt hơn làm tệ hơn hẳn cả 6/6 lần thử; (d) chưa đủ dữ liệu train - gấp 2.5 lần episode không đổi được gì. Đòn bẩy DUY NHẤT thực sự hiệu quả là (e) đổi phân phối dữ liệu huấn luyện (curriculum learning), không phải chỉnh tham số reward hay feature. Bài học chung: khi 1 hướng "tune tham số hợp lý về lý thuyết" liên tục thất bại thực nghiệm theo cùng 1 kiểu (ngưỡng sụp/không đổi), nhiều khả năng nút thắt nằm ở PHÂN PHỐI TRẢI NGHIỆM huấn luyện chứ không phải trọng số/feature.

**Biến động giữa seed vẫn lớn** kể cả với curriculum learning (coins 1.68 -> 3.19, suicide 38.0% -> 93.6%) - lớn hơn hẳn linear ở cả Task 1 (độ lệch chuẩn ~0.57/48, ~1.2%) lẫn Task 2 (0.14/0.85, ~16%). Đặc điểm này xuất hiện xuyên suốt mọi cấu hình tabular đã thử (mục 3.2-3.4), không riêng gì 1 hướng - khả năng gắn với việc 600 state độc lập (không chia sẻ trọng số như linear) cần nhiều dữ liệu hơn để MỌI state hội tụ đều, đặc biệt state hiếm.

## 5. Hạn chế và hướng tiếp theo

- **Kết quả tốt nhất tìm được cho `classic`** (curriculum learning, mục 3.4-e: 2.44+/-0.62/9 coin, suicide 73.6%+/-25.2%) vẫn còn xa mức "giải tốt" - trung bình chỉ ~27% trần điểm, đa số round vẫn kết thúc bằng tự sát. Chưa phải ứng viên nộp bài, nhưng là điểm khởi đầu tốt hơn hẳn baseline ban đầu (mục 3.3) và hơn hẳn linear (0.000 tuyệt đối).
- Chưa quét hyperparameter cho curriculum learning (tỉ lệ episode phase 1/phase 2, thử phase 2 dài hơn phase 1, hay ngược lại) - mới thử đúng 1 tỉ lệ (8k/8k) đã cho kết quả tốt hơn hẳn, có khả năng còn tối ưu được thêm.
- Chưa tune `alpha`, `STUCK_PENALTY_WEIGHT` riêng cho tabular - toàn bộ kết quả trên dùng nguyên giá trị kế thừa từ linear hoặc chọn theo lý luận chung (mục 2.3), chưa quét thử.
- Biến động giữa seed lớn (mục 4) là rủi ro thực tế nếu phải chọn 1 bảng Q cụ thể để nộp bài - nên đánh giá thêm 3-5 seed nữa cho cấu hình curriculum trước khi chốt, và cân nhắc chọn seed tốt nhất trong nhiều lần train (không chỉ chạy 1 lần) làm bản nộp cuối.
- Chưa thử `rule_based_agent`/`coin_collector_agent`/`peaceful_agent` làm đối thủ (Task 3-4) cho agent này - mới dừng ở Task 1-2 (không có đối thủ), đúng phạm vi đã đặt ra. State hiện tại (mục 2.1) không mã hoá thông tin đối thủ, sẽ cần mở rộng trước khi thử Task 3-4.

## 6. Chi tiết tái lập

Code: `agent_code/q_tabular/{features,callbacks,train}.py` (tự chứa hoàn toàn, không import từ `q_linear`). Cờ `Q_TABULAR_CONTINUE=1` (trong `callbacks.py`) cho phép tiếp tục train từ 1 bảng Q có sẵn thay vì luôn khởi tạo lại - dùng cho curriculum learning. Công cụ thí nghiệm (không nộp bài): `run_tabular_config.py` (train+eval 1 scenario), `run_tabular_curriculum.py` (train 2 pha + eval). Log/bảng Q từng lần chạy: `agent_code/q_tabular/{log,table}_<tên>.csv|npy`. Kết quả eval thô: `results/eval_<tên>.json`. Bảng Q tốt nhất hiện tại cho `classic` (curriculum, lần 3, coins=3.19): `agent_code/q_tabular/table_curriculum_run3.npy`.
