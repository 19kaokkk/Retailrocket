# ML4 Gợi ý sản phẩm với Federated Learning

Đây là nhánh ML4 mới cho **gợi ý sản phẩm được xem tiếp theo**, sử dụng MB-SASRec
với lịch sử view, addtocart và transaction. Không phải mô hình dự đoán mua theo
phiên trong bộ ML4/ML5 trước. Hai bộ kết quả được giữ riêng, không thay thế số
liệu hoặc đổi tên bài toán của nhau.

## Mở bằng VS Code

1. Giải nén toàn bộ `ML4_Recommendation_Package.zip`.
2. Trong VS Code chọn **File > Open Folder**, mở thư mục `ML4_Recommendation`.
3. Mở `ML4_Recommendation_FL.ipynb`, chọn **Select Kernel** và Python của bạn.
4. Nếu thiếu thư viện, chạy ô `%pip install -r requirements.txt`, sau đó restart kernel.
5. Chọn **Run All**. Mặc định notebook đọc kết quả chạy thử đã có; không tự khởi
   động một thí nghiệm lớn.

Yêu cầu Python 3.11 trở lên. Notebook cũng có thể mở bằng Colab; khi đó cần đưa
nguyên thư mục lên Drive và đặt `PROJECT_DIR` tới thư mục đó. Không chỉ tải riêng
notebook vì nó cần code và các mảng trong `data`.

## Chạy đầy đủ

Trong ô cấu hình notebook, đổi:

```python
MODE = "full"
RUN_TRAINING = True
```

Sau đó chạy tiếp các ô phía dưới. Bộ đầy đủ dùng cả 50 client, 115.150 khách hàng
train có ít nhất hai sự kiện đa hành vi trước validation target, 172.513 khách
hàng đánh giá, ba seed 2026/2027/2028, 10 round, một lượt dữ liệu cục bộ/round.
Mô hình có hidden size 50, hai block, một head, dropout 0.5, maxlen 50; batch 128,
Adam learning rate 0.001, clip chuẩn gradient toàn batch là 5. FedProx dùng μ=0.01
trên mọi tham số huấn luyện. Giá trị μ này được đặt trước, chưa được tối ưu bằng
quét siêu tham số; không chọn lại theo test.

CPU có thể chạy về mặt chức năng; bộ đầy đủ cần nhiều thời gian và bộ nhớ hơn
chạy thử. Chưa đo thời gian đầy đủ nên không đưa ra cam kết thời lượng. Có thể dùng
kernel PyTorch hỗ trợ CUDA trên máy có GPU hoặc Colab. Checkpoint lưu sau mỗi round;
chạy lại cùng cấu hình/thư mục sẽ tiếp tục từ round hoàn tất. Không dùng chung thư
mục kết quả khi thay cấu hình. Nếu thay code hoặc cấu hình, chọn thư mục kết quả mới.

Chạy ngoài notebook:

```powershell
python run_ml4_recommendation.py --profile smoke
python run_ml4_recommendation.py --profile full
python test_ml4_recommendation.py
```

## Phân biệt kết quả

- `results_smoke`: **chỉ kiểm tra kỹ thuật**, 96 khách hàng train thuộc ba client,
  88 khách hàng đánh giá đủ điều kiện, hai round, một seed. Không dùng làm kết quả
  cuối để trả lời RQ4, không so trực tiếp với bảng ML3.
- `results_full`: chỉ được tạo khi bạn chạy cấu hình đầy đủ; bộ bàn giao **chưa có
  kết quả huấn luyện đầy đủ** và không giả lập số liệu.
- Trong mỗi thư mục kết quả: `results_summary.csv`, `results_per_run.csv`,
  `model_comparison.png`, `training_curves.png`, `client_distribution.png`,
  `ML4_Recommendation_Report.md`, cùng checkpoint/rank/history từng seed.

Một seed không có độ lệch chuẩn mẫu; CSV để trống độ lệch chuẩn, không diễn giải
thanh sai số 0 trên biểu đồ chạy thử là bằng chứng ổn định.

## Những gì đã đối chiếu với ML3

Dữ liệu nguồn có 2.755.637 sự kiện, trong đó 2.664.215 view, 68.966 addtocart và
22.456 transaction. Có 190.980 khách hàng đủ điều kiện theo bộ view ML3;
114.800 sản phẩm trong train catalog. Toàn bộ chuỗi view train, chuỗi view
validation/test, mục tiêu và ma trận ứng viên đã được đối chiếu với file ML3.
ML3 view-only có 113.451 khách hàng tạo được cặp train; đa hành vi có 115.150.
Không chuyển cứng số 113.451 thành kích thước train của MB-SASRec.

Pipeline mới giữ kiến trúc MB-SASRec (add) từ `Big (1).py` và benchmark đánh giá
ML3, nhưng **không nhận các điểm SASRec đã lưu làm baseline Centralized mới**.
Centralized MB-SASRec phải huấn luyện cùng cấu hình, dữ liệu và loss với FL.

Khác biệt có chủ đích với vòng lặp ML3: mỗi khách hàng đóng góp một cửa sổ train
gần nhất dài tối đa 51 sự kiện, sinh 50 vị trí input/target. Loss được lấy trung
bình trên các vị trí hợp lệ của từng khách hàng, rồi trung bình khách hàng;
không trung bình tất cả token chung trong batch như code ML3. Một epoch đi qua
mỗi khách hàng train một lần bằng shuffle không hoàn lại; không lấy mẫu khách
hàng có hoàn lại như `make_batch` gốc. Mục tiêu train gồm next-event ở cả ba loại
hành vi; mục tiêu đánh giá vẫn là next-view. Phải công bố những khác biệt này.

## Phép so sánh Centralized và FL

Mọi phương án dùng cùng seed khởi tạo, kiến trúc, loss, tập khách hàng, số lượt
khách hàng và benchmark. Centralized giữ trạng thái Adam xuyên các round;
FL khởi tạo Adam cục bộ lại ở mỗi client/round. Chia minibatch ở ranh giới client
cũng làm số bước tối ưu khác nhau; cả số bước lẫn lượt khách hàng đều được lưu.
Đây là so sánh theo ngân sách lượt khách hàng, không phải chứng minh các quỹ đạo
tối ưu hoàn toàn tương đương. Không dùng early stopping để vô tình cấp số lượt
train khác nhau; checkpoint tốt nhất được chọn bằng validation NDCG@10 trong
cùng lịch cố định, sau đó mới đánh giá test.

Mỗi visitorid ánh xạ vào một trong 50 client bằng MD5 giống bộ chuẩn bị dữ liệu
trước. FL tổng hợp theo **số khách hàng train**, không theo số phiên hay token.
Mọi client hoạt động trong cấu hình đều tham gia mỗi round. Đây là mô phỏng một
máy, không phải triển khai mạng hay xác nhận dữ liệu được cách ly trên thiết bị.

## Giới hạn benchmark

- Chia hai lượt view cuối của từng khách hàng cho validation/test. Đây không
  phải chia tại một mốc thời gian chung; train của một người có thể muộn hơn
  test của người khác. Không tuyên bố đã chứng minh không có rò rỉ theo thời
  gian toàn hệ thống hoặc đánh giá khách hàng hoàn toàn mới.
- Test history được phép chứa validation target và các hành vi xảy ra trước
  test target, đúng giao thức ML3. Không đưa test target vào train/đầu vào test.
- 40 sự kiện train trùng timestamp với validation target của cùng khách hàng;
  thứ tự được phân định theo vị trí trong file nguồn bằng stable sort. Cần quy
  định lại thứ tự nếu triển khai trực tuyến yêu cầu thời gian nghiêm ngặt.
- Giữ nguyên 101 ứng viên ML3: một positive và 100 negatives. Không coi HR@10
  hay NDCG@10 này là kết quả xếp hạng toàn bộ 114.800 sản phẩm. HR@10 bằng
  Recall@10 trong benchmark chỉ có một positive mỗi khách hàng.
- Negatives ML3 loại lịch sử view, không loại toàn bộ lịch sử đa hành vi.
  Có 5 negative ở validation và 3 ở test đã được quan sát qua hành vi không phải
  view. Giữ nguyên để không thay benchmark, đồng thời công khai hạn chế này.
- Validation giữ so sánh điểm `>` như code chọn checkpoint ML3. Test dùng
  đúng tie priorities đã cung cấp. Không thay cách phá hòa giữa thuật toán.
- Client cân bằng số lượng không chứng minh mức non-IID thấp. Chưa quét μ,
  chưa đo non-IID có kiểm soát và chưa đo chi phí truyền thông mạng thực.

## Tái tạo từ năm ZIP đầu vào

Code chuẩn bị dữ liệu chỉ giải nén các entry cần thiết; kiểm tra đường dẫn,
trùng tên, symlink, giới hạn kích thước và SHA-256. Pickle chỉ chấp nhận dữ liệu
thuần, không cho khởi tạo đối tượng hoặc chạy hàm. Không chạy code trong ZIP.
Không cần pyarrow vì chuỗi được tái dựng trực tiếp từ CSV và đối chiếu các file
SASRec; các Parquet gốc không bị chỉnh sửa và không được dùng để huấn luyện mới.

```powershell
python prepare_recommendation_data.py --archive-dir "C:\duong_dan\nam_zip" --inputs "inputs_ml3" --output "data"
```

`data_manifest.json` lưu fingerprint ZIP, hash các mảng và các kết quả kiểm tra.
Với bộ bàn giao đã có `data`, không cần thực hiện lại bước này.

## Ranh giới với ML5

Chưa triển khai DP-SGD cho mô hình gợi ý trong bộ ML4 này. Gradient clipping
toàn batch của ML4 không phải cắt gradient theo khách hàng và không tạo DP.
Việc giữ nhóm visitor và chuẩn hóa loss theo khách hàng chỉ là tiền đề thiết
kế ML5. Phải quyết định và kiểm toán privacy unit, adjacency, lấy mẫu, clipping,
nhiễu, accountant, roster/catalog và chính sách công bố đầu ra. Không được coi
checkpoint không DP là khởi tạo miễn phí về riêng tư nếu nó đã học trên cùng
dữ liệu cần bảo vệ. Bộ ML5 dự đoán mua cũ không chứng nhận DP cho hệ gợi ý mới.

## Nguồn phương pháp

- FedAvg: https://proceedings.mlr.press/v54/mcmahan17a.html
- FedProx: https://arxiv.org/abs/1812.06127
- Kiến trúc MB-SASRec: mã nguồn ML3 do nhóm cung cấp.
