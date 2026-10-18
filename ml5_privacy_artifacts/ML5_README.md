# Chạy ML5 trong VS Code

Mở thư mục `outputs` bằng **File > Open Folder**. Cài hai extension **Python** và
**Jupyter** của Microsoft. Mở `ML5_Privacy_Utility.ipynb`, chọn **Select Kernel**,
chọn môi trường Python rồi chạy các ô từ trên xuống. Nếu thiếu thư viện, bỏ dấu
`#` ở ô cài đặt và chạy ô đó. Notebook có sẵn phần đọc kết quả đã chạy, nên có thể
xem kết quả trước khi huấn luyện lại.

Các file cần để huấn luyện:

- `run_ml5_privacy.py`, `test_ml5_privacy.py`, `requirements_ml5.txt`.
- `ml4_train.csv.gz`, `ml4_val.csv.gz`, `ml4_test.csv.gz` từ ML4. Không cần giải nén `.gz`.
- `ML5_Privacy_Utility.ipynb`.

Notebook mặc định `RUN_TRAINING = False` để xem kết quả có sẵn. Đổi thành `True`
để chạy lại toàn bộ thí nghiệm. Số lần chạy là 3, với ε mục tiêu 1, 3, 5, 8,
ba thuật toán Centralized, FedAvg, FedProx và hai baseline mỗi thuật toán.
Thời gian phụ thuộc máy; tổng cộng 54 mô hình. Kết quả tạo trong `ml5_results`.
Đóng file CSV đang mở trong Excel trước khi chạy lại để tránh khóa file.

Chạy bằng Terminal của VS Code cũng được:

```powershell
python -m pip install -r requirements_ml5.txt
python test_ml5_privacy.py
python run_ml5_privacy.py --data-dir . --output-dir ml5_results
```

Chạy nhanh thử luồng (kết quả mới lưu riêng, không thay bản đầy đủ):

```powershell
python run_ml5_privacy.py --data-dir . --output-dir ml5_quick --replicates 1 --epsilon-targets 3 --epochs 1 --rounds 2
```

## Phương pháp và giả định

Mỗi mẫu dự đoán là một session, nhưng mỗi đơn vị lấy mẫu/cắt gradient là một người:
lấy trung bình gradient trên TẤT CẢ phiên train của `visitorid`, sau đó Opacus cắt
gradient người dùng và thêm Gaussian. Không cắt từng phiên rồi gọi là user-level DP.
Gradient logistic được tính bằng công thức đóng để chạy nhanh trên CPU; bài kiểm tra
đối chiếu với PyTorch autograd kiểm tra cả người có nhiều phiên, cắt chung weight/bias,
batch rỗng và ngân sách xuyên round. Không giới hạn/truncate phiên của một người.

Lấy mẫu: K ~ Binomial(N,q), chọn K người khác nhau đồng đều. Đây là cùng phân phối
với lấy mẫu Bernoulli(q) độc lập từng người. Một lượt là ceil(N/batch_users) bước,
q = 1/ceil(N/batch_users). Chuẩn hóa tổng gradient bằng qN cố định, kể cả batch rỗng.

Tiền xử lý dùng caps cố định `[1800,100,100,20,100,1,100]` cho bảy đặc trưng ML4,
rồi log1p(x)/log1p(cap); không lấy mean/std/quantile nhạy cảm. Các caps là giả định
miền dữ liệu, không phải siêu tham số được tối ưu bằng test. Purchase weight 100,
C=100 và ngưỡng phân loại 0.5 đều cố định. Loss theo người khác loss theo session
trong ML4, vì vậy baseline không DP được chạy lại trong ML5.

FedAvg/FedProx dùng mọi client mỗi round, trung bình tham số theo số người của client.
FedProx cộng proximal trên weight và bias SAU khi Opacus cắt và thêm nhiễu. Hệ số μ=0.1.
Ngân sách được cộng qua tất cả local step/round của client; max ε_client theo parallel
composition chỉ áp dụng vì các nhóm người rời nhau và roster/weights được coi công khai.
Không dùng secure aggregation; server có thể thấy mô hình cục bộ đã thêm nhiễu.

## Đọc kết quả

- `ml5_results_summary.csv`: mean/std utility, σ, ε thực tế, retention, loss do nhiễu.
- `ml5_results_per_run.csv`: các lần chạy riêng.
- `ml5_privacy_ledger.csv`: q, số bước, ε từng client/từng cấu hình.
- `ml5_statistical_checks.csv`: paired t-test theo nhãn lần chạy và Spearman thăm dò.
- `ml5_privacy_choices.csv`: ε nhỏ nhất đáp ứng ví dụ giữ 95% PR-AUC validation.
- `ml5_experiment_summary.json`: cấu hình, phiên bản thư viện, tham số mô hình và
  ngân sách compose của TẤT CẢ mô hình DP trong lần chạy.
- `ML5_Privacy_Utility_Report.md`: đủ 4.3.6.1 đến 4.3.6.5, có bảng và hình.

PR-AUC được tính bằng average precision. Retention so với baseline không DP cùng
thuật toán. Error bar là sample std của ba lần chạy, không phải khoảng tin cậy.
Các thử nghiệm thống kê với ba lần chạy chỉ có tính tham khảo.

## Không tự nhận bảo đảm riêng tư toàn hệ thống

Tùy chọn `--public-roster roster.csv` nhận danh sách `visitorid,client_id` công khai
từ bên ngoài và giữ slot gradient bằng 0 khi người không còn phiên train. Mặc định
lấy roster từ benchmark đã có, thao tác trích roster này không được privatize.
Kiểm tra kèm theo xác nhận xóa mọi phiên của một người vẫn giữ N/q nếu roster cố định.

ε là phép accounting của cơ chế lý tưởng theo người, với adjacency một người có
hoặc không đóng góp các phiên TRAIN trong một roster công khai cố định. Xóa người
rồi tính lại roster, N, q hoặc trọng số nằm ngoài giả định. Tiền xử lý từ raw events,
thống kê validation/test, metadata client và baseline không DP chưa được privatize.
Người xuất hiện ở nhiều split có thể bị lộ qua metric chưa DP.

Mặc định các lần chạy dùng nguồn PRNG độc lập, seed ngẫu nhiên không lưu/phát hành.
Các lần chạy lại tái lập PHƯƠNG PHÁP nhưng không cho số giống tuyệt đối. Tùy chọn
`--deterministic` tạo seed công khai để debug, **không phải bản phát hành DP**.
Secure mode và sampler chưa kiểm toán CSPRNG/số thực, do đó chưa chứng nhận user-level
DP khi triển khai. Cần kiểm toán adjacency, sampling, clipping, RNG, preprocessing,
metrics và composition trước khi tuyên bố bảo đảm. Công bố cả baseline không DP làm
toàn bộ gói kết quả không có ε hữu hạn, kể cả khi các mô hình DP riêng có ε được tính.

Nguồn: [Opacus DPOptimizer](https://opacus.ai/api/optim/dp_optimizer.html),
[RDP accountant](https://opacus.ai/api/_modules/opacus/accountants/rdp.html),
[secure RNG](https://opacus.ai/api/privacy_engine.html),
[DP-SGD](https://arxiv.org/abs/1607.00133), [FedProx](https://arxiv.org/abs/1812.06127).
