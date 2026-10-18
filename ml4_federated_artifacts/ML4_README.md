# Xem và chạy ML4 trong VS Code

Mở thư mục chứa notebook bằng **File > Open Folder**. Cài extension **Python** và
**Jupyter** của Microsoft, mở `ML4_Federated_Learning.ipynb`, chọn **Select Kernel**.
Notebook đã lưu sẵn bảng và ba biểu đồ, có thể xem ngay khi mở.

Chạy **Run All** với `RUN_TRAINING = False` để đọc kết quả đã có. Đổi thành `True`
để huấn luyện lại Centralized, FedAvg và FedProx. Nếu thiếu thư viện, bỏ dấu `#` ở
ô `%pip install -r requirements_ml4.txt` rồi chạy ô đó.

Thư mục `ml4_results` chứa:

- `ml4_results_summary.csv`: kết quả validation/test của ba mô hình.
- `ml4_training_history.csv`: chỉ số validation qua 20 round của FedAvg/FedProx.
- `ml4_experiment_summary.json`: cấu hình và tham số mô hình đã lưu.
- `ml4_model_comparison.png`: so sánh sáu chỉ số trên test.
- `ml4_training_curves.png`: PR-AUC/F1 validation theo round.
- `ml4_client_distribution.png`: số phiên train từng client và purchase rate trên tổng ba split.
- `ML4_Federated_Learning_Report.md`: nội dung 4.3.5.1 đến 4.3.5.6, có bảng và hình.

Chạy lại bằng Terminal trong thư mục notebook:

```powershell
python -m pip install -r requirements_ml4.txt
python run_ml4_federated.py --data-dir . --output-dir ml4_results --rounds 20 --local-epochs 3 --lr 0.05 --fedprox-mu 0.1
```

Chỉ tạo lại biểu đồ từ kết quả đã lưu:

```powershell
python run_ml4_federated.py --data-dir . --output-dir ml4_results --plots-only
```

Các số hiện tại là một lần chạy, không có error bar qua nhiều seed. FedAvg/FedProx
rất gần nhau nên hai đường có thể trùng ở độ chính xác của hình. Không có lịch sử
epoch centralized, chỉ có kết quả cuối; notebook không dựng thêm đường không được ghi.
PR-AUC là average precision của sklearn. Không so trực tiếp ML4 với ML5 để quy mọi
chênh lệch cho DP, vì ML5 dùng loss theo người và cách biến đổi/lịch tối ưu khác.

Để chạy lại cần ba file `ml4_train.csv.gz`, `ml4_val.csv.gz`, `ml4_test.csv.gz` và
`client_partition_metadata.csv` trong thư mục notebook. Không giải nén `.gz`.
Đóng CSV trong Excel trước khi ghi đè kết quả.
