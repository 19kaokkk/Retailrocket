# ML5 thuc nghiem gioi han thoi gian

Ban nay dung TAP CON, khong phai full va khong phai smoke 16 nguoi dung cu.
Chon truoc 500 nguoi co it nhat 10 su kien trong cua so train, 50 nguoi o moi
client trong 10 client chon ngau nhien. Moi phuong an dung cung cohort va ung vien.
Khong chon cohort, diem dung, epsilon hay checkpoint theo chat luong test.

Giu MB-SASRec hidden 50, 2 block, maxlen 50, dropout 0.5, catalog 114800 items.
3 round, 3 replicate, Centralized/FedAvg/FedProx, no_dp/clip_only/epsilon 1,3,5,8.
Batch muc tieu Centralized 32, FL 16; SGD lr .05; mu .01; clip 1; delta 1e-7.
Chi no_dp duoc tinh gradient theo batch. Phan DP giu nguyen code streaming da test.
Thoi gian thuc te chua duoc bao dam tu microbenchmark.

Dung trong workspace hien tai, can thu muc ../ML5_Recommendation/data va cac
file Python chia se. Khong sua cac file chia se khi dang chay. Nguon va config
co checksum; config/code thay doi can thu muc ket qua moi.

Notebook mac dinh CHI DOC ket qua, khong khoi chay them mot ban song song.
Dung full o VS Code truoc khi chay ban nay. Khong tat may/cho may ngu.
Neu co runner dang chay, khong Run Training them lan nua.

Lenh chay: python run_deadline_ml5.py --confirm-full-stopped --hours 4
Chuan bi cohort, khong train: python run_deadline_ml5.py --prepare-only

Ket qua tang dan o results/results_per_run_all_completed.csv va progress.json.
Sau moi ma tran 18 luot (mot replicate), xuat summary, bieu do, bao cao Markdown
va ML5_Subset_Results.zip. Ban tom tat chi dung cac replicate hoan thanh ca 18
cau hinh, khong tron so lan lap lech nhau hoac chon theo chat luong.
Guard thoi gian dung truoc luot moi neu khong du 7 phut; khong bao dam dung giua
mot round. Khong doi so round cua mot mo hinh dang chay de tiet kiem thoi gian.
Dung som/loi van giu nguyen checkpoint va budget reservations.

Khong so truc tiep voi ML4 full. Khong cam ket diem cao; noise co the lam giam
chat luong. Neu chi co mot replicate, khong co SD mau hay ket luan kiem dinh.

Chi accounting co che huan luyen co dieu kien tren roster/catalog/group co dinh.
Chon active users, cohort, tien xu ly, metric, rank, ID va baseline chua DP.
Khong co chung nhan user-level DP dau cuoi, client-level DP hoac local DP.
PRNG chua kiem toan mat ma. Phat hanh nhieu mo hinh/dung nhieu thu muc tren cung
nguoi can accounting cong gop; xoa ket qua khong xoa ngan sach da tieu.
