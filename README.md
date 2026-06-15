# Dự án Federated Learning End-to-End với Flower, PyTorch & MLflow

Dự án này triển khai một hệ thống Học tập liên hợp (Federated Learning) hoàn chỉnh sử dụng thư viện **Flower (flwr)**, framework **PyTorch**, và tích hợp công cụ giám sát **MLflow** cho MLops.

Mô hình huấn luyện là một mô hình mạng CNN tự định nghĩa (`SimpleCNN`) thực hiện nhiệm vụ phân loại ảnh trên dataset độ khó cao **CIFAR-100** (chứa 100 lớp nhãn khác nhau). Quá trình huấn luyện sử dụng chiến lược **FedAvg** để tổng hợp trọng số từ 2 clients chạy song song và đánh giá tập trung (Centralized Evaluation) trên server.

---

## 📂 Cấu trúc thư mục dự án

```text
g:\Federated_Learning/
│
├── .gemini/
│   ├── implementation_plan.md  # Kế hoạch triển khai
│   └── todo.md                 # Trạng thái công việc
│
├── src/
│   ├── __init__.py
│   ├── model.py                # Định nghĩa SimpleCNN và các hàm train/test
│   ├── dataset.py              # Tải, chuẩn hóa và phân chia dữ liệu CIFAR-100
│   ├── server.py               # Thiết lập Flower Server & Tích hợp MLflow
│   └── client.py               # Thiết lập Flower Client kế thừa NumPyClient
│
├── docker/
│   ├── Dockerfile.server       # Dockerfile xây dựng môi trường cho Server
│   └── Dockerfile.client       # Dockerfile xây dựng môi trường cho Client
│
├── docker-compose.yml          # Điều phối chạy toàn bộ hệ thống
├── requirements.txt            # Danh sách thư viện Python cần thiết
└── README.md                   # Tài liệu hướng dẫn sử dụng (File này)
```

---

## 🛠️ Yêu cầu chuẩn bị
- Đã cài đặt **Python 3.10+** (nếu muốn chạy trực tiếp cục bộ).
- Đã cài đặt **Docker** và **Docker Compose** (nếu muốn chạy qua container).

---

## 🚀 Hướng dẫn cách chạy hệ thống

Theo yêu cầu của bạn, các câu lệnh dưới đây bạn sẽ tự thực hiện trên môi trường của mình (cmd, powershell hoặc docker terminal).

### Cách 1: Chạy bằng Docker Compose (Khuyên dùng - Đơn giản nhất)

Phương pháp này sẽ tự động xây dựng các containers chạy MLflow Server, Flower Server và 2 Clients song song.

1. **Khởi chạy hệ thống:**
   Mở terminal tại thư mục gốc của dự án (`g:\Federated_Learning`) và chạy lệnh sau để build và khởi chạy tất cả services:
   ```bash
   docker-compose up --build
   ```

2. **Cách thức hoạt động trong Docker Compose:**
   - Container `mlflow-server` được dựng và mở cổng `5000`.
   - Container `flower-server` khởi động, chờ kết nối trên cổng `8080` và kết nối với MLflow để ghi nhật ký.
   - Các container `client-1` và `client-2` được khởi động song song, tự động tải/truy xuất dữ liệu từ thư mục dùng chung `./data` trên máy host để tránh việc download lặp lại, sau đó kết nối tới Server để bắt đầu huấn luyện.
   - Quá trình kết nối của clients sử dụng cơ chế `restart: on-failure` để đảm bảo kết nối lại nếu Server chưa sẵn sàng.

3. **Tắt hệ thống:**
   Để dừng tất cả containers và xóa tài nguyên, chạy:
   ```bash
   docker-compose down
   ```

---

### Cách 2: Chạy trực tiếp trên máy cục bộ (Local)

Nếu bạn muốn chạy từng bước trên máy của mình mà không cần Docker:

1. **Tạo môi trường ảo và cài đặt thư viện:**
   ```bash
   # Tạo môi trường ảo python
   python -m venv venv
   
   # Kích hoạt môi trường ảo (Windows)
   .\venv\Scripts\activate
   
   # Cài đặt các thư viện yêu cầu
   pip install -r requirements.txt
   ```

2. **Khởi chạy MLflow Server:**
   Mở một terminal mới, kích hoạt môi trường ảo và chạy:
   ```bash
   mlflow ui --port 5000
   ```
   *Lúc này bạn có thể truy cập `http://localhost:5000` để xem giao diện MLflow.*

3. **Khởi chạy Flower Server:**
   Mở một terminal khác, kích hoạt môi trường ảo và chạy:
   ```bash
   python src/server.py --rounds 5 --server-address 0.0.0.0:8080 --mlflow-uri http://localhost:5000
   ```
   *Server sẽ chờ kết nối từ ít nhất 2 clients.*

4. **Khởi chạy 2 Clients (2 tiến trình riêng biệt):**
   - **Client 1:** Mở terminal mới, kích hoạt môi trường ảo và chạy:
     ```bash
     python src/client.py --client-id 0 --server-address 127.0.0.1:8080 --epochs 2 --lr 0.001
     ```
   - **Client 2:** Mở terminal mới, kích hoạt môi trường ảo và chạy:
     ```bash
     python src/client.py --client-id 1 --server-address 127.0.0.1:8080 --epochs 2 --lr 0.001
     ```

---

## 📊 Theo dõi và xem kết quả trên MLflow

Sau khi hệ thống bắt đầu huấn luyện, Flower Server sẽ tự động đánh giá mô hình toàn cục sau mỗi vòng huấn luyện (Round) và log các chỉ số lên MLflow.

1. Truy cập vào địa chỉ **`http://localhost:5000`** trên trình duyệt của bạn.
2. Tìm Experiment: **`Federated Learning - Flower & CIFAR-100`**.
3. Chọn Run đang chạy hoặc đã chạy xong để xem:
   - **Parameters:** Cấu hình huấn luyện (`num_rounds`, `device`, `dataset`, `model_architecture`).
   - **Metrics:** Biểu đồ thay đổi của **`global_loss`** và **`global_accuracy`** qua các vòng huấn luyện (rounds).

---

## 📝 Giải thích Chi tiết các thành phần

- **`src/model.py` (`SimpleCNN`):** Sử dụng 3 lớp Convolutional kết hợp với Batch Normalization và Max Pooling để trích xuất đặc trưng của ảnh 32x32. Sau đó chuyển qua lớp fully connected với Dropout để hạn chế tối đa Overfitting trước khi đưa ra 100 lớp nhãn của CIFAR-100.
- **`src/dataset.py`:** Hàm `load_datasets` tự động tải tập CIFAR-100 về máy. Khi huấn luyện Federated Learning, nó chia ngẫu nhiên (I.I.D) tập train (50k ảnh) làm 2 nửa riêng biệt cho 2 clients (mỗi client giữ 25k ảnh) nhằm mô phỏng bài toán dữ liệu phân tán.
- **`src/server.py`:** Cấu hình Flower Server sử dụng thuật toán tổng hợp trọng số phổ biến `FedAvg`. Tích hợp hàm `evaluate_fn` để sau khi kết thúc một round huấn luyện, Server sẽ trực tiếp dùng tập test trung tâm (10k ảnh) để đo lường độ chính xác và mất mát toàn cục, giúp biểu đồ MLflow phản ánh đúng sức mạnh thực tế của mô hình liên hợp.
- **`src/client.py`:** Đóng vai trò là Flower Client kế thừa `flwr.client.NumPyClient`. Mỗi client tự huấn luyện cục bộ bằng PyTorch trên tập dữ liệu của riêng mình trong `epochs=2` vòng lặp với thuật toán tối ưu `AdamW`, sau đó gửi cập nhật trọng số hiệu chỉnh về Server để tổng hợp.
