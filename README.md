# Federated Learning Project using Flower, PyTorch, and MLflow

Dự án này triển khai một hệ thống Federated Learning (Học tập liên hợp) sử dụng framework Flower, PyTorch và tích hợp MLflow để quản lý mô hình (MLops). 

Hệ thống huấn luyện mô hình phân loại ảnh SimpleCNN trên bộ dữ liệu CIFAR-100 với cấu hình mặc định gồm 1 Server và 2 Clients chạy song song.

---

## Kiến trúc hệ thống

Dưới đây là sơ đồ luồng hoạt động và kết nối giữa các thành phần trong mạng:

```mermaid
graph TD
    subgraph Môi trường chạy (Docker Network / Localhost)
        mlflow[MLflow Server <br> Cổng 5000]
        server[Flower Server <br> Cổng 8080]
        client1[Flower Client 1 <br> ID: 0]
        client2[Flower Client 2 <br> ID: 1]
        
        server -->|Ghi log cấu hình & metrics| mlflow
        client1 -->|Kết nối / Nhận trọng số / Gửi cập nhật| server
        client2 -->|Kết nối / Nhận trọng số / Gửi cập nhật| server
    end
    
    subgraph Bộ lưu trữ dữ liệu
        shared_data[(Thư mục dữ liệu chung <br> ./data)]
        client1 -.->|Đọc phân vùng dữ liệu train 1| shared_data
        client2 -.->|Đọc phân vùng dữ liệu train 2| shared_data
        server -.->|Đọc dữ liệu test tập trung| shared_data
    end
```

Luồng xử lý chính:
1. **Server** khởi động và chờ kết nối từ ít nhất 2 Clients.
2. Các **Clients** kết nối thành công, Server gửi trọng số ban đầu của mô hình toàn cục (Global Model) xuống các Clients.
3. Mỗi Client thực hiện huấn luyện cục bộ (Local Training) trên phân vùng dữ liệu CIFAR-100 được chia riêng biệt.
4. Clients gửi các cập nhật trọng số mới về Server.
5. Server tổng hợp trọng số bằng thuật toán **FedAvg** để tạo mô hình toàn cục mới.
6. Server đánh giá hiệu năng của mô hình toàn cục trên tập test trung tâm (Centralized Evaluation) và ghi nhận kết quả (Loss, Accuracy) lên **MLflow Server**.
7. Lặp lại quy trình trên qua số Rounds cấu hình sẵn.

---

## Cấu trúc thư mục

```text
g:\Federated_Learning/
├── .gemini/
│   ├── implementation_plan.md  # Kế hoạch triển khai dự án
│   └── todo.md                 # Tiến độ thực hiện công việc
│
├── src/
│   ├── __init__.py
│   ├── model.py                # Định nghĩa kiến trúc SimpleCNN và hàm train/test
│   ├── dataset.py              # Tải dữ liệu CIFAR-100 và chia dữ liệu (I.I.D)
│   ├── server.py               # Thiết lập Flower Server & Logging sang MLflow
│   └── client.py               # Cấu hình Flower Client
│
├── docker/
│   ├── Dockerfile.server       # Môi trường chạy Flower Server
│   └── Dockerfile.client       # Môi trường chạy Flower Client
│
├── docker-compose.yml          # Docker Compose điều phối chạy hệ thống
├── requirements.txt            # Danh sách thư viện Python phụ thuộc
└── README.md                   # Hướng dẫn chạy và kiến trúc dự án
```

---

## Cách vận hành hệ thống

### Cách 1: Chạy bằng Docker Compose (Khuyên dùng)

Cách chạy này tự động thiết lập MLflow, Flower Server và 2 Clients song song.

1. **Khởi chạy:**
   Chạy lệnh sau tại thư mục gốc của dự án:
   ```bash
   docker compose up -d --build
   ```

2. **Cách thức hoạt động:**
   - Container `mlflow-server` khởi chạy và mở cổng `5000`.
   - Container `flower-server` khởi chạy, lắng nghe cổng `8080`.
   - `client-1` và `client-2` kết nối tới `flower-server` và cùng chia sẻ thư mục lưu trữ `./data` trên máy host để tránh tải lại dữ liệu CIFAR-100 nhiều lần.
   - Cơ chế `restart: on-failure` giúp các client tự kết nối lại nếu server khởi động chậm hơn.

3. **Dừng hệ thống:**
   ```bash
   docker compose down
   ```

### Cách 2: Chạy trực tiếp trên máy cục bộ (Local)

1. **Khởi tạo môi trường ảo và cài đặt thư viện:**
   ```bash
   python -m venv venv
   # Kích hoạt venv (Windows)
   .\venv\Scripts\activate
   # Cài đặt dependencies
   pip install -r requirements.txt
   ```

2. **Chạy MLflow Server:**
   Mở terminal thứ nhất, kích hoạt venv và chạy:
   ```bash
   mlflow server --host 0.0.0.0 --port 5000
   ```

3. **Chạy Flower Server:**
   Mở terminal thứ hai, kích hoạt venv và chạy:
   ```bash
   python src/server.py --rounds 5 --server-address 0.0.0.0:8080 --mlflow-uri http://localhost:5000
   ```

4. **Chạy 2 Clients (chạy song song trên 2 terminal mới):**
   - **Client 1:**
     ```bash
     python src/client.py --client-id 0 --server-address 127.0.0.1:8080 --epochs 2 --lr 0.001
     ```
   - **Client 2:**
     ```bash
     python src/client.py --client-id 1 --server-address 127.0.0.1:8080 --epochs 2 --lr 0.001
     ```

---

## Giám sát kết quả trên MLflow

1. Mở trình duyệt và truy cập địa chỉ: `http://localhost:5000`
2. Chọn Experiment **`Federated Learning - Flower & CIFAR-100`** ở danh sách bên trái.
3. Xem các thông tin trong Run hiện tại:
   - **Parameters:** Các tham số cấu hình như `num_rounds`, `dataset` (CIFAR-100), `device` và mô hình `SimpleCNN`.
   - **Metrics:** Đồ thị trực quan hóa của `global_loss` và `global_accuracy` biến động qua các round huấn luyện.
