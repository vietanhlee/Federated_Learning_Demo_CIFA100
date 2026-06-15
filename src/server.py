import os
import argparse
from collections import OrderedDict
import torch
import flwr as fl
from flwr.common import parameters_to_ndarrays
import mlflow

from src.model import SimpleCNN, test
from src.dataset import load_datasets

def get_evaluate_fn(testloader: torch.utils.data.DataLoader, device: torch.device):
    """Trả về hàm evaluate_fn để Flower Server đánh giá mô hình tập trung (Centralized Evaluation) sau mỗi round."""
    
    def evaluate(server_round: int, parameters: fl.common.Parameters, config: dict):
        # Khởi tạo mô hình cục bộ trên Server để đánh giá
        model = SimpleCNN(num_classes=100)
        
        # Nạp trọng số từ các clients (dạng Parameters) vào mô hình PyTorch
        ndarrays = parameters_to_ndarrays(parameters)
        params_dict = zip(model.state_dict().keys(), ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        # Đánh giá hiệu suất mô hình trên tập test trung tâm
        loss, accuracy = test(model, testloader, device)
        print(f"\n--- Round {server_round} central evaluation ---")
        print(f"Global Loss: {loss:.4f} | Global Accuracy: {accuracy:.4f}\n")
        
        # Log kết quả huấn luyện lên MLflow
        try:
            mlflow.log_metric("global_loss", loss, step=server_round)
            mlflow.log_metric("global_accuracy", accuracy, step=server_round)
        except Exception as e:
            print(f"Cảnh báo: Không thể log metrics lên MLflow. Lỗi: {e}")
            
        return loss, {"accuracy": accuracy}
        
    return evaluate

def main():
    parser = argparse.ArgumentParser(description="Flower Federated Learning Server")
    parser.add_argument("--rounds", type=int, default=5, help="Số vòng huấn luyện Federated Learning (mặc định: 5)")
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8080", help="Địa chỉ chạy server (mặc định: 0.0.0.0:8080)")
    parser.add_argument("--mlflow-uri", type=str, default="http://localhost:5000", help="URI của MLflow tracking server")
    args = parser.parse_args()

    # Thiết lập thiết bị chạy (CPU hoặc GPU nếu có)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Server chạy trên thiết bị: {device}")

    # Tải tập test trung tâm để đánh giá mô hình toàn cục
    # Chỉ lấy phần testloader, không lấy trainloader
    _, testloader = load_datasets(num_clients=2)

    # Cấu hình tracking URI cho MLflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("Federated Learning - Flower & CIFAR-100")
    
    # Bắt đầu một Run trong MLflow
    mlflow.start_run()
    mlflow.log_param("num_rounds", args.rounds)
    mlflow.log_param("device", str(device))
    mlflow.log_param("dataset", "CIFAR-100")
    mlflow.log_param("model_architecture", "SimpleCNN")

    # Khởi tạo chiến lược FedAvg với hàm evaluate_fn tích hợp MLflow
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,           # Chọn 100% clients tham gia huấn luyện ở mỗi round
        fraction_evaluate=0.0,      # Tắt đánh giá phân tán phía client, chỉ dùng đánh giá tập trung phía server
        min_fit_clients=2,          # Yêu cầu tối thiểu 2 clients để bắt đầu huấn luyện
        min_evaluate_clients=0,
        min_available_clients=2,    # Đợi tối thiểu 2 clients kết nối mới khởi động
        evaluate_fn=get_evaluate_fn(testloader, device)
    )

    print(f"Flower Server đang khởi động tại địa chỉ {args.server_address}...")
    
    # Khởi chạy Flower Server
    try:
        fl.server.start_server(
            server_address=args.server_address,
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
        )
    except Exception as e:
        print(f"Lỗi khi chạy Flower Server: {e}")
    finally:
        # Đóng mlflow run sau khi hoàn tất hoặc gặp lỗi
        mlflow.end_run()
        print("Đã kết thúc MLflow run.")

if __name__ == "__main__":
    main()
