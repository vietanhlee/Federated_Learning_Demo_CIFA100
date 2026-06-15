import argparse
from collections import OrderedDict
import numpy as np
import torch
import flwr as fl

from src.model import SimpleCNN, train, test
from src.dataset import load_datasets

# Hàm chuyển đổi trọng số từ PyTorch State Dict sang danh sách các NumPy arrays
def get_parameters(net: torch.nn.Module) -> list[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

# Hàm nạp trọng số từ danh sách NumPy arrays vào PyTorch State Dict
def set_parameters(net: torch.nn.Module, parameters: list[np.ndarray]) -> None:
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# Triển khai Flower NumPyClient
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net: torch.nn.Module, trainloader: torch.utils.data.DataLoader, 
                 testloader: torch.utils.data.DataLoader, device: torch.device, epochs: int, lr: float):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.epochs = epochs
        self.lr = lr

    def get_parameters(self, config: dict) -> list[np.ndarray]:
        return get_parameters(self.net)

    def fit(self, parameters: list[np.ndarray], config: dict) -> tuple[list[np.ndarray], int, dict]:
        # Cập nhật trọng số toàn cục nhận từ server
        set_parameters(self.net, parameters)
        
        # Huấn luyện cục bộ (local training) trên tập dữ liệu của client này
        train(self.net, self.trainloader, epochs=self.epochs, lr=self.lr, device=self.device)
        
        # Trả về trọng số đã được cập nhật, kích thước tập train và thông tin phụ trợ (nếu có)
        return get_parameters(self.net), len(self.trainloader.dataset), {}

    def evaluate(self, parameters: list[np.ndarray], config: dict) -> tuple[float, int, dict]:
        # Cập nhật trọng số toàn cục nhận từ server
        set_parameters(self.net, parameters)
        
        # Đánh giá cục bộ trên tập kiểm thử
        loss, accuracy = test(self.net, self.testloader, self.device)
        
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}


def main():
    parser = argparse.ArgumentParser(description="Flower Federated Learning Client")
    parser.add_argument("--client-id", type=int, required=True, choices=[0, 1], help="ID của client (0 hoặc 1)")
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8080", help="Địa chỉ của Flower Server (mặc định: 127.0.0.1:8080)")
    parser.add_argument("--epochs", type=int, default=2, help="Số epoch huấn luyện cục bộ ở mỗi round (mặc định: 2)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate cho optimizer cục bộ (mặc định: 0.001)")
    args = parser.parse_args()

    # Thiết lập thiết bị chạy (CPU hoặc GPU nếu có)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Client {args.client-id} chạy trên thiết bị: {device}")

    # Khởi tạo mô hình CNN
    net = SimpleCNN(num_classes=100)

    # Tải và lấy phần dataset tương ứng cho client này
    # Ở đây num_clients=2 cố định theo yêu cầu chạy 2 tiến trình
    trainloaders, testloader = load_datasets(num_clients=2)
    client_trainloader = trainloaders[args.client_id]

    # Khởi tạo Flower Client
    client = FlowerClient(net, client_trainloader, testloader, device, args.epochs, args.lr)
    
    # Kết nối đến Flower Server và bắt đầu quá trình Federated Learning
    print(f"Client {args.client_id} đang kết nối tới Server tại {args.server_address}...")
    fl.client.start_numpy_client(server_address=args.server_address, client=client)
    print(f"Client {args.client_id} đã dừng.")

if __name__ == "__main__":
    main()
