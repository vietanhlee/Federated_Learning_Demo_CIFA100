import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class SimpleCNN(nn.Module):
    """Mô hình CNN cơ bản nhưng đủ mạnh để phân loại CIFAR-100 (100 lớp)."""
    def __init__(self, num_classes: int = 100):
        super(SimpleCNN, self).__init__()
        # Layer 1: Input (3, 32, 32) -> (32, 32, 32) -> Pool -> (32, 16, 16)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Layer 2: Input (32, 16, 16) -> (64, 16, 16) -> Pool -> (64, 8, 8)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Layer 3: Input (64, 8, 8) -> (128, 8, 8) -> Pool -> (128, 4, 4)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train(net: nn.Module, trainloader: DataLoader, epochs: int, lr: float, device: torch.device) -> None:
    """Huấn luyện mô hình cục bộ trên client."""
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")


def test(net: nn.Module, testloader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Đánh giá mô hình trên tập kiểm thử (trả về loss và accuracy)."""
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy
