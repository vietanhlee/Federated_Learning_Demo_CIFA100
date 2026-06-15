import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_datasets(num_clients: int, batch_size: int = 64) -> tuple[list[DataLoader], DataLoader]:
    """Tải và phân chia dataset CIFAR-100 cho các client (I.I.D partitioning)."""
    
    # Chuẩn hóa ảnh CIFAR-100 theo mean và std chuẩn
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    print("Đang tải dữ liệu CIFAR-100 từ torchvision...")
    # Tải dataset về thư mục './data'
    trainset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

    # Chia dữ liệu huấn luyện (50,000 ảnh) thành các phần bằng nhau cho các client
    total_samples = len(trainset)
    samples_per_client = total_samples // num_clients
    lengths = [samples_per_client] * num_clients
    # Bù số lượng dư nếu phép chia không hết
    lengths[-1] += total_samples - sum(lengths)

    # Sử dụng generator seed cố định để đảm bảo phân chia giống nhau giữa các lần chạy
    datasets_split = random_split(trainset, lengths, generator=torch.Generator().manual_seed(42))
    
    # Tạo DataLoaders tương ứng cho mỗi client
    trainloaders = []
    for i, ds in enumerate(datasets_split):
        # Thiết lập DataLoader với shuffle=True cho quá trình huấn luyện
        trainloaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0))
        print(f"Client {i} được gán: {len(ds)} ảnh train.")

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Tập kiểm thử trung tâm (testset) có: {len(testset)} ảnh.")
    
    return trainloaders, testloader
