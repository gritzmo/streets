import argparse
from pathlib import Path
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms


class SpriteDataset(Dataset):
    """Dataset that treats each character/action combination as a class."""

    def __init__(self, root: Path, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        root = Path(root)
        for char_dir in sorted(root.iterdir()):
            if not char_dir.is_dir():
                continue
            char = char_dir.name
            for action_dir in sorted(char_dir.iterdir()):
                if not action_dir.is_dir():
                    continue
                label = f"{char}_{action_dir.name}"
                idx = self.class_to_idx.setdefault(label, len(self.class_to_idx))
                for img_path in sorted(action_dir.iterdir()):
                    if img_path.is_file() and img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                        self.samples.append((img_path, idx))
        self.num_classes = len(self.class_to_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class SpriteCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_data(data_dir: Path, batch_size: int):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    dataset = SpriteDataset(data_dir, transform=transform)
    num_classes = dataset.num_classes
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader, num_classes, dataset.class_to_idx


def train(model, train_loader, val_loader, device, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total if total > 0 else 0
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} - Val Acc: {accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sprite classifier")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to directory containing character folders",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--model-path", type=Path, default=Path("model.pth"), help="Output model file")
    args = parser.parse_args()

    train_loader, val_loader, num_classes, class_to_idx = load_data(args.data_dir, args.batch_size)
    model = SpriteCNN(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(model, train_loader, val_loader, device, args.epochs, args.lr)

    torch.save({
        "model_state": model.state_dict(),
        "class_to_idx": class_to_idx,
    }, args.model_path)
    print(f"Model saved to {args.model_path}")
