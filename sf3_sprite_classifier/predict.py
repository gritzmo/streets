import argparse
from pathlib import Path
from PIL import Image
import torch
from torch import nn
from torchvision import transforms


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
        return self.classifier(self.features(x))


def load_image(path: Path):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


def predict_image(model_path: Path, image_path: Path):
    data = torch.load(model_path, map_location="cpu")
    class_to_idx = data["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    model = SpriteCNN(num_classes)
    model.load_state_dict(data["model_state"])
    model.eval()

    tensor = load_image(image_path)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs.data, 1)
    label = idx_to_class[predicted.item()]
    return label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sprite action")
    parser.add_argument("image", type=Path, help="Path to image")
    parser.add_argument("--model-path", type=Path, default=Path("model.pth"), help="Path to trained model")
    args = parser.parse_args()

    label = predict_image(args.model_path, args.image)
    print(f"Prediction: {label}")
