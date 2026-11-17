import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from digit_recognition import DigitCNN, MNIST_MEAN, MNIST_STD


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    example_count = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        example_count += labels.size(0)
    return total_loss / example_count


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    example_count = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            example_count += labels.size(0)
    return total_loss / example_count, correct / example_count


def train_classifier(epochs: int, batch_size: int, data_dir: Path, checkpoint_path: Path):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    training_loss = None
    for epoch in range(epochs):
        training_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"training loss: {training_loss:.4f} | test loss: {test_loss:.4f} | "
            f"test accuracy: {test_accuracy * 100:.2f}%"
        )

    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.cpu().state_dict(),
        "epochs": epochs,
        "training_loss": training_loss,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "normalization": {"mean": MNIST_MEAN, "std": MNIST_STD},
    }, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Final training loss: {training_loss:.4f}")
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_accuracy * 100:.2f}%")
    return training_loss, test_loss, test_accuracy


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the separate MNIST digit CNN.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints") / "mnist_cnn.pt",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    train_classifier(args.epochs, args.batch_size, args.data_dir, args.checkpoint)


if __name__ == "__main__":
    main()
