import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, channels: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, channels: int = 1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class DCGAN:
    def __init__(self, latent_dim: int = 100, device: str = "cpu"):
        self.device = device
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim=self.latent_dim, channels=1).to(device)
        self.discriminator = Discriminator(channels=1).to(device)
        self.generator_criterion = nn.BCELoss()
        self.discriminator_criterion = nn.BCELoss()
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.training_history = []
        self.epochs_completed = 0
        self.latest_checkpoint = None

    def set_seed(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def get_data_loader(self, batch_size: int = 64, root_dir: str = "./data"):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        dataset = datasets.MNIST(root=root_dir, train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    @staticmethod
    def default_output_dirs(base_dir: str = ".") -> dict:
        root = Path(base_dir)
        return {
            "root": root,
            "models": root / "models",
            "samples": root / "generated_samples",
            "history": root / "training_history",
            "checkpoints": root / "checkpoints",
        }

    def save_checkpoint(self, checkpoint_dir: str, epoch: int, metadata: dict | None = None):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"gan_epoch_{epoch}.pt"
        payload = {
            "epoch": epoch,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "generator_optimizer": self.generator_optimizer.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            "latent_dim": self.latent_dim,
            "training_history": self.training_history,
            "metadata": metadata or {},
        }
        torch.save(payload, checkpoint_path)
        self.latest_checkpoint = str(checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
        self.discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
        self.latent_dim = checkpoint.get("latent_dim", self.latent_dim)
        self.training_history = checkpoint.get("training_history", [])
        self.epochs_completed = max(0, int(checkpoint.get("epoch", -1)) + 1)
        self.latest_checkpoint = checkpoint_path
        return checkpoint

    def train_one_epoch(self, data_loader: DataLoader, epoch_index: int):
        self.generator.train()
        self.discriminator.train()

        generator_losses = []
        discriminator_losses = []
        start_time = time.time()

        for real_images, _ in data_loader:
            real_images = real_images.to(self.device)
            batch_size = real_images.size(0)

            real_labels = torch.ones(batch_size, 1, device=self.device)
            fake_labels = torch.zeros(batch_size, 1, device=self.device)

            self.discriminator.zero_grad()
            real_output = self.discriminator(real_images).view(-1, 1)
            real_loss = self.discriminator_criterion(real_output, real_labels)

            noise = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_images = self.generator(noise)
            fake_output = self.discriminator(fake_images.detach()).view(-1, 1)
            fake_loss = self.discriminator_criterion(fake_output, fake_labels)

            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            self.generator.zero_grad()
            generated_output = self.discriminator(fake_images).view(-1, 1)
            generator_loss = self.generator_criterion(generated_output, real_labels)
            generator_loss.backward()
            self.generator_optimizer.step()

            generator_losses.append(generator_loss.item())
            discriminator_losses.append(discriminator_loss.item())

        epoch_generator_loss = float(np.mean(generator_losses)) if generator_losses else 0.0
        epoch_discriminator_loss = float(np.mean(discriminator_losses)) if discriminator_losses else 0.0
        elapsed = time.time() - start_time
        record = {
            "epoch": epoch_index,
            "generator_loss": epoch_generator_loss,
            "discriminator_loss": epoch_discriminator_loss,
            "elapsed_seconds": round(elapsed, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.training_history.append(record)
        self.epochs_completed = epoch_index + 1
        return record

    def save_training_history(self, history_path: str):
        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.training_history)
        df.to_csv(history_path, index=False)
        return history_path

    def save_generated_grid(self, output_dir: str, count: int = 16, filename: str = "sample_grid.png", seed: int | None = None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if seed is not None:
            self.set_seed(seed)

        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(count, self.latent_dim, device=self.device)
            generated = self.generator(noise).cpu()
        generated = (generated + 1.0) / 2.0

        grid_size = int(np.ceil(np.sqrt(count)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
        axes = np.array(axes).reshape(-1)

        for idx, ax in enumerate(axes):
            ax.axis("off")
            if idx < count:
                image = generated[idx].squeeze(0).numpy()
                ax.imshow(image, cmap="gray", vmin=0, vmax=1)

        for ax in axes[count:]:
            ax.set_visible(False)

        fig.tight_layout()
        fig_path = output_dir / filename
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        return fig_path

    def generate_images(self, count: int = 1, seed: int | None = None, base_dir: str = "."):
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        if seed is not None:
            self.set_seed(seed)

        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(count, self.latent_dim, device=self.device)
            generated = self.generator(noise).cpu()

        images = []
        for index, tensor in enumerate(generated):
            array = tensor.squeeze(0).numpy()
            array = ((array + 1.0) / 2.0 * 255).astype(np.uint8)
            image = Image.fromarray(array, mode="L")
            path = base_dir / f"generated_{index:03d}.png"
            image.save(path)
            images.append(path)
        return images

    @staticmethod
    def load_history(history_path: str):
        history_path = Path(history_path)
        if not history_path.exists():
            return []
        df = pd.read_csv(history_path)
        return df.to_dict("records")

    @staticmethod
    def latest_checkpoint_path(checkpoint_dir: str):
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return None
        checkpoints = sorted(
            checkpoint_dir.glob("gan_epoch_*.pt"),
            key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
        )
        return str(checkpoints[-1]) if checkpoints else None

    @staticmethod
    def dataset_summary(root_dir: str = "./data"):
        dataset = datasets.MNIST(root=root_dir, train=True, download=True, transform=transforms.ToTensor())
        samples = []
        for idx in range(10):
            image, label = dataset[idx]
            samples.append({"image": image[0].numpy(), "label": int(label)})
        return {
            "dataset_name": "MNIST",
            "samples": samples,
            "num_classes": 10,
            "image_shape": (28, 28),
            "num_examples": len(dataset),
            "class_counts": {str(i): int((np.array(dataset.targets) == i).sum()) for i in range(10)},
        }


def ensure_directories(base_dir: str = "."):
    dirs = DCGAN.default_output_dirs(base_dir)
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def model_setup(base_dir: str = ".", latent_dim: int = 100, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DCGAN(latent_dim=latent_dim, device=device)
    ensure_directories(base_dir)
    model.set_seed(42)
    return model


def save_run_metadata(base_dir: str, config: dict):
    root = Path(base_dir)
    path = root / "training_history" / "run_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    return path


if __name__ == "__main__":
    model = model_setup(base_dir=".")
    loader = model.get_data_loader(batch_size=64, root_dir="./data")
    model.train_one_epoch(loader, 0)
    print("Model training step completed.")
