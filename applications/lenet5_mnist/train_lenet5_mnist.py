import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import os

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------
# Directories
# -------------------------
os.makedirs("data/train_images", exist_ok=True)
os.makedirs("data/test_images", exist_ok=True)
os.makedirs("data/val_images", exist_ok=True)

# -------------------------
# Transforms
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# -------------------------
# Load MNIST
# -------------------------
full_train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
full_test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)

# -------------------------
# Split test dataset into 6k test and 4k validation
# -------------------------
test_indices = list(range(10000))
test_split = 6000
test_dataset = Subset(full_test_dataset, test_indices[:test_split])
val_dataset = Subset(full_test_dataset, test_indices[test_split:])

# -------------------------
# Store images (optional)
# -------------------------
for idx, (img, label) in enumerate(full_train_dataset):
    torch.save((img, label), f"data/train_images/{idx}.pt")
for idx, (img, label) in enumerate(test_dataset):
    torch.save((img, label), f"data/test_images/{idx}.pt")
for idx, (img, label) in enumerate(val_dataset):
    torch.save((img, label), f"data/val_images/{idx}.pt")

# -------------------------
# Dataloaders
# -------------------------
train_loader = DataLoader(full_train_dataset, batch_size=256, shuffle=True)

# -------------------------
# 3-conv LeNet-5
# -------------------------
class LeNet5_3Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32*1*1, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# -------------------------
# Model, loss, optimizer
# -------------------------
model = LeNet5_3Conv().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# -------------------------
# Training loop
# -------------------------
epochs = 10
for epoch in range(epochs):
    model.train()
    loss_sum = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss_sum:.4f} Accuracy: {acc:.2f}%")
    scheduler.step()

# -------------------------
# Save weights
# -------------------------
torch.save(model.state_dict(), "lenet5_3conv_weights.pth")
print("Training complete. Weights saved.")

