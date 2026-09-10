import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn



class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(
            32 * 8 * 8,
            10
        )

    def forward(self, x):
        print("Input        :", x.shape)

        x = self.conv1(x)
        print("After Conv1  :", x.shape)
        print(self.conv1)

        x = self.relu(x)
        print("After ReLU   :", x.shape)

        x = self.pool(x)
        print("After Pool1  :", x.shape)

        x = self.conv2(x)
        print("After Conv2  :", x.shape)

        x = self.relu(x)
        print("After ReLU   :", x.shape)

        x = self.pool(x)
        print("After Pool2  :", x.shape)

        x = self.flatten(x)
        print("After Flatten:", x.shape)

        x = self.fc(x)
        print("Output       :", x.shape)

        return x


# Tạo model
model = SimpleCNN()

# Một ảnh RGB kích thước 32x32
image = torch.randn(1, 3, 32, 32)

# Forward
output = model(image)