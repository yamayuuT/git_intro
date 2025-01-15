#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from sklearn.metrics import classification_report, confusion_matrix
from numba import jit

########################################################################
# 1. デバイス選択 (MPS or CUDA or CPU)
########################################################################
def get_device():
    """
    Apple Silicon (MPS) または CUDA が利用可能ならそれを返し、
    どちらもなければ CPU を返す。
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA CUDA.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    return device

########################################################################
# 2. Numba で高速化したい関数の例 (任意のカスタム処理)
########################################################################
@jit(nopython=True)
def custom_metric_fast(preds, targets):
    """
    例: 予測と正解ラベルを受け取り、任意の計算をする。
    ここでは単純に正解数をカウントして正解率を返すだけの例。
    """
    correct = 0
    for i in range(len(preds)):
        if preds[i] == targets[i]:
            correct += 1
    return correct / len(preds)

########################################################################
# 3. ネットワーク定義 (簡単な CNN の例)
########################################################################
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 32 -> 16

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 16 -> 8

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)   # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

########################################################################
# 4. 学習と検証を行う関数
########################################################################
def train_one_epoch(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # tqdm で進捗表示
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for data, target in pbar:
        data, target = data.to(device), target.to(device)

        # 勾配初期化
        optimizer.zero_grad()

        # 順伝播
        outputs = model(data)
        loss = criterion(outputs, target)

        # 逆伝播
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == target).sum().item()
        total += target.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, device, valid_loader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(valid_loader, desc=f"Epoch {epoch} [Val]")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            
            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

########################################################################
# 5. テスト (最終評価) を行う関数
########################################################################
def test(model, device, test_loader):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Test"):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            trues.extend(target.cpu().numpy())

    # scikit-learn の classification_report などで評価
    print("=== Classification Report ===")
    print(classification_report(trues, preds, digits=4))

    # 混同行列
    cm = confusion_matrix(trues, preds)
    print("=== Confusion Matrix ===")
    print(cm)

    # numba を用いたカスタムメトリクス例
    acc_custom = custom_metric_fast(np.array(preds), np.array(trues))
    print(f"Custom Metric (Numba) - Accuracy: {acc_custom:.4f}")

########################################################################
# 6. メイン処理
########################################################################
def main():
    # ハイパーパラメータ設定
    batch_size = 128
    num_epochs = 10
    learning_rate = 1e-3
    num_classes = 10  # CIFAR-10

    # デバイス取得
    device = get_device()

    # データ変換(前処理 & データ拡張)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    # CIFAR-10 データセットダウンロード & データローダ作成
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform_test)

    # train / valid 分割 (例: 5:1)
    n_train = int(len(train_dataset)*0.8)
    n_valid = len(train_dataset) - n_train
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset, [n_train, n_valid]
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                               shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    # モデル構築
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 学習ループ
    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []

    for epoch in range(1, num_epochs+1):
        # 訓練
        train_loss, train_acc = train_one_epoch(
            model, device, train_loader, optimizer, criterion, epoch)
        # 検証
        valid_loss, valid_acc = validate(
            model, device, valid_loader, criterion, epoch)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)

        print(f"Epoch [{epoch}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
              f"Val Loss: {valid_loss:.4f}, Val Acc: {valid_acc:.4f}")

    # 学習曲線プロット
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), valid_losses, label='Valid Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Acc')
    plt.plot(range(1, num_epochs+1), valid_accuracies, label='Valid Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    plt.show()

    # テストデータで評価
    test(model, device, test_loader)

    # 学習済みモデルの保存
    save_path = os.path.join('./checkpoints', 'model_cifar10.pth')
    os.makedirs('./checkpoints', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}.")

if __name__ == '__main__':
    main()
