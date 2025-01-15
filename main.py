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

# 追加インストールが必要
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

########################################################################
# 1. デバイス選択 (MPS or CUDA or CPU)
########################################################################
def get_device():
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
    correct = 0
    for i in range(len(preds)):
        if preds[i] == targets[i]:
            correct += 1
    return correct / len(preds)

########################################################################
# 3. ネットワーク定義 (簡単な CNN の例)
########################################################################
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
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
            nn.Dropout(dropout_rate),  # dropout_rate を可変に
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
def train_one_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == target).sum().item()
        total += target.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, device, valid_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in valid_loader:
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

def test(model, device, test_loader):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            trues.extend(target.cpu().numpy())

    print("=== Classification Report ===")
    print(classification_report(trues, preds, digits=4))

    cm = confusion_matrix(trues, preds)
    print("=== Confusion Matrix ===")
    print(cm)

    acc_custom = custom_metric_fast(np.array(preds), np.array(trues))
    print(f"Custom Metric (Numba) - Accuracy: {acc_custom:.4f}")

########################################################################
# --- ベイズ最適化 ここから
#     「learning_rate」「dropout_rate」などのハイパーパラメータを最適化する例
########################################################################
# 1. 探索空間の設定
space  = [
    Real(1e-4, 1e-1,  name="learning_rate", prior="log-uniform"),  # 学習率
    Real(0.0, 0.8,    name="dropout_rate"),                        # ドロップアウト率
]

# 2. ベイズ最適化の目的関数
@use_named_args(space)
def objective(**params):
    """
    引数: params (learning_rate, dropout_rate, etc.)
    戻り値: ベイズ最適化で最小化したい値 (ここでは validation loss)
    """
    # 実験ごとに毎回データセットやモデルを再定義
    device = get_device()

    # データ変換
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

    # CIFAR-10
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform_test)

    # train/valid 分割
    n_train = int(len(train_dataset)*0.8)
    n_valid = len(train_dataset) - n_train
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset, [n_train, n_valid]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=128, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # モデル, 損失関数, オプティマイザの準備
    model = SimpleCNN(num_classes=10, dropout_rate=params["dropout_rate"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    # 学習 (ここでは簡易的にエポック数を少なく設定)
    num_epochs = 5
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
        valid_loss, valid_acc = validate(model, device, valid_loader, criterion)
        #print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}")

    # ベイズ最適化の目的: validation loss の最小化
    return valid_loss

########################################################################
# メイン関数: 通常の固定ハイパーパラメータ実験 + ベイズ最適化
########################################################################
def main():
    # 通常学習 (固定ハイパーパラメータ) 例 ------------------------
    # ※ ベイズ最適化のみやりたい場合は、この通常学習部分は不要です。

    print("======== 通常学習 (例) ========")
    device = get_device()
    batch_size = 128
    num_epochs = 2
    learning_rate = 1e-3
    dropout_rate = 0.5

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

    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform_test)

    n_train = int(len(train_dataset)*0.8)
    n_valid = len(train_dataset) - n_train
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [n_train, n_valid])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                               shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    model = SimpleCNN(num_classes=10, dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs+1):
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
        valid_loss, valid_acc = validate(model, device, valid_loader, criterion)
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}")

    test(model, device, test_loader)

    # モデル保存 (任意)
    save_path = os.path.join('./checkpoints', 'model_cifar10_fixed.pth')
    os.makedirs('./checkpoints', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Fixed-model saved to {save_path}.")

    # -------------------------------------------------------------
    # ベイズ最適化の実行
    # -------------------------------------------------------------
    print("\n======== ベイズ最適化の実行 ========")
    res = gp_minimize(
        func=objective,     # 目的関数 (validation loss を最小化)
        dimensions=space,   # 探索空間
        n_calls=10,         # 試行回数 (実験回数)
        random_state=42,    # 乱数シード
        n_initial_points=3  # 最初にランダムで試す回数
    )

    # 最適化結果の表示
    print("Best score (validation loss):", res.fun)
    print("Best hyperparameters:")
    best_params = dict(zip(["learning_rate", "dropout_rate"], res.x))
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # -------------------------------------------------------------
    # 最適ハイパーパラメータで学習し直してテスト評価
    # -------------------------------------------------------------
    print("\n======== 最適ハイパーパラメータで再学習 ========")
    best_lr = best_params["learning_rate"]
    best_dropout = best_params["dropout_rate"]
    device = get_device()

    # 同様に再構築
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform_test)
    n_train = int(len(train_dataset)*0.8)
    n_valid = len(train_dataset) - n_train
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [n_train, n_valid])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                               shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    # 新しいモデル・オプティマイザを作成
    model_best = SimpleCNN(num_classes=10, dropout_rate=best_dropout).to(device)
    criterion_best = nn.CrossEntropyLoss()
    optimizer_best = optim.Adam(model_best.parameters(), lr=best_lr)

    # もう少し長めに学習 (例: 10 epoch)
    num_epochs_best = 10
    for epoch in range(1, num_epochs_best+1):
        train_loss, train_acc = train_one_epoch(model_best, device, train_loader, optimizer_best, criterion_best)
        valid_loss, valid_acc = validate(model_best, device, valid_loader, criterion_best)
        print(f"[Best Param Model] Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}")

    # テスト評価
    print("\n=== Test evaluation with best params ===")
    test(model_best, device, test_loader)

    # 保存
    save_path_best = os.path.join('./checkpoints', 'model_cifar10_best.pth')
    os.makedirs('./checkpoints', exist_ok=True)
    torch.save(model_best.state_dict(), save_path_best)
    print(f"Best-model saved to {save_path_best}.")

if __name__ == '__main__':
    main()
