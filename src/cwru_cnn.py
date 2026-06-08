"""
1D-CNN 轴承故障诊断(深度学习,与随机森林对比)
=================================================
与传统方法的区别:
  随机森林:人工设计 17 个时频域特征 -> 分类
  1D-CNN  :把原始振动信号片段直接喂给网络,卷积层自动学习特征 -> 分类(端到端)

本脚本:
  1. 把原始信号切成片段(不提特征,保留原始波形)
  2. 标准化 + 划分训练/测试
  3. 定义并训练一个小型 1D-CNN
  4. 在测试集评估,与 cwru_train.py 的随机森林结果对比
"""

from pathlib import Path
import re

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from cwru_features import get_de_signal, SEGMENT_LENGTH

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CWRU_RAW_DIR = PROJECT_ROOT / "data" / "cwru"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
for d in (MODEL_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

LABEL_ORDER = ["normal", "inner_race", "ball", "outer_race"]
LABEL_TO_IDX = {name: i for i, name in enumerate(LABEL_ORDER)}
FNAME_PATTERN = re.compile(r"^\d+_(normal|inner_race|ball|outer_race)_(\d)hp\.mat$")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_segments():
    """
    把每个文件的原始信号切成片段,返回 (X, y)。
    X 形状 (样本数, 片段长度),保留原始波形(不提特征)。
    """
    X, y = [], []
    for mat_path in sorted(CWRU_RAW_DIR.glob("*.mat")):
        m = FNAME_PATTERN.match(mat_path.name)
        if not m:
            continue
        label = m.group(1)
        sig = get_de_signal(loadmat(mat_path))
        n_seg = len(sig) // SEGMENT_LENGTH
        for i in range(n_seg):
            seg = sig[i * SEGMENT_LENGTH:(i + 1) * SEGMENT_LENGTH]
            X.append(seg)
            y.append(LABEL_TO_IDX[label])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


class CNN1D(nn.Module):
    """
    小型一维卷积网络。
    三组 [卷积 -> 批归一化 -> ReLU -> 池化] 逐步提取并压缩特征,
    最后全连接层输出 4 个类别的得分。
    """

    def __init__(self, n_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            # 输入 (batch, 1通道, 2048点)
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),    # 不管前面长度多少,压成每通道1个值
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.features(x)            # (batch, 64, 1)
        x = x.flatten(1)               # (batch, 64)
        return self.classifier(x)


def standardize(X_train, X_test):
    """按训练集的均值/标准差标准化(测试集用训练集的统计量,避免信息泄露)。"""
    mean = X_train.mean()
    std = X_train.std() + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def to_loader(X, y, batch_size=32, shuffle=False):
    """numpy -> DataLoader。增加通道维:(N, 2048) -> (N, 1, 2048)。"""
    X_t = torch.tensor(X).unsqueeze(1)
    y_t = torch.tensor(y)
    ds = torch.utils.data.TensorDataset(X_t, y_t)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds = []
    for xb, _ in loader:
        out = model(xb.to(DEVICE))
        preds.append(out.argmax(1).cpu().numpy())
    return np.concatenate(preds)


def main(epochs=30):
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"设备:{DEVICE}\n")

    X, y = build_segments()
    print(f"样本数:{len(y)},每段长度:{X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    X_train, X_test = standardize(X_train, X_test)

    train_loader = to_loader(X_train, y_train, shuffle=True)
    test_loader = to_loader(X_test, y_test, shuffle=False)

    model = CNN1D(n_classes=len(LABEL_ORDER)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量:{n_params:,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        if ep % 5 == 0 or ep == 1:
            acc = accuracy_score(y_test, predict(model, test_loader))
            print(f"  epoch {ep:>3d}  loss={loss:.4f}  测试准确率={acc:.4f}")

    y_pred = predict(model, test_loader)
    final_acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, labels=range(len(LABEL_ORDER)),
        target_names=LABEL_ORDER, zero_division=0)
    print(f"\n1D-CNN 最终测试准确率:{final_acc:.4f}\n")
    print(report)

    torch.save(model.state_dict(), MODEL_DIR / "cwru_cnn1d.pt")

    with open(REPORT_DIR / "cwru_cnn_summary.md", "w", encoding="utf-8") as f:
        f.write("# 1D-CNN 深度学习诊断报告\n\n")
        f.write("## 方法\n\n把原始振动信号片段(2048点)直接输入一维卷积网络,"
                "由网络自动学习判别特征,无需人工特征工程(端到端)。\n\n")
        f.write(f"- 设备:{DEVICE}\n- 模型参数量:{n_params:,}\n")
        f.write(f"- 测试准确率:**{final_acc:.4f}**\n\n```text\n{report}\n```\n")
        f.write("\n## 与随机森林对比\n\n两种范式在 CWRU 四分类上均可达到很高准确率。"
                "随机森林依赖人工设计的时频域特征、可解释性强、训练快;"
                "1D-CNN 端到端自动学特征、省去特征工程,但需要更多数据与算力、可解释性弱。\n")
    print(f"已保存模型与报告。")


if __name__ == "__main__":
    main()


