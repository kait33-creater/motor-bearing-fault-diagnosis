"""
1D-CNN 跨负载泛化评估
=====================
对照随机森林的 cwru_eval_crossload.py,给深度学习也做同样的严格评估:
用部分负载训练,在模型从没见过的负载上测试,检验 CNN 学到的特征是否也与工况无关。

与随机划分的区别:随机划分可能有同段泄露导致虚高;跨负载是真正的泛化考验。
"""

from pathlib import Path
import re

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from sklearn.metrics import accuracy_score

from cwru_features import get_de_signal, SEGMENT_LENGTH
from cwru_cnn import CNN1D, standardize, to_loader, train_epoch, predict, DEVICE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CWRU_RAW_DIR = PROJECT_ROOT / "data" / "cwru"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

LABEL_ORDER = ["normal", "inner_race", "ball", "outer_race"]
LABEL_TO_IDX = {n: i for i, n in enumerate(LABEL_ORDER)}
FNAME_PATTERN = re.compile(r"^\d+_(normal|inner_race|ball|outer_race)_(\d)hp\.mat$")


def build_with_load():
    """切片段,同时记录每个样本的负载。返回 X, y, loads。"""
    X, y, loads = [], [], []
    for mat_path in sorted(CWRU_RAW_DIR.glob("*.mat")):
        m = FNAME_PATTERN.match(mat_path.name)
        if not m:
            continue
        label, load = m.group(1), int(m.group(2))
        sig = get_de_signal(loadmat(mat_path))
        for i in range(len(sig) // SEGMENT_LENGTH):
            seg = sig[i * SEGMENT_LENGTH:(i + 1) * SEGMENT_LENGTH]
            X.append(seg); y.append(LABEL_TO_IDX[label]); loads.append(load)
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int64),
            np.array(loads, dtype=np.int64))


def train_and_eval(X_tr, y_tr, X_te, y_te, epochs=30):
    """训练一个 CNN 并返回测试准确率。"""
    torch.manual_seed(42)
    X_tr, X_te = standardize(X_tr, X_te)
    train_loader = to_loader(X_tr, y_tr, shuffle=True)
    test_loader = to_loader(X_te, y_te, shuffle=False)

    model = CNN1D(n_classes=len(LABEL_ORDER)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion)
    return accuracy_score(y_te, predict(model, test_loader))


def main():
    print(f"设备:{DEVICE}\n")
    X, y, loads = build_with_load()
    unique_loads = sorted(np.unique(loads))

    print("1D-CNN 留一负载交叉验证(测试负载训练时从未见过):")
    results = {}
    for test_load in unique_loads:
        tr = loads != test_load
        te = loads == test_load
        acc = train_and_eval(X[tr], y[tr], X[te], y[te])
        results[test_load] = acc
        print(f"  训练负载={[l for l in unique_loads if l != test_load]}  "
              f"测试负载={test_load}HP  ->  准确率 {acc:.4f}")

    mean_acc = float(np.mean(list(results.values())))
    print(f"\n  跨负载平均准确率:{mean_acc:.4f}")

    with open(REPORT_DIR / "cwru_cnn_crossload.md", "w", encoding="utf-8") as f:
        f.write("# 1D-CNN 跨负载泛化评估\n\n")
        f.write("用部分负载训练,在模型从未见过的负载上测试,检验深度学习特征是否与工况无关。\n\n")
        f.write("| 测试负载 | 准确率 |\n|---|---|\n")
        for load, acc in results.items():
            f.write(f"| {load} HP | {acc:.4f} |\n")
        f.write(f"\n跨负载平均准确率:**{mean_acc:.4f}**\n")
    print("已保存报告:cwru_cnn_crossload.md")


if __name__ == "__main__":
    main()
