"""
CWRU 轴承故障诊断模型训练脚本
==============================
用提取好的特征表训练分类器,完成四分类:正常 / 内圈 / 滚珠 / 外圈故障。

与旧的 ml_model.py 的本质区别:
  旧版数据是规则模拟生成的,模型只是在重新学规则(循环论证,准确率虚高);
  本版数据是 CWRU 真实振动信号提取的特征,准确率反映模型真实诊断能力。

产物:
  - 混淆矩阵图、特征重要性图(outputs/figures/)
  - 模型文件(outputs/models/)
  - 文字报告(outputs/reports/)
"""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_CSV = PROJECT_ROOT / "data" / "processed" / "cwru_features.csv"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
for d in (FIGURE_DIR, MODEL_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

LABEL_ORDER = ["normal", "inner_race", "ball", "outer_race"]
LABEL_CN = {"normal": "正常", "inner_race": "内圈", "ball": "滚珠", "outer_race": "外圈"}


def set_plot_style():
    """统一绘图风格:无衬线字体、去顶/右边框、300dpi。"""
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def load_data():
    """读取特征表,拆成特征矩阵 X 和标签 y。"""
    df = pd.read_csv(FEATURE_CSV)
    feature_cols = [c for c in df.columns if c != "label"]
    # 强制转成标准 numpy 数组(pandas 3.0 默认 PyArrow 后端,sklearn 无法直接索引)
    X = np.asarray(df[feature_cols], dtype=float)
    y = np.asarray(df["label"], dtype=object)
    return X, y, feature_cols


def plot_confusion(cm, labels_cn, out_path):
    """画混淆矩阵热力图,格子里标注样本数。"""
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(labels_cn)))
    ax.set_yticks(range(len(labels_cn)))
    ax.set_xticklabels(labels_cn)
    ax.set_yticklabels(labels_cn)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("随机森林故障诊断混淆矩阵")

    # 在每个格子里写数字,深色背景用白字
    thresh = cm.max() / 2
    for i in range(len(labels_cn)):
        for j in range(len(labels_cn)):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"已保存:{out_path.name}")


def plot_importance(importances, feature_cols, out_path, top_n=10):
    """画特征重要性条形图(取最重要的 top_n 个)。"""
    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(range(len(order)), importances[order][::-1], color="#4C72B0")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_cols[i] for i in order][::-1])
    ax.set_xlabel("重要性")
    ax.set_title(f"随机森林特征重要性(Top {top_n})")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"已保存:{out_path.name}")


def main():
    set_plot_style()
    X, y, feature_cols = load_data()
    print(f"样本数:{len(y)},特征数:{len(feature_cols)}\n")

    # 分层划分:保证每个类别在训练/测试集里的比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",   # 应对 normal 样本偏多
        random_state=42,
    )

    # 5 折交叉验证:比单次划分更能反映模型真实水平(样本少时尤其重要)
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"5折交叉验证准确率:{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"测试集准确率:{test_acc:.4f}\n")

    report = classification_report(
        y_test, y_pred, labels=LABEL_ORDER,
        target_names=[LABEL_CN[l] for l in LABEL_ORDER], zero_division=0,
    )
    print(report)


    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)
    plot_confusion(cm, [LABEL_CN[l] for l in LABEL_ORDER],
                   FIGURE_DIR / "cwru_confusion_matrix.png")

    # 特征重要性
    plot_importance(model.feature_importances_, feature_cols,
                    FIGURE_DIR / "cwru_feature_importance.png")

    # 保存模型
    model_path = MODEL_DIR / "cwru_fault_rf_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols,
                     "labels": LABEL_ORDER}, f)
    print(f"已保存模型:{model_path.name}")

    # 文字报告
    report_path = REPORT_DIR / "cwru_model_summary.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# CWRU 轴承故障诊断模型报告\n\n")
        f.write("## 数据\n\n")
        f.write("CWRU 真实轴承振动数据集(12kHz 驱动端,0HP,0.007英寸故障),"
                "原始信号分段加窗后提取时域+频域特征,共 "
                f"{len(y)} 个样本、{len(feature_cols)} 个特征。\n\n")
        f.write("## 模型\n\n")
        f.write("随机森林(200 棵树),四分类:正常 / 内圈 / 滚珠 / 外圈故障。\n\n")
        f.write("## 结果\n\n")
        f.write(f"- 5 折交叉验证准确率:{cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
        f.write(f"- 测试集准确率:{test_acc:.4f}\n\n")
        f.write("```text\n")
        f.write(report)
        f.write("\n```\n")
    print(f"已保存报告:{report_path.name}")


if __name__ == "__main__":
    main()



