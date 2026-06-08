"""
CWRU 跨负载泛化评估(cross-load evaluation)
=============================================
这是项目最严格、最有说服力的评估。

问题:之前用随机划分,测试集准确率接近100%,但偏乐观——
       因为训练和测试数据来自同一负载、同一段连续信号。
真正的考验:电机负载变化时振动会变,模型在"没见过的负载"上还认得出故障吗?
       这叫域偏移(domain shift),是故障诊断研究的核心难点。

做法:用 0HP / 1HP / 2HP 负载的数据训练,在 3HP(模型从没见过的负载)上测试。
      跨负载准确率才是诚实的泛化能力,可写进简历。

依赖:复用 cwru_features.py 里的特征提取函数,保证特征定义完全一致。
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 复用已写好的特征提取逻辑(不重复造轮子,保证一致)
from cwru_features import extract_features_from_signal, get_de_signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CWRU_RAW_DIR = PROJECT_ROOT / "data" / "cwru"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_ORDER = ["normal", "inner_race", "ball", "outer_race"]
LABEL_CN = {"normal": "正常", "inner_race": "内圈", "ball": "滚珠", "outer_race": "外圈"}

# 只用带负载标记的文件(形如 106_inner_race_1hp.mat)
FNAME_PATTERN = re.compile(r"^\d+_(normal|inner_race|ball|outer_race)_(\d)hp\.mat$")


def build_dataset():
    """
    读取所有带负载标记的文件,提取特征,返回一个带 label 和 load 列的 DataFrame。
    load 列标明该样本来自哪个负载,用于按负载划分训练/测试。
    """
    all_rows = []
    for mat_path in sorted(CWRU_RAW_DIR.glob("*.mat")):
        m = FNAME_PATTERN.match(mat_path.name)
        if not m:
            continue  # 跳过不带负载标记的旧文件
        label, load_hp = m.group(1), int(m.group(2))

        signal = get_de_signal(loadmat(mat_path))
        rows = extract_features_from_signal(signal, label)
        for r in rows:
            r["load"] = load_hp
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    return df


def split_xy(df):
    """从 DataFrame 拆出特征矩阵和标签(强制标准 numpy,兼容 pandas 3.0)。"""
    feature_cols = [c for c in df.columns if c not in ("label", "load")]
    X = np.asarray(df[feature_cols], dtype=float)
    y = np.asarray(df["label"], dtype=object)
    return X, y, feature_cols


def make_model():
    return RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=2,
        class_weight="balanced", random_state=42,
    )


def eval_cross_load(df):
    """
    留一负载交叉验证:轮流把某个负载当测试集,其余负载训练。
    返回每个测试负载的准确率,以及平均值。
    """
    feature_cols = [c for c in df.columns if c not in ("label", "load")]
    loads = sorted(df["load"].unique())
    results = {}

    for test_load in loads:
        train_df = df[df["load"] != test_load]
        test_df = df[df["load"] == test_load]

        X_tr = np.asarray(train_df[feature_cols], dtype=float)
        y_tr = np.asarray(train_df["label"], dtype=object)
        X_te = np.asarray(test_df[feature_cols], dtype=float)
        y_te = np.asarray(test_df["label"], dtype=object)

        model = make_model()
        model.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, model.predict(X_te))
        results[test_load] = acc
        print(f"  训练负载={[l for l in loads if l != test_load]}  "
              f"测试负载={test_load}HP  ->  准确率 {acc:.4f}")

    mean_acc = np.mean(list(results.values()))
    print(f"  跨负载平均准确率:{mean_acc:.4f}")
    return results, mean_acc


def eval_random_split(df):
    """对照组:随机划分(忽略负载)。预期偏高,用来和跨负载对比。"""
    from sklearn.model_selection import train_test_split
    X, y, _ = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    model = make_model()
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    print(f"  随机划分准确率:{acc:.4f}")
    return acc


def main():
    print("========== CWRU 跨负载泛化评估 ==========\n")
    df = build_dataset()
    print(f"总样本数:{len(df)}")
    print(f"各负载样本数:{df['load'].value_counts().sort_index().to_dict()}\n")

    print("【对照】随机划分(忽略负载,预期偏乐观):")
    random_acc = eval_random_split(df)

    print("\n【严格】留一负载交叉验证(测试负载训练时从未见过):")
    cross_results, cross_mean = eval_cross_load(df)

    # 写报告
    report_path = REPORT_DIR / "cwru_crossload_eval.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# CWRU 跨负载泛化评估报告\n\n")
        f.write("## 评估设计\n\n")
        f.write("- **随机划分(对照)**:打乱所有样本随机分训练/测试,不区分负载。"
                "因训练测试可能来自同一负载、相邻信号段,结果偏乐观。\n")
        f.write("- **留一负载交叉验证(严格)**:轮流用一个负载(0/1/2/3 HP)做测试集,"
                "其余负载训练。测试负载在训练时完全没出现,考验真实泛化能力。\n\n")
        f.write("## 结果对比\n\n")
        f.write(f"- 随机划分准确率:**{random_acc:.4f}**\n")
        f.write(f"- 跨负载平均准确率:**{cross_mean:.4f}**\n\n")
        f.write("| 测试负载 | 准确率 |\n|---|---|\n")
        for load, acc in cross_results.items():
            f.write(f"| {load} HP | {acc:.4f} |\n")
        f.write("\n## 结论\n\n")
        gap = random_acc - cross_mean
        f.write(f"跨负载准确率比随机划分低约 {gap:.3f},这个差距正是"
                "负载变化引起的域偏移代价。跨负载结果才反映模型在新工况下的真实诊断能力。\n")
    print(f"\n报告已保存:{report_path}")


if __name__ == "__main__":
    main()


