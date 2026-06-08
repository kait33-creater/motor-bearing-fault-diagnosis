"""
CWRU 轴承故障诊断 —— 推理模块
================================
把"一段原始振动信号"变成"故障类型诊断结果"的核心逻辑,与界面(Streamlit)解耦。

设计要点:
  - 复用 cwru_features.py 里已有的特征提取函数,不重写,避免训练/推理特征口径不一致;
  - 严格按训练时保存的 feature_cols 顺序排列特征列;
  - 概率严格按 model.classes_ 映射,避免类别错位(classes_ 是字母序,与 labels 顺序不同);
  - 多段信号逐段预测,再做多数投票 + 概率平均,比单段更稳。

本模块不依赖 streamlit,可直接用命令行测试。
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy.io import loadmat

# 复用特征提取逻辑与分段长度,保证与训练完全一致
from cwru_features import (
    time_domain_features,
    freq_domain_features,
    SEGMENT_LENGTH,
    SAMPLING_RATE,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "cwru_fault_rf_model.pkl"

# 英文标签 -> 中文公开名(用于界面展示)
LABEL_CN = {
    "normal": "正常",
    "inner_race": "内圈故障",
    "ball": "滚珠故障",
    "outer_race": "外圈故障",
}


def load_model(model_path=DEFAULT_MODEL_PATH):
    """加载训练好的模型包,返回 (model, feature_cols, labels)。"""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"未找到模型文件:{model_path}\n请先运行 python src/cwru_train.py 训练模型。"
        )
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["feature_cols"], bundle["labels"]


def load_signal_from_mat(file_or_bytes):
    """
    从 .mat 文件读取驱动端(DE)振动信号,压平成一维 float 数组。
    file_or_bytes 可以是路径,也可以是文件对象(Streamlit 上传的 BytesIO)。
    """
    mat = loadmat(file_or_bytes)
    de_keys = [k for k in mat.keys() if k.endswith("_DE_time")]
    if not de_keys:
        raise ValueError(
            "未在 .mat 文件中找到驱动端振动信号(*_DE_time 变量)。"
            "请确认上传的是 CWRU 驱动端数据文件。"
        )
    signal = np.asarray(mat[de_keys[0]], dtype=float).ravel()
    return signal


def load_signal_from_csv(file_or_bytes):
    """
    从 .csv 文件读取振动信号:取第一列数值列,压平成一维 float 数组。
    兼容带表头/不带表头两种情况。
    """
    # 先按有表头读;若第一列不是数值,则当作无表头重读
    df = pd.read_csv(file_or_bytes)
    first_col = df.iloc[:, 0]
    if not np.issubdtype(first_col.dtype, np.number):
        if hasattr(file_or_bytes, "seek"):
            file_or_bytes.seek(0)
        df = pd.read_csv(file_or_bytes, header=None)
        first_col = df.iloc[:, 0]
    signal = np.asarray(first_col, dtype=float).ravel()
    signal = signal[~np.isnan(signal)]  # 去掉可能的 NaN 尾行
    return signal


def signal_to_feature_matrix(signal, feature_cols):
    """
    把一维信号切成 2048 点的段,对每段提取 17 维特征,
    严格按 feature_cols 的顺序排列成矩阵 (n_segments, 17)。
    返回 (X, n_segments)。信号过短则抛出异常。
    """
    n_segments = len(signal) // SEGMENT_LENGTH
    if n_segments < 1:
        raise ValueError(
            f"信号长度不足:仅 {len(signal)} 点,至少需要 {SEGMENT_LENGTH} 点才能提取一段特征。"
        )

    rows = []
    for i in range(n_segments):
        seg = signal[i * SEGMENT_LENGTH:(i + 1) * SEGMENT_LENGTH]
        feats = {}
        feats.update(time_domain_features(seg))
        feats.update(freq_domain_features(seg))
        # 按训练时的列顺序取值,杜绝字典顺序导致的特征错位
        rows.append([feats[c] for c in feature_cols])

    X = np.asarray(rows, dtype=float)
    # 数值兜底:个别段可能因常数信号产生 inf/nan,用 0 替代,避免预测报错
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, n_segments


def diagnose_signal(signal, model, feature_cols, sampling_rate=SAMPLING_RATE):
    """
    对一段振动信号做诊断。
    流程:切段 -> 逐段提特征 -> 逐段预测 -> 多数投票定结论 + 概率按段平均算置信度。

    返回 dict:
      label_en / label_cn : 最终故障类型(英文键 / 中文名)
      confidence          : 最终类别的平均概率(0~1)
      proba               : {中文类别名: 平均概率} 的完整分布
      n_segments          : 参与投票的段数
      votes               : {中文类别名: 投票段数}
      duration_sec        : 信号时长(秒)
    """
    X, n_segments = signal_to_feature_matrix(signal, feature_cols)

    # 逐段预测类别(投票用)
    seg_preds = model.predict(X)
    # 逐段预测概率,列顺序 = model.classes_(字母序),与 feature labels 顺序不同
    seg_proba = model.predict_proba(X)
    mean_proba = seg_proba.mean(axis=0)  # 各类别在所有段上的平均概率

    classes = list(model.classes_)
    # 平均概率最高的类别作为最终结论(比单纯投票更稳,且与置信度自洽)
    best_idx = int(np.argmax(mean_proba))
    label_en = classes[best_idx]

    proba = {LABEL_CN.get(c, c): float(mean_proba[i]) for i, c in enumerate(classes)}
    votes = {LABEL_CN.get(c, c): int((seg_preds == c).sum()) for c in classes}

    return {
        "label_en": label_en,
        "label_cn": LABEL_CN.get(label_en, label_en),
        "confidence": float(mean_proba[best_idx]),
        "proba": proba,
        "n_segments": n_segments,
        "votes": votes,
        "duration_sec": len(signal) / sampling_rate,
    }
