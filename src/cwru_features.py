"""
CWRU 振动信号特征提取脚本
==========================
把原始振动波形(每段十几万个点)转换成机器学习能用的特征表。

核心思路两步:
  1. 分段(加窗):把每段长信号切成许多固定长度的小段,每小段当作一个样本。
     这样 4 段信号就能产生几百个训练样本。
  2. 特征提取:对每个小段计算一组能刻画"故障特征"的统计量
     —— 时域特征 + 频域特征。

输出:data/processed/cwru_features.csv —— 每行一个样本,列是特征 + 类别标签。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import kurtosis, skew

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CWRU_RAW_DIR = PROJECT_ROOT / "data" / "cwru"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_CSV = PROCESSED_DIR / "cwru_features.csv"

SAMPLING_RATE = 12000      # 采样率 12kHz
SEGMENT_LENGTH = 2048      # 每段长度(点)。2048 ≈ 0.17 秒 ≈ 电机转 5 圈,且是 2 的幂,利于 FFT

# 文件名 -> 类别标签
FILE_LABELS = {
    "97_normal.mat": "normal",
    "105_inner_race.mat": "inner_race",
    "118_ball.mat": "ball",
    "130_outer_race.mat": "outer_race",
}


def time_domain_features(seg):
    """
    时域特征:直接从波形的形状算统计量。
    故障会让信号出现"冲击",这些特征就是用来量化冲击的。
    """
    abs_seg = np.abs(seg)
    rms = np.sqrt(np.mean(seg ** 2))          # 均方根:信号的"能量/有效幅值",故障越重越大
    peak = np.max(abs_seg)                     # 峰值:最大冲击幅度

    feats = {
        # 基本统计量
        "mean": np.mean(seg),                  # 均值(通常接近0)
        "std": np.std(seg),                    # 标准差:波动幅度
        "rms": rms,                            # 均方根
        "peak": peak,                          # 峰值
        "abs_mean": np.mean(abs_seg),          # 整流平均幅值

        # 形状类无量纲指标(对故障最敏感,且不受信号整体大小影响)
        "kurtosis": kurtosis(seg),             # 峭度:波形"尖锐程度"。正常≈0,有冲击时显著>0,是冲击型故障的金标准
        "skewness": skew(seg),                 # 偏度:波形对称性
        "crest_factor": peak / rms,            # 峰值因子 = 峰值/RMS:衡量是否有突出尖峰
        "shape_factor": rms / np.mean(abs_seg),       # 波形因子
        "impulse_factor": peak / np.mean(abs_seg),    # 脉冲因子:对孤立冲击敏感
        "clearance_factor": peak / (np.mean(np.sqrt(abs_seg)) ** 2),  # 裕度因子:对早期微弱故障最敏感
    }
    return feats


def freq_domain_features(seg, fs=SAMPLING_RATE):
    """
    频域特征:先做 FFT 把信号从"时间"变到"频率",再从频谱里取统计量。
    故障的周期性冲击会在频谱里形成特定的峰,所以频域能补充时域看不到的信息。
    """
    n = len(seg)
    # 实数信号用 rfft,只取正频率部分
    spectrum = np.abs(np.fft.rfft(seg))        # 幅值谱
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)     # 对应的频率轴(Hz)

    # 把幅值谱归一化成"概率分布",用于算频谱的重心和分散程度
    psd = spectrum ** 2                         # 功率谱
    psd_sum = np.sum(psd) + 1e-12               # 防止除零

    # 频率重心:能量主要集中在高频还是低频。故障常使能量往高频搬
    freq_center = np.sum(freqs * psd) / psd_sum
    # 频率标准差:能量在频率上的分散程度
    freq_std = np.sqrt(np.sum(((freqs - freq_center) ** 2) * psd) / psd_sum)

    feats = {
        "fft_mean": np.mean(spectrum),          # 频谱平均幅值
        "fft_std": np.std(spectrum),            # 频谱幅值波动
        "fft_max": np.max(spectrum),            # 频谱最大峰
        "freq_center": freq_center,             # 频率重心(平均频率)
        "freq_std": freq_std,                   # 频率分散度
        "spectral_energy": psd_sum,             # 总频谱能量
    }
    return feats


def extract_features_from_signal(signal, label):
    """
    把一整段长信号切成多个小段,对每段提取时域+频域特征。
    返回一个特征字典列表(每段一条)。
    """
    rows = []
    n_segments = len(signal) // SEGMENT_LENGTH   # 不足一段的尾部丢弃

    for i in range(n_segments):
        seg = signal[i * SEGMENT_LENGTH:(i + 1) * SEGMENT_LENGTH]
        feats = {}
        feats.update(time_domain_features(seg))
        feats.update(freq_domain_features(seg))
        feats["label"] = label
        rows.append(feats)

    return rows


def get_de_signal(mat_data):
    """从 .mat 字典里取驱动端(DE)时域信号,压平成一维。"""
    for key in mat_data.keys():
        if key.endswith("_DE_time"):
            return np.asarray(mat_data[key]).ravel()
    raise KeyError("未找到 *_DE_time 变量")


def main():
    print("========== CWRU 特征提取 ==========\n")

    all_rows = []
    for fname, label in FILE_LABELS.items():
        mat_path = CWRU_RAW_DIR / fname
        if not mat_path.exists():
            print(f"[缺失] {fname},请先运行 cwru_download.py")
            continue

        signal = get_de_signal(loadmat(mat_path))
        rows = extract_features_from_signal(signal, label)
        all_rows.extend(rows)
        print(f"{label:12s}: 信号 {len(signal):>7d} 点 -> {len(rows):>3d} 个样本")

    df = pd.DataFrame(all_rows)
    df.to_csv(FEATURE_CSV, index=False, encoding="utf-8-sig")

    print(f"\n特征表保存到:{FEATURE_CSV}")
    print(f"总样本数:{len(df)},特征数:{df.shape[1] - 1}(不含标签)")
    print("\n各类别样本数:")
    print(df["label"].value_counts())
    print("\n特征列:")
    print([c for c in df.columns if c != "label"])


if __name__ == "__main__":
    main()


