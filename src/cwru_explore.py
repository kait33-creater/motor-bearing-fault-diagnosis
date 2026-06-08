"""
CWRU 数据集探索脚本
====================
做三件事,帮助理解原始振动信号:
  1. 读取每个 .mat 文件,打印里面的变量名和信号长度
  2. 画出四种状态各一段振动波形,直观对比"正常 vs 故障"
  3. 报告采样率、信号点数等基本信息

.mat 文件里的变量命名规则(CWRU 官方):
  X097_DE_time —— 097 号文件,DE=驱动端(Drive End)加速度计采集的时域信号
  X097_FE_time —— FE=风扇端(Fan End)
  X097_BA_time —— BA=基座(Base)
  X097RPM      —— 实验时的转速
我们主要用 DE(驱动端)信号,因为故障轴承装在驱动端。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CWRU_RAW_DIR = PROJECT_ROOT / "data" / "cwru"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# 采样率:本批数据为 12kHz 驱动端
SAMPLING_RATE = 12000  # Hz

# 文件名 -> 中文标签(与下载脚本一致)
FILE_LABELS = {
    "97_normal.mat": "正常",
    "105_inner_race.mat": "内圈故障",
    "118_ball.mat": "滚珠故障",
    "130_outer_race.mat": "外圈故障",
}


def set_chinese_font():
    """设置中文字体,避免图表中文乱码。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False


def get_de_signal(mat_data):
    """
    从 loadmat 读出的字典里取出驱动端(DE)时域振动信号。
    变量名形如 X097_DE_time,编号会变,所以用后缀匹配。
    """
    for key in mat_data.keys():
        if key.endswith("_DE_time"):
            # loadmat 读出来是 (N, 1) 的二维数组,压平成一维
            return np.asarray(mat_data[key]).ravel()
    raise KeyError("未找到 *_DE_time 变量")


def explore_one(mat_path):
    """读取单个文件,打印结构信息,返回 DE 信号。"""
    mat_data = loadmat(mat_path)

    # 过滤掉 loadmat 自带的元信息键(以 __ 开头)
    var_names = [k for k in mat_data.keys() if not k.startswith("__")]

    signal = get_de_signal(mat_data)
    duration = len(signal) / SAMPLING_RATE

    print(f"文件:{mat_path.name}")
    print(f"  变量:{var_names}")
    print(f"  DE信号长度:{len(signal)} 点  ≈ {duration:.1f} 秒")
    print(f"  幅值范围:[{signal.min():.3f}, {signal.max():.3f}]")
    print()
    return signal


def plot_waveform_comparison(signals_dict, n_points=2000):
    """
    画四种状态的振动波形对比图(每种取前 n_points 个点)。
    2000 个点 ≈ 0.17 秒,足够看清冲击形态。
    """
    set_chinese_font()

    fig, axes = plt.subplots(len(signals_dict), 1, figsize=(10, 8), sharex=True)
    time_axis = np.arange(n_points) / SAMPLING_RATE * 1000  # 毫秒

    for ax, (label, signal) in zip(axes, signals_dict.items()):
        ax.plot(time_axis, signal[:n_points], linewidth=0.6)
        ax.set_ylabel("加速度")
        ax.set_title(label, loc="left", fontsize=11)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("时间 (毫秒)")
    fig.suptitle("CWRU 轴承振动信号:正常 vs 故障", fontsize=13)
    fig.tight_layout()

    out_path = FIGURE_DIR / "cwru_waveform_comparison.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"已保存波形对比图:{out_path}")


def main():
    print("========== CWRU 数据集探索 ==========\n")

    signals = {}
    for fname, label in FILE_LABELS.items():
        mat_path = CWRU_RAW_DIR / fname
        if not mat_path.exists():
            print(f"[缺失] {fname},请先运行 cwru_download.py")
            continue
        signals[label] = explore_one(mat_path)

    if len(signals) == len(FILE_LABELS):
        plot_waveform_comparison(signals)

    print("\n========== 探索完成 ==========")


if __name__ == "__main__":
    main()

