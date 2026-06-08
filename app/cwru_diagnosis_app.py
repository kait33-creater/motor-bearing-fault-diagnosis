"""
CWRU 轴承故障诊断 —— 在线诊断看板
====================================
上传一段电机振动信号(.mat / .csv),实时输出轴承故障类型与各类别置信度。

运行:
    python -m streamlit run app/cwru_diagnosis_app.py

诊断逻辑见 src/cwru_inference.py(与本界面解耦,可单独命令行测试)。
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# 让本文件能 import src/ 下的推理模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cwru_inference import (  # noqa: E402
    load_model,
    load_signal_from_mat,
    load_signal_from_csv,
    diagnose_signal,
    SEGMENT_LENGTH,
    SAMPLING_RATE,
)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
    "axes.unicode_minus": False,
})

st.set_page_config(page_title="电机轴承故障在线诊断", page_icon="🩺", layout="wide")


@st.cache_resource
def get_model():
    """加载模型(缓存,避免每次交互都重新读盘)。"""
    return load_model()


st.title("🩺 电机轴承故障在线诊断")
st.markdown(
    "上传一段电机驱动端振动信号,系统提取时频域特征并用随机森林模型实时诊断轴承状态"
    "(正常 / 内圈故障 / 滚珠故障 / 外圈故障),给出故障类型与各类别置信度。"
)

try:
    model, feature_cols, labels = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

with st.expander("📋 使用说明 / 数据格式", expanded=False):
    st.markdown(
        f"""
- **支持格式**:`.mat`(CWRU 原始文件,含 `*_DE_time` 驱动端信号)或 `.csv`(单列振动幅值,可带表头)。
- **采样率**:模型基于 {SAMPLING_RATE} Hz 训练,信号会按每 {SEGMENT_LENGTH} 点分段诊断。
- **最短长度**:至少 {SEGMENT_LENGTH} 点(约 {SEGMENT_LENGTH / SAMPLING_RATE:.2f} 秒)才能诊断,越长越稳。
- **试用样本**:仓库 `data/cwru/` 下的 `.mat` 文件可直接上传测试(需先运行 `python src/cwru_download.py` 下载)。
"""
    )

uploaded = st.file_uploader("选择振动信号文件", type=["mat", "csv"])

if uploaded is None:
    st.info("请上传一个 .mat 或 .csv 振动信号文件开始诊断。")
    st.stop()

# 读取信号
try:
    if uploaded.name.lower().endswith(".mat"):
        signal = load_signal_from_mat(uploaded)
    else:
        signal = load_signal_from_csv(uploaded)
except Exception as e:
    st.error(f"读取信号失败:{e}")
    st.stop()

# 诊断
try:
    result = diagnose_signal(signal, model, feature_cols)
except ValueError as e:
    st.error(str(e))
    st.stop()

# ===== 诊断结论 =====
st.subheader("诊断结果")
label_cn = result["label_cn"]
conf = result["confidence"]
if result["label_en"] == "normal":
    st.success(f"✅ 诊断结论:**{label_cn}**　(置信度 {conf:.1%})")
else:
    st.error(f"⚠️ 诊断结论:**{label_cn}**　(置信度 {conf:.1%})")

c1, c2, c3 = st.columns(3)
c1.metric("信号时长", f"{result['duration_sec']:.1f} s")
c2.metric("诊断段数", result["n_segments"])
c3.metric("采样点数", f"{len(signal):,}")

# ===== 各类别置信度 =====
st.subheader("各类别置信度")
proba = result["proba"]
order = sorted(proba, key=proba.get, reverse=True)
fig_p, ax_p = plt.subplots(figsize=(7, 2.6))
colors = ["#C44E52" if k == result["label_cn"] else "#B0B0B0" for k in order]
ax_p.barh(order[::-1], [proba[k] for k in order[::-1]],
          color=colors[::-1])
ax_p.set_xlim(0, 1)
ax_p.set_xlabel("平均概率")
for i, k in enumerate(order[::-1]):
    ax_p.text(proba[k] + 0.01, i, f"{proba[k]:.1%}", va="center", fontsize=9)
ax_p.spines["top"].set_visible(False)
ax_p.spines["right"].set_visible(False)
st.pyplot(fig_p)
plt.close(fig_p)

# ===== 信号波形预览 =====
st.subheader("信号波形预览")
preview_n = min(len(signal), SEGMENT_LENGTH * 3)  # 最多画前 3 段,避免点太多卡顿
t = np.arange(preview_n) / SAMPLING_RATE
fig_w, ax_w = plt.subplots(figsize=(10, 3))
ax_w.plot(t, signal[:preview_n], linewidth=0.6, color="#4C72B0")
ax_w.set_xlabel("时间 (s)")
ax_w.set_ylabel("加速度幅值")
ax_w.set_title(f"上传信号波形(前 {preview_n:,} 点)")
ax_w.spines["top"].set_visible(False)
ax_w.spines["right"].set_visible(False)
st.pyplot(fig_w)
plt.close(fig_w)

st.caption(
    "诊断说明:信号被切分为多个 2048 点的片段,逐段提取 17 维时频域特征后由随机森林分类,"
    "最终类别取各段平均概率最高者,置信度为该类别的平均概率。"
)
