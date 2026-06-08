from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


# =========================
# 项目路径配置
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "motor_features.csv"
RULE_RESULT_PATH = PROJECT_ROOT / "data" / "processed" / "motor_rule_warning_result.csv"

STATUS_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "status_random_forest_model.pkl"
FAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "fault_random_forest_model.pkl"

FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"


# =========================
# 页面基础设置
# =========================
st.set_page_config(
    page_title="电机运行状态监测与故障预警系统",
    page_icon="⚙️",
    layout="wide",
)


# =========================
# 数据与模型加载函数
# =========================
@st.cache_data
def load_data():
    """
    加载规则预警结果数据。
    """
    df = pd.read_csv(RULE_RESULT_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_resource
def load_models():
    """
    加载机器学习模型。
    """
    with open(STATUS_MODEL_PATH, "rb") as f:
        status_model = pickle.load(f)

    with open(FAULT_MODEL_PATH, "rb") as f:
        fault_model = pickle.load(f)

    return status_model, fault_model


def get_feature_columns():
    """
    机器学习模型使用的特征字段。
    """
    return [
        "voltage",
        "current",
        "temperature",
        "vibration",
        "speed",
        "load_rate",
        "current_mean_30min",
        "current_std_30min",
        "temperature_mean_30min",
        "temperature_max_30min",
        "temperature_change",
        "vibration_mean_30min",
        "vibration_max_30min",
        "speed_mean_30min",
        "speed_std_30min",
        "voltage_mean_30min",
        "voltage_std_30min",
        "load_rate_mean_30min",
        "current_load_ratio",
        "temperature_load_ratio",
    ]


def get_latest_record(df, motor_id):
    """
    获取指定电机的最新一条记录。
    """
    motor_df = df[df["motor_id"] == motor_id].sort_values("timestamp")
    return motor_df.iloc[-1]


# =========================
# 主页面
# =========================
st.title("⚙️ 电机运行状态监测与故障预警系统")

st.markdown(
    """
本系统面向工业电机运行状态监测场景，基于电压、电流、温度、振动、转速、负载率等参数，
实现电机运行状态分析、规则阈值预警、机器学习故障识别和维护建议输出。
"""
)


# =========================
# 加载数据
# =========================
try:
    df = load_data()
    status_model, fault_model = load_models()
except FileNotFoundError as e:
    st.error("未找到必要的数据文件或模型文件，请先运行 data_generate.py、preprocess.py、feature_engineering.py、rule_warning.py 和 ml_model.py。")
    st.exception(e)
    st.stop()


# =========================
# 侧边栏
# =========================
st.sidebar.header("系统控制面板")

motor_ids = sorted(df["motor_id"].unique())
selected_motor = st.sidebar.selectbox("选择电机编号", motor_ids)

show_raw_data = st.sidebar.checkbox("显示原始数据表", value=False)

motor_df = df[df["motor_id"] == selected_motor].sort_values("timestamp")
latest_record = get_latest_record(df, selected_motor)


# =========================
# 关键指标展示
# =========================
st.subheader(f"📌 {selected_motor} 最新运行状态")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("规则判断状态", latest_record["rule_status"])

with col2:
    st.metric("规则故障类型", latest_record["rule_fault_type"])

with col3:
    st.metric("风险得分", int(latest_record["rule_risk_score"]))

with col4:
    st.metric("最新负载率", f"{latest_record['load_rate']:.2f} %")


st.info(f"维护建议：{latest_record['maintenance_advice']}")


# =========================
# 机器学习模型预测最新状态
# =========================
st.subheader("🤖 机器学习模型最新预测")

feature_cols = get_feature_columns()
latest_X = latest_record[feature_cols].to_frame().T

ml_status_pred = status_model.predict(latest_X)[0]
ml_fault_pred = fault_model.predict(latest_X)[0]

col5, col6 = st.columns(2)

with col5:
    st.metric("Random Forest 状态预测", ml_status_pred)

with col6:
    st.metric("Random Forest 故障类型预测", ml_fault_pred)


# =========================
# 参数趋势图
# =========================
st.subheader("📈 电机运行参数趋势")

tab1, tab2, tab3 = st.tabs(["电气参数", "机械参数", "负载参数"])

with tab1:
    st.markdown("### 电压变化趋势")
    st.line_chart(
        motor_df.set_index("timestamp")[["voltage"]]
    )

    st.markdown("### 电流变化趋势")
    st.line_chart(
        motor_df.set_index("timestamp")[["current"]]
    )

with tab2:
    st.markdown("### 温度变化趋势")
    st.line_chart(
        motor_df.set_index("timestamp")[["temperature"]]
    )

    st.markdown("### 振动变化趋势")
    st.line_chart(
        motor_df.set_index("timestamp")[["vibration"]]
    )

    st.markdown("### 转速变化趋势")
    st.line_chart(
        motor_df.set_index("timestamp")[["speed"]]
    )

with tab3:
    st.markdown("### 负载率变化趋势")
    st.line_chart(
        motor_df.set_index("timestamp")[["load_rate"]]
    )


# =========================
# 状态分布统计
# =========================
st.subheader("📊 数据分布统计")

col7, col8 = st.columns(2)

with col7:
    st.markdown("### 规则预警状态分布")
    status_counts = df["rule_status"].value_counts()
    st.bar_chart(status_counts)

with col8:
    st.markdown("### 规则故障类型分布")
    fault_counts = df["rule_fault_type"].value_counts()
    st.bar_chart(fault_counts)


# =========================
# 模型结果图片展示
# =========================
st.subheader("🧠 模型评估结果")

status_cm_path = FIGURE_DIR / "ml_status_confusion_matrix.png"
fault_cm_path = FIGURE_DIR / "ml_fault_confusion_matrix.png"

col9, col10 = st.columns(2)

with col9:
    if status_cm_path.exists():
        st.image(
            status_cm_path.read_bytes(),
            caption="运行状态识别混淆矩阵",
            width="stretch",
        )
    else:
        st.warning(f"未找到运行状态混淆矩阵图片：{status_cm_path}")

with col10:
    if fault_cm_path.exists():
        st.image(
            fault_cm_path.read_bytes(),
            caption="故障类型识别混淆矩阵",
            width="stretch",
        )
    else:
        st.warning(f"未找到故障类型混淆矩阵图片：{fault_cm_path}")


# =========================
# 高风险样本展示
# =========================
st.subheader("🚨 高风险样本列表")

high_risk_df = df[df["rule_status"] == "故障"][
    [
        "timestamp",
        "motor_id",
        "voltage",
        "current",
        "temperature",
        "vibration",
        "speed",
        "load_rate",
        "rule_risk_score",
        "rule_status",
        "rule_fault_type",
        "maintenance_advice",
    ]
].head(20)

st.dataframe(high_risk_df, use_container_width=True)


# =========================
# 原始数据展示
# =========================
if show_raw_data:
    st.subheader("📄 数据表预览")
    st.dataframe(df.head(200), use_container_width=True)


# =========================
# 页面底部说明
# =========================
st.markdown("---")
st.markdown(
    """
**说明：** 当前系统基于模拟电机运行数据构建，主要用于验证电机状态监测、规则预警和机器学习故障识别的完整流程。
后续可进一步接入真实传感器数据、增加在线监测模块，并优化模型泛化能力。
"""
)
