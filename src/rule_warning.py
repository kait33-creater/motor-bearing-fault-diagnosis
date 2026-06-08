from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 输入数据路径
FEATURE_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "motor_features.csv"

# 输出路径
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RULE_RESULT_PATH = PROCESSED_DATA_DIR / "motor_rule_warning_result.csv"
RULE_REPORT_PATH = REPORT_DIR / "rule_warning_summary.md"


def load_feature_data(file_path):
    """
    读取特征工程后的数据。
    """
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def calculate_risk_score(row):
    """
    根据电机运行参数计算风险得分。

    风险因素包括：
    1. 电流偏高；
    2. 温度偏高；
    3. 振动偏高；
    4. 电压波动；
    5. 负载率偏高；
    6. 转速异常波动。

    当前版本原则：
    - 避免过度误报；
    - 保证故障样本不会被大量漏报；
    - 兼顾瞬时异常和 30 分钟滑动窗口异常。
    """

    risk_score = 0

    # 1. 电流异常：瞬时电流明显偏高，或 30 分钟均值持续偏高
    if row["current"] > 16 or row["current_mean_30min"] > 14.5:
        risk_score += 1

    # 2. 温度异常：瞬时温度明显偏高，或持续高温
    if (
        row["temperature"] > 80
        or row["temperature_mean_30min"] > 70
        or row["temperature_max_30min"] > 88
    ):
        risk_score += 1

    # 3. 振动异常：振动明显升高
    if row["vibration"] > 1.15 or row["vibration_max_30min"] > 1.45:
        risk_score += 1

    # 4. 电压异常：电压明显越界或 30 分钟内波动较大
    if row["voltage"] < 200 or row["voltage"] > 240 or row["voltage_std_30min"] > 15:
        risk_score += 1

    # 5. 负载异常：瞬时负载率或 30 分钟平均负载率偏高
    if row["load_rate"] > 90 or row["load_rate_mean_30min"] > 85:
        risk_score += 1

    # 6. 转速异常：转速明显偏离额定区间，或波动较大
    if row["speed"] < 1400 or row["speed"] > 1560 or row["speed_std_30min"] > 60:
        risk_score += 1

    return risk_score


def infer_status(risk_score):
    """
    根据风险得分判断运行状态。

    risk_score = 0：正常
    risk_score = 1：预警
    risk_score >= 2：故障

    说明：
    在故障预警场景中，两个及以上关键指标异常时，
    应判定为故障风险，避免漏报。
    """

    if risk_score == 0:
        return "正常"
    elif risk_score == 1:
        return "预警"
    else:
        return "故障"


def infer_fault_type(row):
    """
    根据主要异常参数判断故障类型。

    判断原则：
    1. 过载优先看电流 + 负载率；
    2. 过热优先看温度；
    3. 轴承异常优先看振动和转速波动；
    4. 电压波动优先看电压越界和电压波动。
    """

    # 过载：电流和负载率同时偏高，或持续偏高
    if (
        (row["current"] > 16 and row["load_rate"] > 85)
        or (row["current_mean_30min"] > 14.5 and row["load_rate_mean_30min"] > 82)
    ):
        return "过载"

    # 过热：温度明显升高，可能伴随电流升高
    if (
        row["temperature"] > 80
        or row["temperature_mean_30min"] > 70
        or row["temperature_max_30min"] > 88
    ):
        return "过热"

    # 轴承异常：振动升高或转速波动明显
    if (
        row["vibration"] > 1.15
        or row["vibration_max_30min"] > 1.45
        or row["speed"] < 1400
        or row["speed"] > 1560
        or row["speed_std_30min"] > 60
    ):
        return "轴承异常"

    # 电压波动：电压明显越界或波动较大
    if row["voltage"] < 200 or row["voltage"] > 240 or row["voltage_std_30min"] > 15:
        return "电压波动"

    return "正常"


def get_rule_fault_type(row):
    """
    根据规则状态生成故障类型。

    如果规则状态为正常，则故障类型必须为正常；
    如果规则状态为预警或故障，但具体故障类型没有识别出来，
    则默认归为过载风险，避免出现“状态异常但故障类型正常”的矛盾。
    """

    if row["rule_status"] == "正常":
        return "正常"

    fault_type = infer_fault_type(row)

    if fault_type == "正常":
        return "过载"

    return fault_type


def generate_maintenance_advice(status, fault_type):
    """
    根据状态和故障类型生成维护建议。
    """

    if status == "正常":
        return "设备运行状态正常，建议保持常规巡检。"

    if fault_type == "过载":
        return "存在过载风险，建议检查负载情况，必要时降低负载或检查电机额定容量。"

    if fault_type == "过热":
        return "存在过热风险，建议检查散热条件、环境温度和电机绕组温升情况。"

    if fault_type == "轴承异常":
        return "存在轴承异常风险，建议检查轴承磨损、润滑状态和机械安装情况。"

    if fault_type == "电压波动":
        return "存在电压波动风险，建议检查供电电压稳定性和电气连接情况。"

    return "存在异常风险，建议安排进一步检修。"


def apply_rule_warning(df):
    """
    应用规则预警模型。
    """

    df = df.copy()

    df["rule_risk_score"] = df.apply(calculate_risk_score, axis=1)
    df["rule_status"] = df["rule_risk_score"].apply(infer_status)

    df["rule_fault_type"] = df.apply(get_rule_fault_type, axis=1)

    df["maintenance_advice"] = df.apply(
        lambda row: generate_maintenance_advice(
            row["rule_status"],
            row["rule_fault_type"],
        ),
        axis=1,
    )

    return df


def evaluate_rule_model(df):
    """
    评估规则预警结果与原始标签的一致性。
    """

    status_accuracy = accuracy_score(df["status"], df["rule_status"])
    fault_accuracy = accuracy_score(df["fault_type"], df["rule_fault_type"])

    status_report = classification_report(
        df["status"],
        df["rule_status"],
        zero_division=0,
    )

    fault_report = classification_report(
        df["fault_type"],
        df["rule_fault_type"],
        zero_division=0,
    )

    status_cm = confusion_matrix(df["status"], df["rule_status"])
    fault_cm = confusion_matrix(df["fault_type"], df["rule_fault_type"])

    return {
        "status_accuracy": status_accuracy,
        "fault_accuracy": fault_accuracy,
        "status_report": status_report,
        "fault_report": fault_report,
        "status_cm": status_cm,
        "fault_cm": fault_cm,
    }


def save_rule_report(df, eval_result, output_path):
    """
    保存规则预警分析报告。
    """

    rule_status_counts = df["rule_status"].value_counts()
    rule_fault_counts = df["rule_fault_type"].value_counts()

    high_risk_cases = df[df["rule_status"] == "故障"].head(10)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 规则阈值故障预警模型报告\n\n")

        f.write("## 1. 模型说明\n\n")
        f.write(
            "本模型基于电机运行参数设置工程阈值，"
            "从电流、温度、振动、电压、负载率和转速等方面计算风险得分，"
            "并根据风险得分将电机运行状态划分为正常、预警和故障。\n\n"
        )

        f.write("## 2. 规则预警状态分布\n\n")
        for status, count in rule_status_counts.items():
            ratio = count / len(df) * 100
            f.write(f"- {status}：{count} 条，占比 {ratio:.2f}%\n")

        f.write("\n## 3. 规则故障类型分布\n\n")
        for fault_type, count in rule_fault_counts.items():
            ratio = count / len(df) * 100
            f.write(f"- {fault_type}：{count} 条，占比 {ratio:.2f}%\n")

        f.write("\n## 4. 与原始标签的一致性评估\n\n")
        f.write(f"- 状态识别准确率：{eval_result['status_accuracy']:.4f}\n")
        f.write(f"- 故障类型识别准确率：{eval_result['fault_accuracy']:.4f}\n\n")

        f.write("### 4.1 运行状态分类报告\n\n")
        f.write("```text\n")
        f.write(eval_result["status_report"])
        f.write("\n```\n\n")

        f.write("### 4.2 故障类型分类报告\n\n")
        f.write("```text\n")
        f.write(eval_result["fault_report"])
        f.write("\n```\n\n")

        f.write("### 4.3 运行状态混淆矩阵\n\n")
        f.write("```text\n")
        f.write(str(eval_result["status_cm"]))
        f.write("\n```\n\n")

        f.write("### 4.4 故障类型混淆矩阵\n\n")
        f.write("```text\n")
        f.write(str(eval_result["fault_cm"]))
        f.write("\n```\n\n")

        f.write("## 5. 典型故障样本示例\n\n")

        if len(high_risk_cases) == 0:
            f.write("当前规则模型未识别到故障样本。\n")
        else:
            show_cols = [
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

            f.write("```text\n")
            f.write(high_risk_cases[show_cols].to_string(index=False))
            f.write("\n```\n\n")

        f.write("## 6. 初步结论\n\n")
        f.write(
            "规则阈值模型具有较强的可解释性，能够直接反映电机运行参数与故障风险之间的关系。"
            "从工程应用角度看，该方法适合作为早期预警系统的基线模型。"
            "不过，规则模型依赖人工阈值设定，可能存在误报或漏报问题。"
            "后续可以进一步引入机器学习模型，与规则模型进行对比分析，提升故障识别的稳定性和泛化能力。\n"
        )


def save_result(df, output_path):
    """
    保存规则预警结果。
    """
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main():
    print("开始读取特征数据...")
    df = load_feature_data(FEATURE_DATA_PATH)

    print("开始执行规则预警模型...")
    result_df = apply_rule_warning(df)

    print("开始评估规则预警结果...")
    eval_result = evaluate_rule_model(result_df)

    print("\n========== 规则预警结果 ==========")

    print("真实状态分布：")
    print(result_df["status"].value_counts())

    print("\n规则状态分布：")
    print(result_df["rule_status"].value_counts())

    print("\n真实故障类型分布：")
    print(result_df["fault_type"].value_counts())

    print("\n规则故障类型分布：")
    print(result_df["rule_fault_type"].value_counts())

    print("\n状态识别准确率：")
    print(eval_result["status_accuracy"])

    print("\n故障类型识别准确率：")
    print(eval_result["fault_accuracy"])

    save_result(result_df, RULE_RESULT_PATH)
    save_rule_report(result_df, eval_result, RULE_REPORT_PATH)

    print("\n========== 规则预警模型完成 ==========")
    print(f"规则预警结果保存路径：{RULE_RESULT_PATH}")
    print(f"规则预警报告保存路径：{RULE_REPORT_PATH}")


if __name__ == "__main__":
    main()