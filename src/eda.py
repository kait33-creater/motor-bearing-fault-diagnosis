from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 数据路径
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "motor_data_cleaned.csv"

# 输出路径
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def set_chinese_font():
    """
    设置中文字体，避免 matplotlib 中文乱码。
    Windows 一般可以使用 Microsoft YaHei 或 SimHei。
    """
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_cleaned_data(file_path):
    """
    读取清洗后的电机运行数据。
    """
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def plot_status_distribution(df):
    """
    绘制运行状态分布图。
    """
    status_counts = df["status"].value_counts()

    plt.figure(figsize=(8, 5))
    status_counts.plot(kind="bar")
    plt.title("电机运行状态分布")
    plt.xlabel("运行状态")
    plt.ylabel("样本数量")
    plt.xticks(rotation=0)
    plt.tight_layout()

    output_path = FIGURE_DIR / "status_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"已保存：{output_path}")


def plot_fault_type_distribution(df):
    """
    绘制故障类型分布图。
    """
    fault_counts = df["fault_type"].value_counts()

    plt.figure(figsize=(8, 5))
    fault_counts.plot(kind="bar")
    plt.title("电机故障类型分布")
    plt.xlabel("故障类型")
    plt.ylabel("样本数量")
    plt.xticks(rotation=0)
    plt.tight_layout()

    output_path = FIGURE_DIR / "fault_type_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"已保存：{output_path}")


def plot_parameter_trend(df, motor_id, column, ylabel, filename):
    """
    绘制某台电机某个参数的时间趋势图。
    """
    motor_df = df[df["motor_id"] == motor_id].copy()

    plt.figure(figsize=(12, 5))
    plt.plot(motor_df["timestamp"], motor_df[column], linewidth=1)
    plt.title(f"{motor_id} 电机 {ylabel} 时间变化趋势")
    plt.xlabel("时间")
    plt.ylabel(ylabel)
    plt.xticks(rotation=30)
    plt.tight_layout()

    output_path = FIGURE_DIR / filename
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"已保存：{output_path}")


def generate_eda_report(df):
    """
    生成第一版探索性数据分析报告。
    """
    status_counts = df["status"].value_counts()
    fault_counts = df["fault_type"].value_counts()

    numeric_cols = [
        "voltage",
        "current",
        "temperature",
        "vibration",
        "speed",
        "load_rate",
    ]

    desc = df[numeric_cols].describe().round(2)

    report_path = REPORT_DIR / "eda_summary.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 电机运行数据探索性分析报告\n\n")

        f.write("## 1. 数据基本情况\n\n")
        f.write(f"- 数据总量：{df.shape[0]} 行\n")
        f.write(f"- 字段数量：{df.shape[1]} 列\n")
        f.write(f"- 电机数量：{df['motor_id'].nunique()} 台\n")
        f.write(f"- 起始时间：{df['timestamp'].min()}\n")
        f.write(f"- 结束时间：{df['timestamp'].max()}\n\n")

        f.write("## 2. 运行状态分布\n\n")
        for status, count in status_counts.items():
            ratio = count / len(df) * 100
            f.write(f"- {status}：{count} 条，占比 {ratio:.2f}%\n")

        f.write("\n## 3. 故障类型分布\n\n")
        for fault_type, count in fault_counts.items():
            ratio = count / len(df) * 100
            f.write(f"- {fault_type}：{count} 条，占比 {ratio:.2f}%\n")

        f.write("\n## 4. 数值字段统计描述\n\n")
        f.write(desc.to_markdown())
        f.write("\n\n")

        f.write("## 5. 初步分析结论\n\n")
        f.write(
            "从运行状态分布来看，数据中正常状态样本占比较高，"
            "同时包含一定比例的预警和故障样本，能够支撑后续故障预警模型分析。\n\n"
        )
        f.write(
            "从故障类型分布来看，数据包含过载、过热、轴承异常和电压波动等典型电机异常场景，"
            "能够体现电流、温度、振动、转速、电压等参数与故障状态之间的关系。\n\n"
        )
        f.write(
            "后续可以进一步构造滑动窗口特征，例如电流均值、温度变化率、振动峰值、转速波动率等，"
            "用于规则预警和机器学习故障识别。\n"
        )

    print(f"已保存：{report_path}")


def main():
    set_chinese_font()

    print("开始读取清洗后数据...")
    df = load_cleaned_data(CLEANED_DATA_PATH)

    print(f"数据规模：{df.shape[0]} 行，{df.shape[1]} 列")
    print("\n运行状态分布：")
    print(df["status"].value_counts())

    print("\n故障类型分布：")
    print(df["fault_type"].value_counts())

    print("\n开始生成图表...")

    plot_status_distribution(df)
    plot_fault_type_distribution(df)

    # 先以 M001 电机为例绘制时间趋势图
    motor_id = "M001"

    plot_parameter_trend(
        df,
        motor_id=motor_id,
        column="voltage",
        ylabel="电压 / V",
        filename="M001_voltage_trend.png",
    )

    plot_parameter_trend(
        df,
        motor_id=motor_id,
        column="current",
        ylabel="电流 / A",
        filename="M001_current_trend.png",
    )

    plot_parameter_trend(
        df,
        motor_id=motor_id,
        column="temperature",
        ylabel="温度 / ℃",
        filename="M001_temperature_trend.png",
    )

    plot_parameter_trend(
        df,
        motor_id=motor_id,
        column="vibration",
        ylabel="振动强度",
        filename="M001_vibration_trend.png",
    )

    plot_parameter_trend(
        df,
        motor_id=motor_id,
        column="speed",
        ylabel="转速 / rpm",
        filename="M001_speed_trend.png",
    )

    plot_parameter_trend(
        df,
        motor_id=motor_id,
        column="load_rate",
        ylabel="负载率 / %",
        filename="M001_load_rate_trend.png",
    )

    print("\n开始生成分析报告...")
    generate_eda_report(df)

    print("\n探索性数据分析完成！")


if __name__ == "__main__":
    main()