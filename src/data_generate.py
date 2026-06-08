from pathlib import Path

import numpy as np
import pandas as pd


# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 原始数据保存目录
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def infer_fault_type(
    voltage,
    current,
    temperature,
    vibration,
    speed,
    load_rate,
):
    """
    当状态不是正常，但原始故障类型仍然是正常时，
    根据异常参数补充一个合理的故障类型。
    """
    if temperature > 75 and current > 13:
        return "过热"
    elif current > 15 or load_rate > 85:
        return "过载"
    elif vibration > 1.0 or speed < 1420 or speed > 1540:
        return "轴承异常"
    elif voltage < 205 or voltage > 235:
        return "电压波动"
    else:
        return "过载"


def generate_motor_data(n_per_motor=1000, seed=42):
    """
    生成电机运行状态模拟数据。
    每台电机生成 n_per_motor 条数据。
    默认 3 台电机，共 3000 条数据。
    """

    rng = np.random.default_rng(seed)

    motor_ids = ["M001", "M002", "M003"]
    all_records = []

    fault_types = [
        "正常",
        "过载",
        "过热",
        "轴承异常",
        "电压波动",
    ]

    fault_probabilities = [
        0.60,  # 正常
        0.12,  # 过载
        0.12,  # 过热
        0.09,  # 轴承异常
        0.07,  # 电压波动
    ]

    start_time = pd.Timestamp("2026-05-01 08:00:00")

    for motor_id in motor_ids:
        timestamps = pd.date_range(
            start=start_time,
            periods=n_per_motor,
            freq="5min",
        )

        sampled_faults = rng.choice(
            fault_types,
            size=n_per_motor,
            p=fault_probabilities,
        )

        for timestamp, fault_type in zip(timestamps, sampled_faults):
            # 正常工况基础值
            voltage = rng.normal(220, 4)
            current = rng.normal(10, 1.2)
            temperature = rng.normal(50, 5)
            vibration = rng.normal(0.45, 0.08)
            speed = rng.normal(1480, 20)
            load_rate = rng.normal(60, 10)

            # 根据故障类型调整参数
            if fault_type == "过载":
                current += rng.normal(6, 1.0)
                load_rate += rng.normal(30, 5)
                temperature += rng.normal(8, 3)

            elif fault_type == "过热":
                temperature += rng.normal(30, 5)
                current += rng.normal(3, 0.8)

            elif fault_type == "轴承异常":
                vibration += rng.normal(0.9, 0.2)
                speed += rng.normal(0, 60)

            elif fault_type == "电压波动":
                voltage += rng.normal(0, 18)
                current += rng.normal(0, 3)

            # 限制数据范围，避免出现明显不合理数值
            voltage = np.clip(voltage, 170, 260)
            current = np.clip(current, 0, 30)
            temperature = np.clip(temperature, 20, 120)
            vibration = np.clip(vibration, 0, 3)
            speed = np.clip(speed, 1200, 1700)
            load_rate = np.clip(load_rate, 0, 120)

            # 根据参数计算风险得分
            risk_score = 0

            if current > 15:
                risk_score += 1
            if temperature > 75:
                risk_score += 1
            if vibration > 1.0:
                risk_score += 1
            if voltage < 205 or voltage > 235:
                risk_score += 1
            if load_rate > 85:
                risk_score += 1
            if speed < 1420 or speed > 1540:
                risk_score += 1

            # status 表示风险等级
            # fault_type 表示具体故障原因
            if risk_score == 0:
                status = "正常"
                fault_type = "正常"
            elif risk_score == 1:
                status = "预警"
            else:
                status = "故障"

            # 如果状态不是正常，但故障类型仍然是正常，则补充具体故障类型
            if status != "正常" and fault_type == "正常":
                fault_type = infer_fault_type(
                    voltage,
                    current,
                    temperature,
                    vibration,
                    speed,
                    load_rate,
                )

            # 注意：这一段必须在 for timestamp, fault_type in zip(...) 循环里面
            all_records.append(
                {
                    "timestamp": timestamp,
                    "motor_id": motor_id,
                    "voltage": round(voltage, 2),
                    "current": round(current, 2),
                    "temperature": round(temperature, 2),
                    "vibration": round(vibration, 3),
                    "speed": round(speed, 2),
                    "load_rate": round(load_rate, 2),
                    "status": status,
                    "fault_type": fault_type,
                }
            )

    df = pd.DataFrame(all_records)

    # 人为加入少量缺失值，方便后续练习数据清洗
    missing_columns = ["voltage", "current", "temperature", "vibration"]

    for col in missing_columns:
        missing_indices = rng.choice(
            df.index,
            size=int(len(df) * 0.01),
            replace=False,
        )
        df.loc[missing_indices, col] = np.nan

    return df


def main():
    df = generate_motor_data(n_per_motor=1000)

    output_path = RAW_DATA_DIR / "motor_data.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("电机运行模拟数据生成完成！")
    print(f"数据保存路径：{output_path}")
    print(f"数据规模：{df.shape[0]} 行，{df.shape[1]} 列")

    print("\n前 5 行数据：")
    print(df.head())

    print("\n各状态数量：")
    print(df["status"].value_counts())

    print("\n各故障类型数量：")
    print(df["fault_type"].value_counts())


if __name__ == "__main__":
    main()