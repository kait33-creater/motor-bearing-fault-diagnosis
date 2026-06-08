from pathlib import Path

import numpy as np
import pandas as pd


# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 输入数据路径
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "motor_data_cleaned.csv"

# 输出数据路径
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_DATA_PATH = PROCESSED_DATA_DIR / "motor_features.csv"


def load_cleaned_data(file_path):
    """
    读取清洗后的电机运行数据。
    """
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def add_rolling_features(df, window_size=6):
    """
    构造滑动窗口特征。

    window_size=6 表示使用最近 6 个采样点。
    当前数据采样间隔为 5 分钟，所以 6 个采样点约等于 30 分钟。
    """

    df = df.copy()
    df = df.sort_values(["motor_id", "timestamp"]).reset_index(drop=True)

    group = df.groupby("motor_id")

    # 电流特征
    df["current_mean_30min"] = group["current"].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )
    df["current_std_30min"] = group["current"].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).std()
    )

    # 温度特征
    df["temperature_mean_30min"] = group["temperature"].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )
    df["temperature_max_30min"] = group["temperature"].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).max()
    )
    df["temperature_change"] = group["temperature"].diff()

    # 振动特征
    df["vibration_mean_30min"] = group["vibration"].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )
    df["vibration_max_30min"] = group["vibration"].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).max()
    )

    # 转速特征
    df["speed_mean_30min"] = group["speed"].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )
    df["speed_std_30min"] = group["speed"].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).std()
    )

    # 电压特征
    df["voltage_mean_30min"] = group["voltage"].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )
    df["voltage_std_30min"] = group["voltage"].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).std()
    )

    # 负载率特征
    df["load_rate_mean_30min"] = group["load_rate"].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )

    # 比值特征
    df["current_load_ratio"] = df["current"] / df["load_rate"].replace(0, np.nan)
    df["temperature_load_ratio"] = df["temperature"] / df["load_rate"].replace(0, np.nan)

    # 缺失值处理
    df = df.fillna(0)

    return df


def check_features(df):
    """
    检查特征工程结果。
    """
    print("========== 特征工程结果检查 ==========")
    print(f"数据规模：{df.shape[0]} 行，{df.shape[1]} 列")

    print("\n前 5 行数据：")
    print(df.head())

    print("\n缺失值统计：")
    print(df.isnull().sum())

    print("\n状态分布：")
    print(df["status"].value_counts())

    print("\n故障类型分布：")
    print(df["fault_type"].value_counts())


def save_feature_data(df, output_path):
    """
    保存特征工程后的数据。
    """
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n========== 特征工程完成 ==========")
    print(f"特征数据保存路径：{output_path}")
    print(f"特征数据规模：{df.shape[0]} 行，{df.shape[1]} 列")


def main():
    print("开始读取清洗后数据...")
    df = load_cleaned_data(CLEANED_DATA_PATH)

    print("开始构造特征...")
    feature_df = add_rolling_features(df, window_size=6)

    check_features(feature_df)

    save_feature_data(feature_df, FEATURE_DATA_PATH)


if __name__ == "__main__":
    main()