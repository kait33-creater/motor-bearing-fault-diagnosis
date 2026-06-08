from pathlib import Path

import pandas as pd


# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 原始数据路径
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "motor_data.csv"

# 清洗后数据保存路径
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "motor_data_cleaned.csv"


def load_data(file_path):
    """
    读取原始电机运行数据。
    """
    df = pd.read_csv(file_path)
    return df


def check_data(df):
    """
    检查数据基本情况。
    """
    print("========== 数据基本信息 ==========")
    print(f"数据规模：{df.shape[0]} 行，{df.shape[1]} 列")

    print("\n========== 前 5 行数据 ==========")
    print(df.head())

    print("\n========== 字段类型 ==========")
    print(df.dtypes)

    print("\n========== 缺失值统计 ==========")
    print(df.isnull().sum())

    print("\n========== 状态分布 ==========")
    print(df["status"].value_counts())

    print("\n========== 故障类型分布 ==========")
    print(df["fault_type"].value_counts())


def clean_data(df):
    """
    清洗电机运行数据。

    清洗步骤：
    1. 将 timestamp 转换为时间格式；
    2. 按 motor_id 和 timestamp 排序；
    3. 对数值字段缺失值进行同一电机内前向填充；
    4. 若仍有缺失值，则使用同一电机均值填充；
    5. 删除重复行。
    """

    df = df.copy()

    # 1. 转换时间格式
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 2. 按电机编号和时间排序
    df = df.sort_values(["motor_id", "timestamp"]).reset_index(drop=True)

    # 3. 数值型字段
    numeric_cols = [
        "voltage",
        "current",
        "temperature",
        "vibration",
        "speed",
        "load_rate",
    ]

    # 4. 按 motor_id 分组进行前向填充
    df[numeric_cols] = df.groupby("motor_id")[numeric_cols].ffill()

    # 5. 如果每台电机开头仍有缺失值，用同一电机均值填充
    for col in numeric_cols:
        df[col] = df.groupby("motor_id")[col].transform(
            lambda x: x.fillna(x.mean())
        )

    # 6. 极少数兜底：如果仍然有缺失值，用全局均值填充
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    # 7. 删除重复行
    df = df.drop_duplicates().reset_index(drop=True)

    return df


def save_cleaned_data(df, output_path):
    """
    保存清洗后的数据。
    """
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n========== 数据清洗完成 ==========")
    print(f"清洗后数据保存路径：{output_path}")
    print(f"清洗后数据规模：{df.shape[0]} 行，{df.shape[1]} 列")


def main():
    print("开始读取原始数据...")
    df = load_data(RAW_DATA_PATH)

    print("\n清洗前数据检查：")
    check_data(df)

    print("\n开始清洗数据...")
    cleaned_df = clean_data(df)

    print("\n========== 清洗后缺失值统计 ==========")
    print(cleaned_df.isnull().sum())

    save_cleaned_data(cleaned_df, CLEANED_DATA_PATH)


if __name__ == "__main__":
    main()