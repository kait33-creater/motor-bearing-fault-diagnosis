from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 输入数据路径
FEATURE_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "motor_features.csv"

# 输出路径
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ML_RESULT_PATH = PROCESSED_DATA_DIR / "motor_ml_prediction_result.csv"
ML_REPORT_PATH = REPORT_DIR / "ml_model_summary.md"

STATUS_MODEL_PATH = MODEL_DIR / "status_random_forest_model.pkl"
FAULT_MODEL_PATH = MODEL_DIR / "fault_random_forest_model.pkl"


def set_chinese_font():
    """
    设置中文字体，避免图表中文乱码。
    """
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_feature_data(file_path):
    """
    读取特征工程后的数据。
    """
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def get_feature_columns():
    """
    选择用于机器学习模型训练的特征字段。
    """

    feature_cols = [
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

    return feature_cols


def train_models(X_train, y_train):
    """
    训练两个模型：
    1. Logistic Regression：基础对照模型；
    2. Random Forest：主要机器学习模型。
    """

    logistic_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    random_forest_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )

    logistic_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)

    return logistic_model, random_forest_model


def evaluate_model(model, X_test, y_test):
    """
    评估模型效果。
    """

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()))

    return {
        "y_pred": y_pred,
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(cm, labels, title, output_path):
    """
    绘制混淆矩阵图。
    """

    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.xticks(range(len(labels)), labels, rotation=30)
    plt.yticks(range(len(labels)), labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"已保存：{output_path}")


def save_model(model, output_path):
    """
    使用 pickle 保存模型。
    """
    with open(output_path, "wb") as f:
        pickle.dump(model, f)


def save_report(
    status_results,
    fault_results,
    status_labels,
    fault_labels,
    output_path,
):
    """
    保存机器学习模型报告。
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 机器学习故障识别模型报告\n\n")

        f.write("## 1. 模型说明\n\n")
        f.write(
            "本部分基于特征工程后的电机运行数据，使用 Logistic Regression 和 Random Forest "
            "建立机器学习分类模型，分别完成运行状态识别和故障类型识别。"
            "其中 Logistic Regression 作为基础对照模型，Random Forest 作为主要模型。\n\n"
        )

        f.write("## 2. 运行状态识别结果\n\n")

        f.write("### 2.1 Logistic Regression\n\n")
        f.write(f"- 准确率：{status_results['logistic']['accuracy']:.4f}\n\n")
        f.write("```text\n")
        f.write(status_results["logistic"]["report"])
        f.write("\n```\n\n")

        f.write("### 2.2 Random Forest\n\n")
        f.write(f"- 准确率：{status_results['random_forest']['accuracy']:.4f}\n\n")
        f.write("```text\n")
        f.write(status_results["random_forest"]["report"])
        f.write("\n```\n\n")

        f.write("## 3. 故障类型识别结果\n\n")

        f.write("### 3.1 Logistic Regression\n\n")
        f.write(f"- 准确率：{fault_results['logistic']['accuracy']:.4f}\n\n")
        f.write("```text\n")
        f.write(fault_results["logistic"]["report"])
        f.write("\n```\n\n")

        f.write("### 3.2 Random Forest\n\n")
        f.write(f"- 准确率：{fault_results['random_forest']['accuracy']:.4f}\n\n")
        f.write("```text\n")
        f.write(fault_results["random_forest"]["report"])
        f.write("\n```\n\n")

        f.write("## 4. 初步结论\n\n")
        f.write(
            "与规则阈值模型相比，机器学习模型能够从多维特征中自动学习状态与故障类型之间的关系，"
            "在故障识别任务中具有更强的数据适应能力。"
            "但机器学习模型的可解释性弱于规则模型，因此在工程应用中更适合与规则阈值方法结合使用，"
            "形成“规则基线 + 数据驱动模型”的综合预警方案。\n"
        )

    print(f"已保存：{output_path}")


def main():
    set_chinese_font()

    print("开始读取特征数据...")
    df = load_feature_data(FEATURE_DATA_PATH)

    feature_cols = get_feature_columns()

    X = df[feature_cols]
    y_status = df["status"]
    y_fault = df["fault_type"]

    print("开始划分训练集和测试集...")

    X_train_status, X_test_status, y_train_status, y_test_status = train_test_split(
        X,
        y_status,
        test_size=0.2,
        random_state=42,
        stratify=y_status,
    )

    X_train_fault, X_test_fault, y_train_fault, y_test_fault = train_test_split(
        X,
        y_fault,
        test_size=0.2,
        random_state=42,
        stratify=y_fault,
    )

    print("开始训练运行状态识别模型...")
    status_logistic, status_rf = train_models(X_train_status, y_train_status)

    print("开始训练故障类型识别模型...")
    fault_logistic, fault_rf = train_models(X_train_fault, y_train_fault)

    print("开始评估运行状态识别模型...")
    status_logistic_result = evaluate_model(
        status_logistic,
        X_test_status,
        y_test_status,
    )
    status_rf_result = evaluate_model(
        status_rf,
        X_test_status,
        y_test_status,
    )

    print("开始评估故障类型识别模型...")
    fault_logistic_result = evaluate_model(
        fault_logistic,
        X_test_fault,
        y_test_fault,
    )
    fault_rf_result = evaluate_model(
        fault_rf,
        X_test_fault,
        y_test_fault,
    )

    print("\n========== 机器学习模型结果 ==========")

    print("\n运行状态识别：")
    print(f"Logistic Regression 准确率：{status_logistic_result['accuracy']:.4f}")
    print(f"Random Forest 准确率：{status_rf_result['accuracy']:.4f}")

    print("\n故障类型识别：")
    print(f"Logistic Regression 准确率：{fault_logistic_result['accuracy']:.4f}")
    print(f"Random Forest 准确率：{fault_rf_result['accuracy']:.4f}")

    # 标签顺序
    status_labels = ["正常", "预警", "故障"]
    fault_labels = ["正常", "过载", "过热", "轴承异常", "电压波动"]

    # 重新计算固定标签顺序下的混淆矩阵
    status_rf_pred = status_rf.predict(X_test_status)
    fault_rf_pred = fault_rf.predict(X_test_fault)

    status_cm = confusion_matrix(y_test_status, status_rf_pred, labels=status_labels)
    fault_cm = confusion_matrix(y_test_fault, fault_rf_pred, labels=fault_labels)

    # 绘制混淆矩阵
    plot_confusion_matrix(
        status_cm,
        status_labels,
        "Random Forest 运行状态识别混淆矩阵",
        FIGURE_DIR / "ml_status_confusion_matrix.png",
    )

    plot_confusion_matrix(
        fault_cm,
        fault_labels,
        "Random Forest 故障类型识别混淆矩阵",
        FIGURE_DIR / "ml_fault_confusion_matrix.png",
    )

    # 保存预测结果
    prediction_df = pd.DataFrame(
        {
            "true_status": y_test_status.values,
            "pred_status": status_rf_pred,
            "true_fault_type": y_test_fault.values,
            "pred_fault_type": fault_rf_pred,
        }
    )
    prediction_df.to_csv(ML_RESULT_PATH, index=False, encoding="utf-8-sig")

    # 保存模型
    save_model(status_rf, STATUS_MODEL_PATH)
    save_model(fault_rf, FAULT_MODEL_PATH)

    # 保存报告
    status_results = {
        "logistic": status_logistic_result,
        "random_forest": status_rf_result,
    }

    fault_results = {
        "logistic": fault_logistic_result,
        "random_forest": fault_rf_result,
    }

    save_report(
        status_results,
        fault_results,
        status_labels,
        fault_labels,
        ML_REPORT_PATH,
    )

    print("\n========== 机器学习建模完成 ==========")
    print(f"预测结果保存路径：{ML_RESULT_PATH}")
    print(f"模型报告保存路径：{ML_REPORT_PATH}")
    print(f"状态识别模型保存路径：{STATUS_MODEL_PATH}")
    print(f"故障类型识别模型保存路径：{FAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()