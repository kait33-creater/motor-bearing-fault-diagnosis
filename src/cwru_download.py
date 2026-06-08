"""
CWRU 轴承数据集下载脚本
========================
从凯斯西储大学(Case Western Reserve University)轴承数据中心官网下载
原始振动信号 .mat 文件。

本脚本只下载"最小四分类数据集":
  - 工况固定为 12kHz 采样、驱动端(Drive End)、0 马力负载、0.007 英寸故障尺寸
  - 四个类别:正常 / 内圈故障 / 滚珠故障 / 外圈故障

数据来源:https://engineering.case.edu/bearingdatacenter
"""

from pathlib import Path
import urllib.request

# 项目根目录(本文件在 src/ 下,上一级就是项目根)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# CWRU 原始数据保存目录
CWRU_RAW_DIR = PROJECT_ROOT / "data" / "cwru"
CWRU_RAW_DIR.mkdir(parents=True, exist_ok=True)

# 官网文件下载 URL 模板
BASE_URL = "https://engineering.case.edu/sites/default/files/{file_id}.mat"

# 要下载的文件:文件编号 -> (类别英文名, 类别中文名)
# 工况:12kHz 驱动端,0HP 负载,0.007 英寸故障
FILES = {
    "97": ("normal", "正常"),
    "105": ("inner_race", "内圈故障"),
    "118": ("ball", "滚珠故障"),
    "130": ("outer_race", "外圈故障"),
}

# 多负载文件:文件编号 -> (类别英文名, 负载HP)
# 12kHz 驱动端,0.007 英寸故障,负载 0/1/2/3 HP
# 用于跨负载泛化评估(cross-load):训练用部分负载,测试用没见过的负载
FILES_MULTILOAD = {
    "97": ("normal", 0),     "98": ("normal", 1),     "99": ("normal", 2),     "100": ("normal", 3),
    "105": ("inner_race", 0), "106": ("inner_race", 1), "107": ("inner_race", 2), "108": ("inner_race", 3),
    "118": ("ball", 0),       "119": ("ball", 1),       "120": ("ball", 2),       "121": ("ball", 3),
    "130": ("outer_race", 0), "131": ("outer_race", 1), "132": ("outer_race", 2), "133": ("outer_race", 3),
}



def download_one(file_id, label_en):
    """
    下载单个 .mat 文件。
    保存文件名形如 130_outer_race.mat,便于后续按类别读取。
    如果文件已存在则跳过,避免重复下载。
    """
    save_path = CWRU_RAW_DIR / f"{file_id}_{label_en}.mat"

    if save_path.exists():
        size_mb = save_path.stat().st_size / 1024 / 1024
        print(f"[跳过] {save_path.name} 已存在({size_mb:.1f} MB)")
        return save_path

    url = BASE_URL.format(file_id=file_id)
    print(f"[下载] {url}")
    try:
        urllib.request.urlretrieve(url, save_path)
        size_mb = save_path.stat().st_size / 1024 / 1024
        print(f"[完成] 保存到 {save_path.name}({size_mb:.1f} MB)")
    except Exception as exc:
        print(f"[失败] {file_id}: {exc}")
        # 下载失败时删掉可能产生的残缺文件
        if save_path.exists():
            save_path.unlink()
        return None

    return save_path


def main():
    print(f"CWRU 数据保存目录:{CWRU_RAW_DIR}")
    print(f"计划下载 {len(FILES_MULTILOAD)} 个文件(多负载,用于跨负载评估)...\n")

    ok = 0
    for file_id, (label_en, load_hp) in FILES_MULTILOAD.items():
        save_name = f"{file_id}_{label_en}_{load_hp}hp"
        # download_one 用 label 拼文件名,这里直接传带负载的名字
        if download_one(file_id, f"{label_en}_{load_hp}hp") is not None:
            ok += 1

    print(f"\n========== 下载结束:{ok}/{len(FILES_MULTILOAD)} 成功 ==========")


if __name__ == "__main__":
    main()
