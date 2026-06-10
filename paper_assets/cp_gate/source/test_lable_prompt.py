# test_label_singleprompt_tp.py
import os, csv, argparse
from pathlib import Path
from collections import defaultdict

# 和 detect 脚本保持一致
DATA_ROOT = (Path(__file__).resolve().parents[1] / "calib_data").as_posix()

# 输入：检测结果 CSV（test_detect_singleprompt.py 的输出）
RAW_CSV = "work/exports/test_single_raw.csv"
# 输出：带 TP/FP 和 gt_total 的标注 CSV
OUT_CSV = "work/annotations/test_single_labeled_tp.csv"


def load_grouped(csv_path):
    """
    把 CSV 按 image 分组：
    返回 (image_list, {image: [rows]}, fieldnames)
    """
    g = defaultdict(list)
    if not os.path.exists(csv_path):
        return [], g, None
    with open(csv_path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        rows = list(rd)
        if not rows:
            return [], g, rd.fieldnames
        for r in rows:
            g[r["image"]].append(r)
        return sorted(g.keys()), g, rd.fieldnames


def save_all(image_order, rows_by_image, out_csv, fieldnames):
    """
    按 image_order 把 rows_by_image 里的所有行写到 out_csv。
    每张图片标完后调用一次，实现实时保存。
    """
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for img in image_order:
            if img not in rows_by_image:
                continue
            for r in rows_by_image[img]:
                wr.writerow(r)


def annotate_cli_tp(data_root, image, recs):
    """
    只标 TP 的命令行交互：
      - 你给出 TP ids
      - 其他所有框自动当 FP
      - 再输入真实物体总数 N
    """
    print(f"\n=== image: {image} ===")
    print("框列表：")
    print(", ".join(
        [f'#{r["det_id"]}:{r["label"]}:{float(r["score"]):.2f}' for r in recs]
    ))
    vis_path = os.path.join(data_root, "work/exports/vis", image)
    print(f"(可视化图片在: {vis_path})")

    tp = input("TP ids (空格分隔，比如: 0 3 5；如果一个 TP 都没有就回车): ").strip()
    gt = input("这张图在当前 prompt 下真实物体总数 N: ").strip()

    # 解析 TP id 集合
    tps = set(int(x) for x in tp.split() if x.isdigit())
    gt_total = int(gt) if gt.isdigit() else None

    for r in recs:
        did = int(r["det_id"])
        if did in tps:
            # 标为 TP，明确不是 FP
            r["is_tp"] = "1"
            r["is_fp"] = ""
        else:
            # 没被你选为 TP 的，一律视为 FP
            r["is_tp"] = ""
            r["is_fp"] = "1"

        if gt_total is not None:
            r["gt_total"] = str(gt_total)

    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DATA_ROOT)
    ap.add_argument("--csv_in", default=RAW_CSV)
    ap.add_argument("--csv_out", default=OUT_CSV)
    args = ap.parse_args()

    data_root = args.data_root
    src_csv = os.path.join(data_root, args.csv_in)
    out_csv = os.path.join(data_root, args.csv_out)

    images, grouped_src, fieldnames = load_grouped(src_csv)
    if not images:
        print("❌ 源 CSV 为空，请先跑 test_detect_singleprompt.py")
        return

    # 确保有 is_tp / is_fp / gt_total 字段
    if "is_tp" not in fieldnames:
        fieldnames = list(fieldnames) + ["is_tp"]
    if "is_fp" not in fieldnames:
        fieldnames = list(fieldnames) + ["is_fp"]
    if "gt_total" not in fieldnames:
        fieldnames = list(fieldnames) + ["gt_total"]

    rows_by_image = {}
    done_images = set()

    # 支持断点续标：如果 OUT_CSV 已存在，则读入已有标注
    if os.path.exists(out_csv):
        _, grouped_labeled, _ = load_grouped(out_csv)
        done_images = set(grouped_labeled.keys())
        for img in done_images:
            rows_by_image[img] = grouped_labeled[img]
        if done_images:
            print(f"🔁 发现已有标注文件，将跳过已完成的 {len(done_images)} 张图。")

    # 逐图标注
    for img in images:
        if img in done_images:
            continue

        recs = grouped_src[img]
        recs = annotate_cli_tp(data_root, img, recs)
        rows_by_image[img] = recs

        # 实时保存
        save_all(images, rows_by_image, out_csv, fieldnames)
        print(f"💾 已保存进度到: {out_csv}")

    print(f"\n✅ 全部 test 标注完成：{out_csv}")
    print("说明：在这个 CSV 中，所有没被你选为 TP 的框都被标成了 is_fp=1。")


if __name__ == "__main__":
    main()
