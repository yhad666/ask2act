import os, csv, argparse, cv2
from pathlib import Path
from collections import defaultdict

DEFAULT_DATA_ROOT = (Path(__file__).resolve().parents[1] / "calib_data").as_posix()


# -------- CSV 工具 --------
def load_grouped(csv_path):
    """
    把 CSV 按 image 分组:
    返回 (image_list, {image: [rows]}, fieldnames)
    如果文件不存在，返回空结构。
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
    按 image_order 的顺序，把 rows_by_image 里的所有行写到 out_csv。
    每次标完一张图片就调用一次，相当于“实时保存”。
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


# -------- 两种标注交互方式 --------
def annotate_cli(vis_path, recs):
    """
    纯命令行交互（headless 模式）
    """
    print(f"\n=== {vis_path} ===")
    print("框：", ", ".join(
        [f'#{r["det_id"]}:{r["label"]}:{float(r["score"]):.2f}' for r in recs]
    ))
    fp = input("FP ids (空格分隔): ").strip()
    tp = input("TP ids (可留空): ").strip()

    fps = set(int(x) for x in fp.split() if x.isdigit())
    tps = set(int(x) for x in tp.split() if x.isdigit())

    for r in recs:
        did = int(r["det_id"])
        r["is_fp"] = "1" if did in fps else r.get("is_fp", "")
        r["is_tp"] = "1" if did in tps else r.get("is_tp", "")
    return recs


def annotate_gui(vis_path, recs):
    """
    有显示器时用 GUI（你的服务器没显示器，一般不会走到这里）
    """
    img = cv2.imread(vis_path)
    if img is None:
        return annotate_cli(vis_path, recs)

    idx = 0
    FP, TP = set(), set()
    H, W = img.shape[:2]

    def draw(idx_):
        d = img.copy()
        if 0 <= idx_ < len(recs):
            r = recs[idx_]
            x1, y1, x2, y2 = map(lambda k: int(float(r[k])), ("x1", "y1", "x2", "y2"))
            cv2.rectangle(d, (x1, y1), (x2, y2), (0, 0, 255), 2)
        txt = f"[{idx_+1}/{len(recs)}] j/k 切换  f 切 FP  t 切 TP  n/s 下一张  q 退出"
        cv2.putText(d, txt, (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(d, f"FP:{sorted(FP)}", (10, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(d, f"TP:{sorted(TP)}", (10, 45),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return d

    while True:
        cv2.imshow("label", draw(idx))
        k = cv2.waitKey(0) & 0xFF
        if k in (ord("j"), 81):   # 左
            idx = (idx - 1) % max(1, len(recs))
        elif k in (ord("k"), 83): # 右
            idx = (idx + 1) % max(1, len(recs))
        elif k == ord("f"):
            FP.symmetric_difference_update({int(recs[idx]["det_id"])})
        elif k == ord("t"):
            TP.symmetric_difference_update({int(recs[idx]["det_id"])})
        elif k in (ord("n"), ord("s")):
            break
        elif k == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit

    cv2.destroyAllWindows()

    for r in recs:
        did = int(r["det_id"])
        r["is_fp"] = "1" if did in FP else r.get("is_fp", "")
        r["is_tp"] = "1" if did in TP else r.get("is_tp", "")
    return recs


# -------- 主逻辑（支持断点续标） --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--csv", default="work/exports/detections.csv",
                    help="原始检测结果 CSV，相对 data_root 的路径")
    ap.add_argument("--out", default="work/annotations/detections_labeled.csv",
                    help="带标注的输出 CSV，相对 data_root 的路径")
    ap.add_argument("--headless", action="store_true",
                    help="仅用命令行交互，不启用 GUI")
    args = ap.parse_args()

    data_root = args.data_root
    src_csv   = os.path.join(data_root, args.csv)
    out_csv   = os.path.join(data_root, args.out)

    # 1) 读取原始 detections.csv
    images, grouped_src, fieldnames = load_grouped(src_csv)
    if not images:
        print("❌ 没在 detections.csv 里找到任何行，请先跑 data_process.py")
        return

    # 确保有 is_tp / is_fp 字段
    if "is_tp" not in fieldnames:
        fieldnames = list(fieldnames) + ["is_tp"]
    if "is_fp" not in fieldnames:
        fieldnames = list(fieldnames) + ["is_fp"]

    # 2) 如果已经有标注文件，加载进来（断点续标）
    _, grouped_labeled, _ = load_grouped(out_csv)
    rows_by_image = {}

    done_images = set(grouped_labeled.keys())
    for img in done_images:
        # 直接用已有标注
        rows_by_image[img] = grouped_labeled[img]

    if done_images:
        print(f"🔁 发现已有标注文件，将跳过已完成的 {len(done_images)} 张图。")

    # 3) 逐张标注
    for rel in images:
        # 已经标过的图片直接跳过
        if rel in rows_by_image:
            continue

        vis_path = os.path.join(
            data_root, "work/exports/vis", rel
        )  # 与 data_process.py 的可视化路径对应

        recs = grouped_src[rel]

        # 进入交互标注
        try:
            if args.headless:
                recs = annotate_cli(vis_path, recs)
            else:
                recs = annotate_gui(vis_path, recs)
        except Exception as e:
            print(f"[warn] GUI 失败，改用命令行: {e}")
            recs = annotate_cli(vis_path, recs)

        # 更新内存缓存
        rows_by_image[rel] = recs

        # ✅ 每张图片标注完立刻保存到 out_csv（防止中途挂掉）
        save_all(images, rows_by_image, out_csv, fieldnames)
        print(f"💾 已保存进度到: {out_csv}")

    print(f"\n✅ 全部标注完成：{out_csv}")


if __name__ == "__main__":
    main()
