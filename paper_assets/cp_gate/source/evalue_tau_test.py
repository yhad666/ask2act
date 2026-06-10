
import os, csv, json, argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# test 数据根目录
TEST_ROOT = (Path(__file__).resolve().parents[1] / "calib_data").as_posix()

# test 标注好的 CSV（含 is_fp, gt_total）
TEST_LABELED_CSV = "work/annotations/test_single_labeled_tp.csv"

# calib 的 tau_grid.json 路径（相对 TEST_ROOT 或绝对路径）
TAU_GRID_JSON = "../calib_data/work/tau/tau_grid.json"

OUT_PREFIX = "work/eval/curve_single"  # 会生成 curve_single.*


def load_per_image_stats(csv_path):
    by_image = defaultdict(list)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"test 标注 CSV 不存在: {csv_path}")
    with open(csv_path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            by_image[r["image"]].append(r)

    per_image = {}
    for img, rows in by_image.items():
        N_set = set()
        for r in rows:
            v = r.get("gt_total", "").strip()
            if v:
                try:
                    N_set.add(int(v))
                except Exception:
                    pass
        if not N_set:
            print(f"⚠️ 图像 {img} 没有有效 gt_total，跳过。")
            continue
        if len(N_set) > 1:
            print(f"⚠️ 图像 {img} 出现多个 gt_total {N_set}，取最大。")
        N_i = max(N_set)

        tp_scores, fp_scores = [], []
        for r in rows:
            try:
                s = float(r["score"])
            except Exception:
                continue
            if r.get("is_fp", "").strip() == "1":
                fp_scores.append(s)
            else:
                tp_scores.append(s)

        per_image[img] = dict(
            N=N_i,
            tp_scores=np.asarray(tp_scores, float) if tp_scores else np.zeros((0,), float),
            fp_scores=np.asarray(fp_scores, float) if fp_scores else np.zeros((0,), float),
        )

    return per_image


def load_tau_grid(tau_grid_path):
    if not os.path.exists(tau_grid_path):
        raise FileNotFoundError(f"tau_grid 不存在: {tau_grid_path}")
    with open(tau_grid_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"tau_grid.json 格式不认识: {tau_grid_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=TEST_ROOT)
    ap.add_argument("--test_csv", default=TEST_LABELED_CSV)
    ap.add_argument("--tau_grid", default=TAU_GRID_JSON)
    ap.add_argument("--out_prefix", default=OUT_PREFIX)
    args = ap.parse_args()

    data_root = args.data_root
    test_csv_path = os.path.join(data_root, args.test_csv)

    per_image = load_per_image_stats(test_csv_path)
    if not per_image:
        print("❌ 没有任何带 gt_total 的图像，无法评估。")
        return

    images = sorted(per_image.keys())
    global_N = sum(per_image[img]["N"] for img in images)
    print(f"✅ 有效 test 图像数: {len(images)}, 真实物体总数 N = {global_N}")

    tau_grid_path = args.tau_grid
    if not os.path.isabs(tau_grid_path):
        tau_grid_path = os.path.join(data_root, tau_grid_path)
    tau_records = load_tau_grid(tau_grid_path)
    print(f"📐 tau_grid 里共有 {len(tau_records)} 条 (eps_fp, eps_tp) 记录。")

    results = []

    for rec in tau_records:
        eps_fp = float(rec.get("eps_fp", 0.0))
        eps_tp = float(rec.get("eps_tp", 0.0))
        tau_final = float(rec["tau_final"])

        TP_total = 0
        FP_total = 0

        for img in images:
            info = per_image[img]
            tp_scores = info["tp_scores"]
            fp_scores = info["fp_scores"]

            TP_keep = int(np.sum(tp_scores >= tau_final))
            FP_keep = int(np.sum(fp_scores >= tau_final))

            TP_total += TP_keep
            FP_total += FP_keep

        recall_obj = TP_total / global_N if global_N > 0 else 0.0
        denom = TP_total + FP_total
        if denom > 0:
            precision = TP_total / denom
        else:
            precision = 1.0
        if precision + recall_obj > 0:
            f1 = 2 * precision * recall_obj / (precision + recall_obj)
        else:
            f1 = 0.0

        fp_per_image = FP_total / len(images) if images else 0.0

        out_rec = dict(
            eps_fp=eps_fp,
            eps_tp=eps_tp,
            tau_final=tau_final,
            TP=TP_total,
            FP=FP_total,
            N=global_N,
            recall_obj=recall_obj,
            precision=precision,
            f1=f1,
            fp_per_image=fp_per_image,
        )
        results.append(out_rec)

    out_json_path = os.path.join(data_root, args.out_prefix + ".json")
    out_csv_path = os.path.join(data_root, args.out_prefix + ".csv")
    Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["eps_fp", "eps_tp", "tau_final",
                      "TP", "FP", "N",
                      "recall_obj", "precision", "f1", "fp_per_image"]
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in results:
            wr.writerow(r)

    print(f"\n✅ eval 结果已保存：")
    print(f"- JSON: {out_json_path}")
    print(f"- CSV : {out_csv_path}")

    taus = [r["tau_final"] for r in results]
    recall = [r["recall_obj"] for r in results]
    precision = [r["precision"] for r in results]
    f1 = [r["f1"] for r in results]
    fp_per_image = [r["fp_per_image"] for r in results]

    idx = sorted(range(len(taus)), key=lambda i: taus[i])
    taus = [taus[i] for i in idx]
    recall = [recall[i] for i in idx]
    precision = [precision[i] for i in idx]
    f1 = [f1[i] for i in idx]
    fp_per_image = [fp_per_image[i] for i in idx]

    plt.figure()
    plt.plot(taus, recall, label="recall_obj")
    plt.plot(taus, precision, label="precision")
    plt.plot(taus, f1, label="F1")
    plt.xlabel("tau_final")
    plt.ylabel("metric value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    metrics_png = os.path.join(data_root, args.out_prefix + "_metrics.png")
    plt.savefig(metrics_png, dpi=200)

    plt.figure()
    plt.plot(taus, fp_per_image, label="FP per image")
    plt.xlabel("tau_final")
    plt.ylabel("FP per image")
    plt.grid(True)
    plt.tight_layout()
    fp_png = os.path.join(data_root, args.out_prefix + "_fp.png")
    plt.savefig(fp_png, dpi=200)

    print(f"- 曲线图: {metrics_png}")
    print(f"- 曲线图: {fp_png}\n")


if __name__ == "__main__":
    main()
