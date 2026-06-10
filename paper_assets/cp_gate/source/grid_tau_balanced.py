# grid_tau_balanced.py
import os, csv, json, argparse, datetime
import numpy as np
from pathlib import Path

# ★★ 按你的工程改这里：calib 数据根目录
DATA_ROOT = (Path(__file__).resolve().parents[1] / "calib_data").as_posix()

# calib 标注后的 CSV（含 is_fp），相对 DATA_ROOT 的路径
CALIB_CSV = "work/annotations/detections_labeled.csv"

# 扫 eps 的范围（你在这里改就行）
EPS_FP_LIST = [x / 100.0 for x in range(1, 30)]  # 0.01 ~ 0.15
EPS_TP_LIST = [x / 100.0 for x in range(1, 30)]  # 0.01 ~ 0.15

OUT_JSON = "work/tau/tau_grid.json"
OUT_CSV  = "work/tau/tau_grid.csv"


def finite_quantile(arr, q):
    """安全分位数：arr 为空时返回 None"""
    a = np.asarray(arr, float)
    if a.size == 0:
        return None
    a.sort()
    k = int(np.ceil(q * a.size)) - 1
    k = max(0, min(k, a.size - 1))
    return float(a[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DATA_ROOT)
    ap.add_argument("--csv", default=CALIB_CSV)
    args = ap.parse_args()

    data_root = args.data_root
    csv_path = os.path.join(data_root, args.csv)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"calib CSV 找不到: {csv_path}")

    fp_scores, tp_scores = [], []

    with open(csv_path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            try:
                s = float(r["score"])
            except Exception:
                continue

            if r.get("is_fp", "").strip() == "1":
                fp_scores.append(s)
            else:
                tp_scores.append(s)

    if not fp_scores:
        raise ValueError("❌ 没有 FP 样本（is_fp=1）。至少要标一些 FP 才能算 τ_fp。")

    fp_scores = np.asarray(fp_scores, float)
    tp_scores = np.asarray(tp_scores, float) if tp_scores else np.zeros((0,), float)

    print("📐 eps_fp 网格:", EPS_FP_LIST)
    print("📐 eps_tp 网格:", EPS_TP_LIST)

    records = []
    for eps_fp in EPS_FP_LIST:
        for eps_tp in EPS_TP_LIST:
            tau_fp = finite_quantile(fp_scores, 1 - eps_fp)
            tau_tp = finite_quantile(tp_scores, eps_tp) if tp_scores.size > 0 else None
            tau_final = max(tau_fp, tau_tp) if tau_tp is not None else tau_fp

            rec = dict(
                eps_fp=float(eps_fp),
                eps_tp=float(eps_tp),
                tau_fp=tau_fp,
                tau_tp=tau_tp,
                tau_final=tau_final,
            )
            records.append(rec)

    out_json_path = os.path.join(data_root, OUT_JSON)
    Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "date": datetime.datetime.now().isoformat(timespec="seconds"),
                "csv": args.csv,
                "records": records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    out_csv_path = os.path.join(data_root, OUT_CSV)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["eps_fp", "eps_tp", "tau_fp", "tau_tp", "tau_final"]
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in records:
            wr.writerow(r)

    print("\n✅ tau 网格已生成：")
    print(f"- JSON: {out_json_path}")
    print(f"- CSV : {out_csv_path}")
    print(f"共 {len(records)} 对 (eps_fp, eps_tp)\n")


if __name__ == "__main__":
    main()
