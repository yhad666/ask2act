# compute_tau_balanced.py
import os, csv, json, argparse, datetime, numpy as np
from pathlib import Path

DEFAULT_DATA_ROOT = (Path(__file__).resolve().parents[1] / "calib_data").as_posix()


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
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--csv", default="work/annotations/detections_labeled.csv",
                    help="标注后的 CSV（含 is_fp）")
    ap.add_argument("--eps_fp", type=float, default=0.02,
                    help="FP 端误差率 ε_FP（默认 0.02 → 98%）")
    ap.add_argument("--eps_tp", type=float, default=0.02,
                    help="TP 端误差率 ε_TP（默认 0.02 → 98%）")
    ap.add_argument("--out", default="work/tau/tau_balanced.json")
    args = ap.parse_args()

    eps_fp = args.eps_fp
    eps_tp = args.eps_tp

    csv_path = os.path.join(args.data_root, args.csv)
    fp_scores = []
    tp_scores = []

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 找不到: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)

        for r in rd:
            try:
                s = float(r["score"])
            except:
                continue

            # 🌟 规则：
            #  - is_fp == 1 → FP
            #  - 其它全部视为 TP（包括 is_tp 空）
            if r.get("is_fp", "").strip() == "1":
                fp_scores.append(s)
            else:
                tp_scores.append(s)

    if not fp_scores:
        raise ValueError("❌ 没有 FP 样本（is_fp=1）。至少要标些 FP 才能算阈值。")

    # τ_FP: FP 的分位（1 - eps_fp）
    tau_fp = finite_quantile(fp_scores, 1 - eps_fp)

    # τ_TP: TP 的分位（eps_tp）
    tau_tp = finite_quantile(tp_scores, eps_tp) if tp_scores else None

    # 最终阈值：取更严格的那个
    tau_final = max(tau_fp, tau_tp) if tau_tp is not None else tau_fp

    out_path = os.path.join(args.data_root, args.out)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    json.dump(
        {
            "tau_final": tau_final,
            "tau_fp": tau_fp,
            "tau_tp": tau_tp,
            "eps_fp": eps_fp,
            "eps_tp": eps_tp,
            "note": "未标 is_fp=1 的全部视为 TP；tau_final = max(tau_fp, tau_tp)",
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        open(out_path, "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )

    print("\n====================== 结果 ======================")
    print(f"FP 误差率 eps_fp = {eps_fp:.4f} → τ_FP = {tau_fp:.4f}")
    if tau_tp is None:
        print("TP 样本为空 → τ_TP = None（阈值由 FP 决定）")
    else:
        print(f"TP 误差率 eps_tp = {eps_tp:.4f} → τ_TP = {tau_tp:.4f}")
    print(f"➡️  最终 τ_final = {tau_final:.4f}")
    print(f"已保存: {out_path}")
    print("=================================================\n")


if __name__ == "__main__":
    main()
