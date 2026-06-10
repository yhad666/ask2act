# infer_with_tau.py
import os, glob, json, argparse, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection

MODEL_ID = "IDEA-Research/grounding-dino-base"
BOX_THR, TEXT_THR, NMS_IOU = 0.20, 0.30, 0.50
DEFAULT_DATA_ROOT = (Path(__file__).resolve().parents[1] / "calib_data").as_posix()


def postproc(proc, outputs, input_ids, h, w):
    """兼容不同 transformers 版本的 DINO 后处理"""
    for kw in (dict(threshold=BOX_THR, text_threshold=TEXT_THR),
               dict(box_threshold=BOX_THR, text_threshold=TEXT_THR)):
        try:
            return proc.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=input_ids,
                target_sizes=[(h, w)],
                **kw
            )[0]
        except TypeError:
            continue
    raise RuntimeError("post_process_grounded_object_detection signature mismatch")


def to_numpy_safe(x):
    """Tensor 在 GPU 上时安全转成 numpy"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def nms_per_class(boxes, scores, labels, iou_thr=0.5):
    """按类别做 NMS"""
    B = np.array(boxes, float)
    S = np.array(scores, float)
    L = np.array(labels, dtype=object)

    keep_b, keep_s, keep_l = [], [], []

    def iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
        inter = iw * ih
        area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
        area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
        return inter / (area_a + area_b - inter + 1e-9)

    for cls in np.unique(L):
        idx = np.where(L == cls)[0]
        b, s = B[idx], S[idx]
        order = np.argsort(-s)
        while len(order) > 0:
            i = order[0]
            keep_b.append(b[i])
            keep_s.append(s[i])
            keep_l.append(cls)
            rest = order[1:]
            order = np.array([j for j in rest if iou(b[i], b[j]) <= iou_thr])

    return np.array(keep_b), np.array(keep_s), np.array(keep_l)


def draw(img, boxes, scores, labels, save_path):
    """在图片上画框 + 标签 + 分数"""
    img = img.convert("RGB")
    w, h = img.size
    thickness = max(3, int(min(w, h) * 0.02))
    font_size = max(18, int(min(w, h) * 0.10))

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    d = ImageDraw.Draw(img)
    for bb, sc, lab in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [float(v) for v in bb]
        # clamp 防越界
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        d.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=thickness)
        tag = f"{lab} {float(sc):.2f}"
        try:
            l, t, r, b = d.textbbox((0, 0), tag, font=font)
            tw, tht = r - l, b - t
        except Exception:
            tw, tht = 9 * len(tag), font_size + 6
        pad = max(4, thickness)
        d.rectangle([x1, y1 - tht - pad, x1 + tw + 2 * pad, y1], fill=(0, 255, 0))
        d.text((x1 + pad, y1 - tht - pad // 2), tag, fill=(0, 0, 0), font=font)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path, quality=95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT,
                    help="一般是 project/calib_data")
    ap.add_argument("--img_dir", default=None,
                    help="要推理的图片根目录，默认= data_root")
    ap.add_argument("--prompt", default="cup . bottle . spoon . fork . knife . plate .")
    ap.add_argument("--tau", default="work/tau/tau_balanced.json",
                    help="compute_tau_balanced 输出的 JSON，相对 data_root")
    ap.add_argument("--no_exif_fix", action="store_true",
                    help="关闭 EXIF 方向修正")
    args = ap.parse_args()

    # 1) 读取 τ
    tau_path = os.path.join(args.data_root, args.tau)
    with open(tau_path, "r", encoding="utf-8") as f:
        tau = float(json.load(f)["tau_final"])
    print(f"✅ 使用 τ_final = {tau:.4f} 进行过滤")

    # 2) 输入/输出路径
    img_root = args.img_dir or args.data_root
    out_dir = os.path.join(args.data_root, "work/infer_vis")

    # 3) 初始化 DINO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    proc = GroundingDinoProcessor.from_pretrained(MODEL_ID)
    model = GroundingDinoForObjectDetection.from_pretrained(MODEL_ID).to(device).eval()

    # 4) 收集图片
    imgs = []
    for e in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        imgs += glob.glob(os.path.join(img_root, "**", e), recursive=True)
    imgs.sort()
    if not imgs:
        print("⚠️ 没有找到任何图片，请检查 --img_dir 或 data_root。")
        return

    # 5) 逐张推理 + 过滤 + 画图
    for p in imgs:
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print("[open failed]", p, e)
            continue

        if not args.no_exif_fix:
            img = ImageOps.exif_transpose(img)

        w, h = img.size
        inp = proc(images=img, text=args.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inp)

        r = postproc(proc, out, inp["input_ids"], h, w)

        labels_raw = r.get("text_labels") or r.get("labels") or []
        boxes = r.get("boxes", [])
        scores = r.get("scores", [])

        if len(boxes) == 0:
            continue

        boxes_np = to_numpy_safe(boxes)
        scores_np = to_numpy_safe(scores)
        labels = [str(l) for l in labels_raw]

        # NMS
        b, s, l = nms_per_class(boxes_np, scores_np, labels, NMS_IOU)

        # 应用 τ
        keep = np.where(s >= tau)[0]
        b, s, l = b[keep], s[keep], np.asarray(l, dtype=object)[keep]
        if len(b) == 0:
            continue

        rel = os.path.relpath(p, img_root).replace("\\", "/")
        save_path = os.path.join(out_dir, rel)
        draw(img, b, s, l, save_path)

    print(f"\n✅ 推理 + τ 过滤 + 可视化完成：{out_dir}/")


if __name__ == "__main__":
    main()
