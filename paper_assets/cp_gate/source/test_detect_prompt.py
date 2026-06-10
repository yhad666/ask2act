# test_detect_prompt.py
import os, glob, csv, argparse
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import torch
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection

# ========== 你要改的配置 ==========
DATA_ROOT = (Path(__file__).resolve().parents[1] / "calib_data").as_posix()

# test 图片所在子目录（相对 DATA_ROOT）
SUBDIRS = ["test"]

# 固定使用的 prompt（你只在这里改）
PROMPT = "cup . bottle . spoon . fork . knife . plate ."

# 顺时针 90° 旋转开关：True/False
ROTATE_IMAGE = False

# 输出 CSV（相对 DATA_ROOT）
OUT_CSV = "work/exports/test_single_raw.csv"
# =================================

MODEL_ID = "IDEA-Research/grounding-dino-base"

# 这里的 BOX_THR 是“baseline box threshold”（不是 CP 的 τ）
# calib 和 test 上最好用同一个值，让 score 分布一致。
BOX_THR = 0.10
TEXT_THR = 0.30
NMS_IOU = 0.50
MAX_PER_IMAGE = 100


def postproc(processor, outputs, input_ids, h, w):
    """DINO 后处理，使用 BOX_THR / TEXT_THR"""
    for kw in (dict(threshold=BOX_THR, text_threshold=TEXT_THR),
               dict(box_threshold=BOX_THR, text_threshold=TEXT_THR)):
        try:
            return processor.post_process_grounded_object_detection(
                outputs=outputs, input_ids=input_ids, target_sizes=[(h, w)], **kw
            )[0]
        except TypeError:
            continue
    raise RuntimeError("post_process_grounded_object_detection signature mismatch")


def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def nms_per_class(boxes, scores, labels, iou_thr=0.5):
    B = np.array(boxes, float)
    S = np.array(scores, float)
    L = np.array(labels, dtype=object)
    keep_b, keep_s, keep_l = [], [], []
    for cls in np.unique(L):
        idx = np.where(L == cls)[0]
        b, s = B[idx], S[idx]
        order = np.argsort(-s)
        while len(order) > 0:
            i = order[0]
            keep_b.append(b[i]); keep_s.append(s[i]); keep_l.append(cls)
            rest = order[1:]
            order = np.array([j for j in rest if iou_xyxy(b[i], b[j]) <= iou_thr])
    return np.array(keep_b), np.array(keep_s), np.array(keep_l)


def to_numpy_safe(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def list_images(root):
    imgs = []
    for sd in SUBDIRS:
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
            imgs += glob.glob(os.path.join(root, sd, "**", ext), recursive=True)
    imgs.sort()
    return imgs


def draw_small(pil_img, rows, save_path):
    img = pil_img.convert("RGB")
    w, h = img.size
    thickness = max(1, int(min(w, h) * 0.008))
    font_size = max(10, int(min(w, h) * 0.05))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(img)
    for r in rows:
        x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=thickness)
        tag = f'#{r["det_id"]} {r["label"]} {r["score"]:.2f}'
        try:
            l, t, rr, b = draw.textbbox((0, 0), tag, font=font); tw, th = rr - l, b - t
        except Exception:
            tw, th = 9 * len(tag), font_size + 6
        pad = max(4, thickness)
        draw.rectangle([x1, y1 - th - pad, x1 + tw + 2 * pad, y1], fill=(0, 255, 0))
        draw.text((x1 + pad, y1 - th - pad // 2), tag, fill=(0, 0, 0), font=font)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path, quality=95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DATA_ROOT)
    args = ap.parse_args()

    data_root = args.data_root
    img_list = list_images(data_root)
    if not img_list:
        print("⚠️ 没找到 test 图片，请检查 DATA_ROOT 和 SUBDIRS。")
        return

    work = os.path.join(data_root, "work")
    exp = os.path.join(work, "exports")
    vis = os.path.join(exp, "vis")
    Path(vis).mkdir(parents=True, exist_ok=True)

    csv_path = os.path.join(data_root, OUT_CSV)
    Path(os.path.dirname(csv_path)).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("PROMPT:", PROMPT)
    print("ROTATE_IMAGE:", ROTATE_IMAGE)

    proc = GroundingDinoProcessor.from_pretrained(MODEL_ID)
    model = GroundingDinoForObjectDetection.from_pretrained(MODEL_ID).to(device).eval()

    fieldnames = [
        "image", "det_id", "label", "score", "x1", "y1", "x2", "y2",
        "is_tp", "is_fp", "gt_total", "source"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()

        for p in img_list:
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                print("[open failed]", p, e)
                continue

            # 先按 EXIF 修正方向
            img = ImageOps.exif_transpose(img)

            # ★★ 按开关顺时针 90° 旋转
            if ROTATE_IMAGE:
                img = img.transpose(Image.ROTATE_270)  # 顺时针 90°

            w, h = img.size

            inp = proc(images=img, text=PROMPT, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inp)
            r = postproc(proc, out, inp["input_ids"], h, w)

            labels_raw = r.get("labels") or r.get("text_labels") or []
            boxes = r.get("boxes", [])
            scores = r.get("scores", [])
            if len(boxes) == 0:
                continue

            labels = [str(l) for l in labels_raw]
            boxes_np = to_numpy_safe(boxes)
            scores_np = to_numpy_safe(scores)

            b, s, l = nms_per_class(boxes_np, scores_np, labels, NMS_IOU)

            if len(b) > MAX_PER_IMAGE:
                order = np.argsort(-np.asarray(s))[:MAX_PER_IMAGE]
                b, s, l = b[order], np.asarray(s)[order], np.asarray(l, dtype=object)[order]

            rel = os.path.relpath(p, data_root).replace("\\", "/")
            rows = []
            for i, (bb, sc, lab) in enumerate(zip(b, s, l)):
                x1, y1, x2, y2 = [float(v) for v in bb.tolist()]
                row = dict(
                    image=rel,
                    det_id=int(i),
                    label=str(lab),
                    score=float(sc),
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    is_tp="", is_fp="", gt_total="",
                    source="test_single",
                )
                wr.writerow(row)
                rows.append(dict(
                    det_id=int(i), label=str(lab), score=float(sc),
                    x1=x1, y1=y1, x2=x2, y2=y2
                ))

            draw_small(img, rows, os.path.join(vis, rel))

    print(f"\n✅ test detect 完成：")
    print(f"- CSV: {csv_path}")
    print(f"- 可视化: {vis}/<同层级>.jpg\n")


if __name__ == "__main__":
    main()
