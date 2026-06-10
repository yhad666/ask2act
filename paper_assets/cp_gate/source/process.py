import os, glob, csv, argparse, random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps
import numpy as np
import torch
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection

# ========= 配置 =========
SUBDIRS = ["test"]  # 修改这里来选择要处理的文件夹（可多选，如 ["bottle", "mix_n"]）
MODEL_ID = "IDEA-Research/grounding-dino-base"
PROMPT   = "cup . bottle . spoon . fork . knife . plate ."
BOX_THR, TEXT_THR, NMS_IOU = 0.20, 0.30, 0.50
MAX_PER_IMAGE = 100  
DEFAULT_DATA_ROOT = (Path(__file__).resolve().parents[1] / "calib_data").as_posix()
# CSV 处理模式: "append"=追加, "overwrite"=覆盖, "create"=仅创建不存在的文件
CSV_MODE = "append"
# 是否旋转图片（顺时针 90°）
ROTATE_IMAGE = True
# 是否覆盖原始图片文件（True=覆盖原图；False=保存到 work/<subdir> 下新建的文件夹）
OVERWRITE_IMAGES = True
# =======================

def postproc(processor, outputs, input_ids, box_th, text_th, h, w):
    # 兼容不同 transformers 版本的关键词
    for kw in (dict(threshold=box_th, text_threshold=text_th),
               dict(box_threshold=box_th, text_threshold=text_th)):
        try:
            return processor.post_process_grounded_object_detection(
                outputs=outputs, input_ids=input_ids, target_sizes=[(h, w)], **kw
            )[0]
        except TypeError:
            continue
    raise RuntimeError("post_process signature mismatch")

def iou_xyxy(a, b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    iw=max(0.0,x2-x1); ih=max(0.0,y2-y1); inter=iw*ih
    area_a=max(0.0,a[2]-a[0])*max(0.0,a[3]-a[1]); area_b=max(0.0,b[2]-b[0])*max(0.0,b[3]-b[1])
    return inter/(area_a+area_b-inter+1e-9)

def nms_per_class(boxes, scores, labels, iou_thr=0.5):
    B=np.array(boxes, float); S=np.array(scores, float); L=np.array(labels, dtype=object)
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

def draw_small(pil_img, rows, save_path):
    img = pil_img.convert("RGB"); w, h = img.size
    thickness = max(1, int(min(w, h) * 0.008))
    font_size = max(10, int(min(w, h) * 0.05))
    try: font = ImageFont.truetype("arial.ttf", font_size)
    except: font = ImageFont.load_default()
    draw = ImageDraw.Draw(img)
    for r in rows:
        x1,y1,x2,y2 = r["x1"],r["y1"],r["x2"],r["y2"]
        draw.rectangle([x1,y1,x2,y2], outline=(0,255,0), width=thickness)
        tag = f'#{r["det_id"]} {r["label"]} {r["score"]:.2f}'
        try: l,t,rr,b = draw.textbbox((0,0), tag, font=font); tw,th = rr-l, b-t
        except: tw,th = 9*len(tag), font_size+6
        pad = max(4, thickness)
        draw.rectangle([x1, y1-th-pad, x1+tw+2*pad, y1], fill=(0,255,0))
        draw.text((x1+pad, y1-th-pad//2), tag, fill=(0,0,0), font=font)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path, quality=95)

def list_images(root):
    imgs = []
    for sd in SUBDIRS:
        for ext in ("*.png","*.jpg","*.jpeg","*.bmp","*.webp"):
            imgs += glob.glob(os.path.join(root, sd, "**", ext), recursive=True)
    imgs.sort()
    return imgs

def augment_variants(img):
    out=[]
    out.append(("flip", img.transpose(Image.FLIP_LEFT_RIGHT)))
    b = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
    c = ImageEnhance.Contrast(b).enhance(random.uniform(0.85, 1.15))
    s = ImageEnhance.Color(c).enhance(random.uniform(0.85, 1.15))
    out.append(("jitter", s))
    return out

def normalize_label(lab):
    """
    - 空或全空白 -> 'unlabeled'
    - 多词 -> 仅保留第一个词，返回(新标签, 被去掉的原串)
    - 单词 -> 原样返回，第二个返回值为 None
    """
    if lab is None:
        return "unlabeled", "<None>"
    if isinstance(lab, (list, tuple)):
        lab = " ".join([str(x) for x in lab if str(x).strip()])
    lab_str = str(lab).strip()
    if not lab_str:
        return "unlabeled", "<empty>"
    parts = lab_str.replace(",", " ").split()
    if len(parts) >= 2:
        return parts[0], lab_str
    return lab_str, None

def to_numpy_safe(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT, help="project/calib_data")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--augment", action="store_true")
    args = ap.parse_args()

    work = os.path.join(args.data_root, "work")
    exp  = os.path.join(work, "exports")
    vis  = os.path.join(exp, "vis")
    csv_path = os.path.join(exp, "detections.csv")
    surgery_log = os.path.join(exp, "label_surgery.log")
    Path(vis).mkdir(parents=True, exist_ok=True)
    Path(surgery_log).parent.mkdir(parents=True, exist_ok=True)
    # 根据模式决定 CSV 文件操作
    csv_exists = os.path.exists(csv_path)
    if CSV_MODE == "overwrite":
        # 覆盖模式：直接删除旧文件
        csv_write_mode = "w"
        if csv_exists:
            os.remove(csv_path)
    elif CSV_MODE == "append":
        # 追加模式：文件存在则追加，不存在则创建
        csv_write_mode = "a" if csv_exists else "w"
    elif CSV_MODE == "create":
        # 创建模式：仅在文件不存在时创建，存在则保持现状
        if csv_exists:
            print(f"⚠️  CSV 文件已存在: {csv_path}，使用 CSV_MODE='overwrite' 来覆盖")
            csv_write_mode = None
        else:
            csv_write_mode = "w"
    else:
        csv_write_mode = "w"
    
    open(surgery_log, "w", encoding="utf-8").close()  # 清空旧日志

    img_list = list_images(args.data_root)

    # 先初始化设备与模型（供 run_one 使用）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = GroundingDinoProcessor.from_pretrained(MODEL_ID)
    model = GroundingDinoForObjectDetection.from_pretrained(MODEL_ID).to(device).eval()

    # 打开 CSV/日志，并在同一作用域内定义 run_one 与执行循环
    if csv_write_mode is None:
        # create 模式且文件已存在，跳过处理
        print(f"⏭️  跳过处理（文件已存在且模式为 create）")
        return
    
    with open(csv_path, csv_write_mode, newline="", encoding="utf-8") as f, \
         open(surgery_log, "a", encoding="utf-8") as flog:

        wr = csv.DictWriter(f, fieldnames=[
            "image","det_id","label","score","x1","y1","x2","y2","is_tp","is_fp","source"
        ])
        # 仅在新文件时写入表头
        if not csv_exists or CSV_MODE == "overwrite":
            wr.writeheader()

        def run_one(image_path, pil_img, src):
            # 推理
            w,h = pil_img.size
            inp = proc(images=pil_img, text=args.prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inp)
            r = postproc(proc, out, inp["input_ids"], BOX_THR, TEXT_THR, h, w)

            labels_raw = r.get("labels") or r.get("text_labels") or []
            boxes = r.get("boxes", [])
            scores = r.get("scores", [])
            if len(boxes) == 0:
                return

            # 保留原始标签（不做规范化）
            norm_labels = [str(lab) for lab in labels_raw]
            removed_flags = [None] * len(norm_labels)

            # Tensor → NumPy
            boxes_np  = to_numpy_safe(boxes)
            scores_np = to_numpy_safe(scores)

            # NMS（按类）
            b, s, l = nms_per_class(boxes_np, scores_np, norm_labels, NMS_IOU)

            # 截断（最高分前 MAX_PER_IMAGE）
            if len(b) > MAX_PER_IMAGE:
                order = np.argsort(-np.asarray(s))[:MAX_PER_IMAGE]
                b, s, l = b[order], np.asarray(s)[order], np.asarray(l, dtype=object)[order]

            # 写 CSV + 画图
            rows = []
            rel = os.path.relpath(image_path, args.data_root).replace("\\", "/")
            for i, (bb, sc, lab) in enumerate(zip(b, s, l)):
                x1,y1,x2,y2 = [float(v) for v in bb.tolist()]
                row = dict(image=rel, det_id=int(i), label=str(lab), score=float(sc),
                           x1=x1, y1=y1, x2=x2, y2=y2, is_tp="", is_fp="", source=src)
                wr.writerow(row); rows.append(row)

            # 记录标签改写日志（逐条）
            for i, lab in enumerate(norm_labels):
                removed = removed_flags[i]
                if (removed is not None) or (lab == "unlabeled"):
                    flog.write(f"{rel} :: det? -> '{removed if removed else 'unlabeled'}' => '{lab}'\n")

            draw_small(pil_img, rows, os.path.join(vis, rel))

        # === 主循环：读图→纠正EXIF→条件旋转→条件覆盖→检测（在同一作用域中可调用 run_one）
        for p in img_list:
            try:
                img = Image.open(p).convert("RGB")
                img = ImageOps.exif_transpose(img)      # 处理 EXIF 方向
                # 条件旋转：根据 ROTATE_IMAGE 配置决定是否旋转
                if ROTATE_IMAGE:
                    img = img.transpose(Image.ROTATE_270)   # 顺时针 90°
                
                # 处理保存位置：覆盖原图 或 保存到 work/<subdir>
                if OVERWRITE_IMAGES:
                    img.save(p)  # 覆盖原图
                else:
                    # 保存处理后的图片到 work/<subdir> 中，保持相同的目录结构
                    rel_path = os.path.relpath(p, args.data_root)
                    parts = rel_path.split(os.sep)
                    if len(parts) > 0:
                        subdir = parts[0]
                        rest_path = os.sep.join(parts[1:])
                        output_path = os.path.join(work, subdir, rest_path)
                        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                        img.save(output_path)

                run_one(p, img, "orig")

                if args.augment:
                    for tag, aug in augment_variants(img):
                        run_one(p, aug, tag)
            except Exception as e:
                print("[detect failed]", p, e)

    print(f"\n✅ 完成：\n- CSV: {csv_path}\n- 可视化: {vis}/<同层级>.jpg\n- 标签改写日志: {surgery_log}")

if __name__ == "__main__":
    main()
