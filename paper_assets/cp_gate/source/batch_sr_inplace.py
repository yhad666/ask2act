import os
from PIL import Image
from RealESRGAN import RealESRGAN
import torch


def super_resolve_inplace(root_dir, scale=4, exts=(".png", ".jpg", ".jpeg", ".bmp")):
    """
    对 root_dir 内所有图片进行 4 倍超分，原地覆盖保存。
    目录结构不变，文件名不变。
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 初始化模型
    model = RealESRGAN(device, scale=scale)

    # 第一次会自动从 HF 下载权重到 ./weights/RealESRGAN_x4.pth
    weights_path = f"weights/RealESRGAN_x{scale}.pth"
    model.load_weights(weights_path, download=True)
    print(f"[INFO] RealESRGAN x{scale} weights loaded from: {weights_path}")

    count = 0

    for root, dirs, files in os.walk(root_dir):
        for name in files:
            if not name.lower().endswith(exts):
                continue

            img_path = os.path.join(root, name)

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[WARN] Cannot open {img_path}: {e}")
                continue

            try:
                sr_img = model.predict(img)
            except Exception as e:
                print(f"[WARN] Failed to super-resolve {img_path}: {e}")
                continue

            sr_img.save(img_path)
            count += 1
            print(f"[OK] {count}: {img_path}")

    print(f"[DONE] Total processed images: {count}")


if __name__ == "__main__":
    ROOT_DIR = r"/home/haoandong/workspace/project/calib_data"

    super_resolve_inplace(
        root_dir=ROOT_DIR,
        scale=4,
    )
