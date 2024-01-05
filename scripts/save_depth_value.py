import cv2
import torch
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) 
print(sys.path)
import numpy as np
from pathlib import Path
from digit_depth.third_party import geom_utils
from digit_depth.train.prepost_mlp import *
from digit_depth.handlers import find_recent_model
import torch

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
torch.seed = seed

# 设置基本路径
base_path = Path(__file__).parent.parent.resolve()

def process_image(frame, model, dm_zero):
    # 对图像进行预处理
    img_np = preproc_mlp(frame)
    img_np = model(img_np).detach().cpu().numpy()
    img_np, _ = post_proc_mlp(img_np)

    # 获取梯度图
    gradx_img, grady_img = geom_utils._normal_to_grad_depth(img_normal=img_np, gel_width=0.01835,
                                                            gel_height=0.02460, bg_mask=None)

    # 重建深度图
    img_depth = geom_utils._integrate_grad_depth(gradx_img, grady_img, boundary=None, bg_mask=None,
                                                 max_depth=0.02076)
    img_depth = img_depth.detach().cpu().numpy()

    # 减去零深度
    diff = img_depth - dm_zero

    # 转换像素值到0-255范围
    diff = diff * 255
    diff = diff * -1

    # 阈值处理
    ret, thresh4 = cv2.threshold(diff, 0, 255, cv2.THRESH_TOZERO)

    # 如果需要椭圆可视化
    if cfg.visualize.ellipse:
        img = thresh4
        pt = ContactArea()
        theta = pt.__call__(target=thresh4)
        #进行可视化
        cv2.imshow("Visualized Depth", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("visualized_depth.png", img)
    else:
        img = thresh4

    return img

def main():
    # 加载模型
    model_path = find_recent_model(f"{base_path}/models")
    model = torch.load(model_path).to(device)
    model.eval()

    # 初始化零深度计算
    dm_zero_counter = 0
    dm_zero = 0

    # 图像文件夹路径
    image_folder_path = base_path / "datasets"

    # 读取所有图像
    image_paths = list(image_folder_path.glob("*.png"))

    for image_path in image_paths:
        # 读取图像
        frame = cv2.imread(str(image_path))

        # 处理每张图像
        img_depth = process_image(frame, model, dm_zero)

        # 累计零深度
        if dm_zero_counter < 50:
            dm_zero += img_depth
            dm_zero_counter += 1
        elif dm_zero_counter == 50:
            dm_zero = dm_zero / 50
            dm_zero_counter += 1

        # 将深度保存为CSV文件在/digit-depth/datasets中
        csv_filename = image_path.stem + "_depthvalue.csv"
        csv_path = image_folder_path / csv_filename
        np.savetxt(csv_path, img_depth, delimiter=",")

    print("深度处理完成，已保存为CSV文件。")

if __name__ == "__main__":
    main()
