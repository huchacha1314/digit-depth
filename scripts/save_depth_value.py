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

#返回50张背景图的平均值
def generate_dm_zero():
    model_path = find_recent_model(f"{base_path}/models")
    model = torch.load(model_path).to(device)
    
    background_folder = '/pathfolder' #50张背景图的文件夹
    dm_zero_counter = 0
    dm_zero = 0
    for filename in sorted(os.listdir(background_folder))[:50]:
        if filename.endswith(".png"):
            background_path = os.path.join(background_folder, filename)
            background = cv2.imread(background_path)
            background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if needed
            
            img_np = preproc_mlp(background)
            img_np = model(img_np).detach().cpu().numpy()
            img_np, _ = post_proc_mlp(img_np)
            gradx_img, grady_img = geom_utils._normal_to_grad_depth(img_normal=img_np,
                                                                    gel_width=cfg.sensor.gel_width,
                                                                    gel_height=cfg.sensor.gel_height, bg_mask=None)
            img_depth = geom_utils._integrate_grad_depth(gradx_img, grady_img,
                                                        boundary=None, bg_mask=None, max_depth=cfg.max_depth)
            img_depth = img_depth.detach().cpu().numpy()

            dm_zero += img_depth
            dm_zero_counter += 1

    dm_zero /= 50
    return dm_zero

def show_E_area():
    # 对图像进行预处理
    dm_zero = generate_dm_zero()
    model_path = find_recent_model(f"{base_path}/models")
    model = torch.load(model_path).to(device)
    frame_folder = '/pathfolder' #带有force value的png文件夹
    for filename in sorted(os.listdir(frames_folder))[:50]:# 记得更改采集图片的数量
        if filename.endswith(".png"):
            frame_path = os.path.join(frames_folder, filename)
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if needed
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
               
                theta, major_axis, major_axis_end, minor_axis, minor_axis_end = pt.__call__(target=thresh4)
                a = np.linalg.norm(np.array(major_axis) - np.array(major_axis_end)) / 2.0
                b = np.linalg.norm(np.array(minor_axis) - np.array(minor_axis_end)) / 2.0
                ellipse_area = np.pi * a * b
    
        
            return ellipse_area





if __name__ == "__main__":
    area=show_E_area()
    print(area)
