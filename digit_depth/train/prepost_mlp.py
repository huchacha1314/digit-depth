# 图像的预处理 和 数据的后处理
import cv2
import torch
import numpy as np
import pandas as pd
import copy
seed = 42
torch.seed = seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preproc_mlp(image) -> torch.Tensor:
    """ Preprocess image for input to model.

    Args: image: OpenCV image in BGR format
    Return: tensor of shape (R*C,5) where R=320 and C=240 for DIGIT images
    5-columns are: X,Y,R,G,B

    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #将BGR 转 RGB
    img = img / 255.0
    #归一化
    xy_coords = np.flip(np.column_stack(np.where(np.all(img >= 0, axis=2))), axis=1)
    #output：（n,[x,y])
    '''
    np.all(img >= 0, axis=2)) 输出布尔值，延3通道方向进行验证，只要三通道中出现负数，判定为False，采集有效rgb数据的x，y坐标值（无效的rgb值小于0）
    np.where(np.all(img >= 0, axis=2)) 输出坐标
    np.column_stack 堆叠
    np.flip 延axis=1 进行翻转
    '''
    rgb = np.reshape(img, (np.prod(img.shape[:2]), 3))
    #output：(n,[r,g,b])
    '''
    img.shape[:2] 返回 x y
    np.prod(img.shape[:2]) x*y 返回 像素值
    np.reshape(img, (np.prod(img.shape[:2]), 3)) 每个像素点，都对应3个r，g ，b值
    '''
    pixel_numbers = np.expand_dims(np.arange(1, xy_coords.shape[0] + 1), axis=1)
    #output： （n，1）标号
    '''
    np.arange(1, xy_coords.shape[0] + 1) 生成一个从1 开始 数量为像素数的 一维向量 n
    扩充一维向量的列 （n，1）
    
    '''
    value_base = np.hstack([pixel_numbers, xy_coords, rgb])
    '''
    第一行：标号1，[x,y],[r,g,b]
    ...
    第N行：标号N，[x,y],[r,g,b]
    '''
    df_base = pd.DataFrame(value_base, columns=['pixel_number', 'X', 'Y', 'R', 'G', 'B'])
    df_base['X'] = df_base['X'] / 240
    df_base['Y'] = df_base['Y'] / 320
    del df_base['pixel_number']
    test_tensor = torch.tensor(df_base[['X', 'Y', 'R', 'G', 'B']].values, dtype=torch.float32).to(device)
    return test_tensor


def post_proc_mlp(model_output: torch.Tensor):
    """ Postprocess model output to get normal map.

    Args: model_output: torch.Tensor of shape (1,3)
    Return: two torch.Tensor of shape (1,3)

    """
    test_np = model_output.reshape(320, 240, 3)
    normal = copy.deepcopy(test_np)  # surface normal image
    test_np = torch.tensor(test_np,
                           dtype=torch.float32)  # convert to torch tensor for later processing in gradient computation
    test_np = test_np.permute(2, 0, 1)  # swap axes to (3,320,240)
    test_np = test_np # convert to uint8 for visualization
    return test_np, normal


