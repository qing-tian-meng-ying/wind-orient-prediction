import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import math
import dateutil
import matplotlib.dates as md
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
from sklearn.metrics import r2_score  # R square
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import matplotx
from torch.autograd import Variable
import os


# 将风速分量转换回角度
def back_angle(x, y):
    angle = []
    for i in range(len(x)):
        if x[i] >= 0 and y[i] > 0:
            angle.append(np.arctan(x[i] / y[i]) / np.pi * 180)
        elif x[i] <= 0 and y[i] > 0:
            angle.append(360 + np.arctan(x[i] / y[i]) / np.pi * 180)
        elif x[i] <= 0 and y[i] < 0:
            angle.append(180 + np.arctan(x[i] / y[i]) / np.pi * 180)
        elif x[i] >= 0 and y[i] < 0:
            angle.append(180 + np.arctan(x[i] / y[i]) / np.pi * 180)
        elif y[i] == 0 and x[i] < 0:
            angle.append(90)
        elif y[i] == 0 and x[i] > 0:
            angle.append(270)
        elif y[i] == 0 and x[i] == 0:
            angle.append(0)
    return angle


# 修正代码
def dispatch(angle: list):
    patch = [0]
    a = angle[0]
    for i in range(len(angle[1:])):
        j = 0
        if a > angle[i + 1]:
            while abs(a - angle[i + 1] - 360 * j) > 180:
                j += 1
            patch.append(j)
            angle[i + 1] = angle[i + 1] + 360 * j
        else:
            while abs(a - angle[i + 1] - 360 * j) > 180:
                j -= 1
            patch.append(j)
            angle[i + 1] = angle[i + 1] + 360 * j
        a = angle[i + 1]
    add = [i * 360 for i in patch]
    return np.array(angle), np.array(patch)


def mape(y_true, y_pred):
    """计算 MAPE 指标"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# 计算MSE、MAE、R2
def count_error(y_true, y_pred, choose_data):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    choose_data = np.array(choose_data)
    length = 0
    totol_mse = mean_squared_error(y_true[length::] * (max(choose_data) - min(choose_data)) + min(choose_data),
                                   y_pred[length::] * (max(choose_data) - min(choose_data)) + min(choose_data))
    totol_mse = '%.4g' % totol_mse
    totol_mae = mean_absolute_error(y_true[length::] * (max(choose_data) - min(choose_data)) + min(choose_data),
                                    y_pred[length::] * (max(choose_data) - min(choose_data)) + min(choose_data))
    totol_mae = '%.4g' % totol_mae
    # # 调用
    R2 = r2_score(y_true[length::] * (max(choose_data) - min(choose_data)) + min(choose_data),
                  y_pred[length::] * (max(choose_data) - min(choose_data)) + min(choose_data))
    R2 = '%.4g' % R2
    return totol_mse, totol_mae, R2


# 重新修正方向偏移
def fix_dir(dir_true, dir_pred):
    for i in range(len(dir_true)):
        while dir_true[i] - dir_pred[i] > 180:
            dir_pred[i] += 360
        while dir_true[i] - dir_pred[i] < -180:
            dir_pred[i] -= 360
    return dir_pred


# 角度计算误差函数
def mean_angle_error(y_true, y_pred):
    cosine_sim = cosine_similarity(y_true, y_pred)
    dist_mat = np.diag(cosine_sim)
    angle_error = np.rad2deg(np.arccos(dist_mat))
    return angle_error


def mean_angular_error(y_true, y_pred):
    """计算角度误差（MAE）"""
    # 将预测值和真实值转化为复平面上的向量
    y_true_cos = np.cos(np.deg2rad(y_true))
    y_true_sin = np.sin(np.deg2rad(y_true))
    y_pred_cos = np.cos(np.deg2rad(y_pred))
    y_pred_sin = np.sin(np.deg2rad(y_pred))
    # 计算向量之间的余弦距离（夹角越小，余弦距离越接近于0）
    dist = cosine_similarity(np.stack([y_true_cos, y_true_sin], axis=-1), np.stack([y_pred_cos, y_pred_sin], axis=-1))
    # 将余弦距离转化为角度误差
    angular_error = np.rad2deg(np.arccos(np.diag(dist)))
    # 计算平均角度误差
    mae = np.mean(angular_error)
    return mae


# 去除0元素
def remove_zero(y_true, y_pred):
    # 找到 y_true 中不为 0 的元素的下标
    nonzero_idx = np.nonzero(y_true)[0]

    # 去除 y_true 中的 0 元素
    y_true_nonzero = y_true[nonzero_idx]

    # 去除对应的 y_pred 中的位置数据
    y_pred_nonzero = y_pred[nonzero_idx]
    return y_true_nonzero, y_pred_nonzero


# 路径文件
path = 'lstm_12step_MSE2048_1024'

loadData_t = np.load(r'.\result'
                     r's\{}\true.npy'.format(path)).squeeze()
loadData_p = np.load(r'.\result'
                     r's\{}\pred.npy'.format(path)).squeeze()

# 检查目标文件夹是否存在，如果不存在则创建
folder_path = os.path.join("error", path)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 获取数组的维度数
num_dims = len(loadData_t.shape)

# 根据维度数判断获取第二维的大小，如果没有第三维则将其设置为1
n = loadData_t.shape[1] if num_dims >= 2 else 1

# file_path = os.path.join(folder_path, "step{}".format(n))
# if not os.path.exists(file_path):
#     os.makedirs(file_path)

# 初始化存储结果的数组
result_array = np.zeros((n, 5, 5, 17))
wind_data = []
orient_data = []

for j in range(n):
    y_true = loadData_t[:, j, :]
    y_pred = loadData_p[:, j, :]
    wind = []
    orient = []
    for i in range(17):
        x = y_true[:, i * 2]
        y = y_pred[:, i * 2]
        u = y_true[:, i * 2 + 1]
        v = y_pred[:, i * 2 + 1]
        # 去除0元素方便计算MAPE
        x_nonzero, y_nonzero = remove_zero(x, y)
        u_nonzero, v_nonzero = remove_zero(u, v)
        totol_mse_x, totol_mae_x, R2_x = count_error(x, y, [0, 1])
        totol_rmse_x = '%.4g' % np.sqrt(float(totol_mse_x))
        totol_mape_x = '%.4g' % mape(x_nonzero, y_nonzero)

        # y方向的分量
        totol_mse_y, totol_mae_y, R2_y = count_error(u, v, [0, 1])
        totol_rmse_y = '%.4g' % np.sqrt(float(totol_mse_y))
        totol_mape_y = '%.4g' % mape(u_nonzero, v_nonzero)

        # x和y的合方向
        y_ori = back_angle(np.array(x), np.array(u))
        # y_ori = np.array(df_dir.iloc[len_train + 0:-1, i])
        y_ori_pred = back_angle(np.array(y), np.array(v))
        # y_ori_pred = back_angle(np.array(y_data[len_train + 4:, i]), np.array(y_data[len_train + 4:, i + 17]))
        totol_mse_o = '%.4g' % mean_angular_error(y_ori, y_ori_pred)
        totol_rmse_o = '%.4g' % np.sqrt(mean_squared_error(y_ori, y_ori_pred))
        angle_error = mean_angle_error(np.stack([x, u], axis=-1), np.stack([y, v], axis=-1))
        orient.append(angle_error)
        totol_mae_o = '%.4g' % np.mean(angle_error)
        # 去除0元素方便计算MAPE
        dir_true_nonzero, dir_pred_nonzero = remove_zero(np.array(y_ori), np.array(y_ori_pred))
        totol_mape_o = '%.4g' % mape(dir_true_nonzero, dir_pred_nonzero)
        R2_o = '%.4g' % r2_score(y_ori, y_ori_pred)

        # 解卷绕
        un_w_y_true = dispatch(y_ori)
        un_w_y_pred = dispatch(y_ori_pred)
        un_w_y_pred = fix_dir(un_w_y_true[0], un_w_y_pred[0])
        totol_mse_e, totol_mae_e, R2_e = count_error(un_w_y_true[0], un_w_y_pred, [0, 1])
        totol_rmse_e = '%.4g' % np.sqrt(float(totol_mse_e))
        # 去除0元素方便计算MAPE
        un_w_y_true_nonzero, un_w_y_pred_nonzero = remove_zero(un_w_y_true[0], un_w_y_pred)
        totol_mape_e = '%.4g' % mape(un_w_y_true_nonzero, un_w_y_pred_nonzero)

        # x和y的合速度
        y_sym = np.sqrt(((np.array(x) ** 2) + (np.array(u) ** 2)))
        # y_sym = np.array(df_wind.iloc[len_train + 0:-1, i])
        y_sym_pred = np.sqrt(((np.array(y) ** 2) + (np.array(v) ** 2)))
        wind.append(np.array([y_sym, y_sym_pred]))
        totol_mse_sym, totol_mae_sym, R2_sym = count_error(y_sym, y_sym_pred, [0, 1])
        totol_rmse_sym = '%.4g' % np.sqrt(float(totol_mse_sym))
        # 去除0元素方便计算MAPE
        wind_true_nonzero, wind_pred_nonzero = remove_zero(y_sym, y_sym_pred)
        totol_mape_sym = '%.4g' % mape(wind_true_nonzero, wind_pred_nonzero)

        # 将结果存入数组
        result_array[j, 0, 0, i] = float(totol_mse_x)
        result_array[j, 1, 0, i] = float(totol_mae_x)
        result_array[j, 2, 0, i] = float(totol_rmse_x)
        result_array[j, 3, 0, i] = float(totol_mape_x)
        result_array[j, 4, 0, i] = float(R2_x)

        result_array[j, 0, 1, i] = float(totol_mse_y)
        result_array[j, 1, 1, i] = float(totol_mae_y)
        result_array[j, 2, 1, i] = float(totol_rmse_y)
        result_array[j, 3, 1, i] = float(totol_mape_y)
        result_array[j, 4, 1, i] = float(R2_y)

        result_array[j, 0, 2, i] = float(totol_mse_o)
        result_array[j, 1, 2, i] = float(totol_mae_o)
        result_array[j, 2, 2, i] = float(totol_rmse_o)
        result_array[j, 3, 2, i] = float(totol_mape_o)
        result_array[j, 4, 2, i] = float(R2_o)

        result_array[j, 0, 3, i] = float(totol_mse_e)
        result_array[j, 1, 3, i] = float(totol_mae_e)
        result_array[j, 2, 3, i] = float(totol_rmse_e)
        result_array[j, 3, 3, i] = float(totol_mape_e)
        result_array[j, 4, 3, i] = float(R2_e)

        result_array[j, 0, 4, i] = float(totol_mse_sym)
        result_array[j, 1, 4, i] = float(totol_mae_sym)
        result_array[j, 2, 4, i] = float(totol_rmse_sym)
        result_array[j, 3, 4, i] = float(totol_mape_sym)
        result_array[j, 4, 4, i] = float(R2_sym)
    wind_data.append(wind)
    orient_data.append(orient)

# 保存为npy文件
np.save(os.path.join(folder_path, "{}_5_5_17error.npy".format(n)), result_array)
np.save(os.path.join(folder_path, "{}wind_error.npy".format(n)), np.array(wind_data))
np.save(os.path.join(folder_path, "{}orient_error.npy".format(n)), np.array(orient_data))