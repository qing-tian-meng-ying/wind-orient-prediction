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
    return np.mean(angle_error)


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


# 路径
path = 'lstm_12step_MSE2048_1024'

loadData_t = np.load(r'.\result'
                     r's\{}\true.npy'.format(path)).squeeze()
loadData_p = np.load(r'.\result'
                     r's\{}\pred.npy'.format(path)).squeeze()

# 检查目标文件夹是否存在，如果不存在则创建
folder_path = os.path.join("img", path)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 获取数组的维度数
num_dims = len(loadData_t.shape)

# 根据维度数判断获取第二维的大小，如果没有第三维则将其设置为1
n = loadData_t.shape[1] if num_dims >= 2 else 1

plt.rcParams['font.family'] = 'Times New Roman'
plt.style.use(matplotx.styles.pitaya_smoothie['light'])
# fig, ax = plt.subplots(figsize=(18, 10))


for j in range(n):
    y_true = loadData_t[:, j, :]
    y_pred = loadData_p[:, j, :]
    img_path = os.path.join(folder_path, "step{}".format(j))
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    for i in range(17):
        x = y_true[:, i * 2]
        y = y_pred[:, i * 2]
        u = y_true[:, i * 2 + 1]
        v = y_pred[:, i * 2 + 1]
        # 去除0元素方便计算MAPE
        x_nonzero, y_nonzero = remove_zero(x, y)
        u_nonzero, v_nonzero = remove_zero(u, v)
        # 绘图
        fig = plt.figure(figsize=(20, 10))
        grid = plt.GridSpec(25, 40, wspace=0.4, hspace=0.3)

        # x方向的分量
        ax_x = plt.subplot(grid[0:4, :37])
        tt = np.arange(1, len(x) + 1)
        ax_x.scatter(tt, np.array(x), label='x_true', s=0.4)
        ax_x.scatter(tt, np.array(y), label='x_pred', s=0.4)
        ax_x.set_ylabel('Horizontal Velocity(m/s)', fontsize=7)
        ax_x.legend(fontsize=10, loc='upper left')

        totol_mse_x, totol_mae_x, R2_x = count_error(x, y, [0, 1])
        totol_mape_x = '%.4g' % mape(x_nonzero, y_nonzero)
        totol_rmse_x = '%.4g' % np.sqrt(float(totol_mse_x))
        texts_x = plt.subplot(grid[0:4, 37:])
        texts_x.text(0.1, 0.1, 'MSE:{}\nRMSE:{}\nMAE:{}\nMAPE:{}\nR square:{}\npoints:{}'.
                     format(totol_mse_x, totol_rmse_x, totol_mae_x, totol_mape_x, R2_x, len(x)), c='black')
        texts_x.axis('off')

        # y方向的分量
        ax_y = plt.subplot(grid[5:9, :37])
        tt = np.arange(1, len(u) + 1)
        ax_y.scatter(tt, np.array(u), label='y_true', s=0.4)
        ax_y.scatter(tt, np.array(v), label='y_pred', s=0.4)
        ax_y.set_ylabel('Vertical Velocity(m/s)', fontsize=7)
        ax_y.legend(fontsize=10, loc='upper left')

        totol_mse_y, totol_mae_y, R2_y = count_error(u, v, [0, 1])
        totol_rmse_y = '%.4g' % np.sqrt(float(totol_mse_y))
        totol_mape_y = '%.4g' % mape(u_nonzero, v_nonzero)
        texts_y = plt.subplot(grid[5:9, 37:])
        texts_y.text(0.1, 0.1, 'MSE:{}\nRMSE:{}\nMAE:{}\nMAPE:{}\nR square:{}\npoints:{}'.
                     format(totol_mse_y, totol_rmse_y, totol_mae_y, totol_mae_y, R2_y, len(u)), c='black')
        texts_y.axis('off')

        # x和y的合方向
        ax_o = plt.subplot(grid[10:15, :37])
        y_ori = back_angle(np.array(x), np.array(u))
        # y_ori = np.array(df_dir.iloc[len_train + 0:-1, i])
        y_ori_pred = back_angle(np.array(y), np.array(v))
        # y_ori_pred = back_angle(np.array(y_data[len_train + 4:, i]), np.array(y_data[len_train + 4:, i + 17]))
        ax_o.scatter(tt, y_ori, label='orient_true', s=0.4)
        ax_o.scatter(tt, y_ori_pred, label='orient_pred', s=0.4)
        ax_o.set_ylabel('Direction', fontsize=10)
        ax_o.legend(fontsize=10, loc='upper left')

        totol_mse_o = '%.4g' % mean_angular_error(y_ori, y_ori_pred)
        totol_rmse_o = '%.4g' % np.sqrt(mean_squared_error(y_ori, y_ori_pred))
        totol_mae_o = '%.4g' % mean_angle_error(np.stack([x, u], axis=-1), np.stack([y, v], axis=-1))
        # 去除0元素方便计算MAPE
        dir_true_nonzero, dir_pred_nonzero = remove_zero(np.array(y_ori), np.array(y_ori_pred))
        totol_mape_o = '%.4g' % mape(dir_true_nonzero, dir_pred_nonzero)
        R2_o = '%.4g' % r2_score(y_ori, y_ori_pred)

        texts_o = plt.subplot(grid[10:15, 37:])
        texts_o.text(0.1, 0.1, 'MangleE:{}\nRMSE:{}\nMAE:{}\nMAPE:{}\nR square:{}\npoints:{}'.
                     format(totol_mse_o, totol_rmse_o, totol_mae_o, totol_mape_o, R2_o, len(y_ori)), c='black')
        texts_o.axis('off')

        # 解卷绕
        ax_e = plt.subplot(grid[16:20, :37])
        un_w_y_true = dispatch(y_ori)
        un_w_y_pred = dispatch(y_ori_pred)
        un_w_y_pred = fix_dir(un_w_y_true[0], un_w_y_pred[0])
        ax_e.scatter(tt, un_w_y_true[0], label='unwrap true', s=0.4)
        ax_e.scatter(tt, un_w_y_pred, label='unwrap pred', s=0.4)
        ax_e.legend(fontsize=10, loc='upper left')
        ax_e.set_ylabel('unwrap-angle', fontsize=10)

        totol_mse_e, totol_mae_e, R2_e = count_error(un_w_y_true[0], un_w_y_pred, [0, 1])
        totol_rmse_e = '%.4g' % np.sqrt(float(totol_mse_e))
        # 去除0元素方便计算MAPE
        un_w_y_true_nonzero, un_w_y_pred_nonzero = remove_zero(un_w_y_true[0], un_w_y_pred)
        totol_mape_e = '%.4g' % mape(un_w_y_true_nonzero, un_w_y_pred_nonzero)

        texts_e = plt.subplot(grid[16:20, 37:])
        texts_e.text(0.1, 0.1, 'MSE:{}\nRMSE:{}\nMAE:{}\nMAPE:{}\nR square:{}\npoints:{}'.
                     format(totol_mse_e, totol_rmse_e, totol_mae_e, totol_mape_e, R2_e, len(un_w_y_pred)), c='black')
        texts_e.axis('off')

        # x和y的合速度
        ax_sym = plt.subplot(grid[21:, :37])
        y_sym = np.sqrt(((np.array(x) ** 2) + (np.array(u) ** 2)))
        # y_sym = np.array(df_wind.iloc[len_train + 0:-1, i])
        y_sym_pred = np.sqrt(((np.array(y) ** 2) + (np.array(v) ** 2)))
        ax_sym.scatter(tt, y_sym, label='v_true', s=0.4)
        ax_sym.scatter(tt, y_sym_pred, label='v_pred', s=0.4)
        ax_sym.set_ylabel('Velocity(m/s)', fontsize=10)
        ax_sym.legend(fontsize=10, loc='upper left')

        totol_mse_sym, totol_mae_sym, R2_sym = count_error(y_sym, y_sym_pred, [0, 1])
        totol_rmse_sym = '%.4g' % np.sqrt(float(totol_mse_sym))
        # 去除0元素方便计算MAPE
        wind_true_nonzero, wind_pred_nonzero = remove_zero(y_sym, y_sym_pred)
        totol_mape_sym = '%.4g' % mape(wind_true_nonzero, wind_pred_nonzero)

        texts_sym = plt.subplot(grid[20:, 37:])
        texts_sym.text(0.1, 0.1, 'MSE:{}\nRMSE:{}\nMAE:{}\nMAPE:{}\nR square:{}\npoints:{}'.
                       format(totol_mse_sym, totol_rmse_sym, totol_mae_sym, totol_mape_sym, R2_sym, len(y_sym)),
                       c='black')
        texts_sym.axis('off')
        plt.savefig(os.path.join(img_path, "{}.png".format(i + 0)), dpi=300)
        # plt.show()


