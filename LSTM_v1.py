import os
import tensorflow._api.v2.compat.v1 as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten, Conv1D
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import BatchNormalization
import numpy as np
import pandas
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import math
import dateutil
import matplotlib.dates as md
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
from sklearn.metrics import r2_score  # R square
import datetime
import matplotx
from tensorflow.keras.models import load_model

# ===================================================================================================================================
# ============================================================各种自定义参数============================================================
# ===================================================================================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 选择运行GPU
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
LossFunction = ['MSE', 'QuantileLoss', 'HuberLoss']
usefunction = 'MSE'
use_step = 36  # 回视窗口大小
pre_step = 12  # 预测的步长
# 读取数据集
df = pd.read_csv(r'data\ETT\uv.csv')
y_data = df.iloc[:, 1:]


def create_dataset(data: list, time_step: int):
    arr_x, arr_y = [], []
    for i in range(len(data) - time_step-pre_step - 1):
        x = data[i: i + time_step]
        y = data[i + time_step:i + time_step+pre_step]
        arr_x.append(x)
        arr_y.append(y)
    return np.array(arr_x), np.array(arr_y)

# 一类损失函数的定义

from tensorflow.keras.losses import Loss, Reduction
import keras.backend as K

class QuantileLoss(Loss):
    def __init__(self, quantile=0.5, reduction=Reduction.AUTO, name='quantile_loss'):
        super(QuantileLoss, self).__init__(reduction=reduction, name=name)
        self.quantile = quantile

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        loss = K.mean(K.maximum(self.quantile * error, (self.quantile - 1) * error))
        return loss


from tensorflow.keras.losses import Loss, Reduction, Huber
import keras.backend as K

# 定义 Huber Loss
class HuberLoss(Loss):
    def __init__(self, delta=0.5, reduction=Reduction.AUTO, name='huber_loss'):
        super(HuberLoss, self).__init__(name=name, reduction=reduction)
        self.delta = delta
        self.huber_loss = Huber(delta=self.delta, reduction=reduction)

    def call(self, y_true, y_pred):
        return self.huber_loss(y_true, y_pred)


# Model Construction
def modelconstruction(use_step, pre_step, num_classes, sellect_compile):
    # Model construction
    # Model 3
    model3 = Sequential()
    model3.add(LSTM(2048, activation='relu', input_shape=(use_step, num_classes), return_sequences=True))
    # model3.add(BatchNormalization())
    model3.add(LSTM(1024, activation='relu'))
    # model3.add(BatchNormalization())
    model3.add(Flatten())
    # model3.add(Dense(34*2, activation='linear'))
    model3.add(Dense(units=pre_step*num_classes, activation='linear'))
    # Model Loss Evaluation

    if sellect_compile=="MSE":
        model3.compile(loss=keras.losses.MSE,
                       optimizer=keras.optimizers.Adam(),
                       metrics=["mean_absolute_error"])
    if sellect_compile=="QuantileLoss":
        model3.compile(loss=QuantileLoss(quantile=0.5),
                       optimizer=keras.optimizers.Adam(),
                       metrics=["mean_absolute_error"])
    if sellect_compile=="HuberLoss":
        model3.compile(loss=HuberLoss(delta=0.5),
                       optimizer=tf.keras.optimizers.Adam(),
                       metrics=["mean_absolute_error"])

    return model3


# Data_Preprocessing
def get_train_batch(datalist, batch_size):
    x_data, y_data = create_dataset(trainlist, use_step)
    while 1:
        for i in range(0, len(datalist), batch_size):
            x = (x_data[i:i + batch_size]).reshape(-1, use_step, 34)
            y = y_data[i:i + batch_size].reshape(-1, pre_step * 34)
            # 最重要的就是这个yield，它代表返回，返回以后循环还是会继续，然后再返回。就比如有一个机器一直在作累加运算，但是会把每次累加中间结果告诉你一样，直到把所有数加完
            yield ({'lstm_input': x}, {'dense': y})


def modeltraining(model, modelname, batch_size, epochs, num_classes, trainlist, vallist):
    checkpoint = ModelCheckpoint(filepath=modelname, monitor='val_loss', mode='auto', save_best_only='True')# val_loss
    model.summary()
    history = model.fit_generator(generator=get_train_batch(trainlist, batch_size),
                        steps_per_epoch=len(trainlist) // batch_size + 1,
                        epochs=epochs, verbose=1,  # 记录信息的方式
                        validation_data=get_train_batch(vallist, batch_size),
                        validation_steps=len(vallist) // batch_size + 1,
                        callbacks=[checkpoint])
    model.summary()
    return history


split_ratio = 0.8
len_train = int(len(y_data) * split_ratio)
trainlist_ = y_data[:len_train - int(len_train/8)]
vallist_ = y_data[len_train + int(len_train/8):]
testlist_ = y_data[len_train:]
from sklearn.preprocessing import StandardScaler

# 创建 StandardScaler 对象
scaler = StandardScaler()

# 对训练数据进行拟合并标准化
trainlist = scaler.fit_transform(trainlist_)

# 使用相同的拟合器对验证数据进行标准化
vallist = scaler.transform(vallist_)

# 使用相同的拟合器对测试数据进行标准化
testlist = scaler.transform(testlist_)

import time

root_p = 'lstm_{}step_{}2048_1024'.format(pre_step, usefunction)

# 检查目标文件夹是否存在，如果不存在则创建
folder_path = os.path.join("checkpoint", root_p)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


# 获取程序开始时间
start_time = time.time()
modelname = os.path.join("checkpoint", root_p, 'lstm.h5')
batch_size = 512
epochs = 1
num_classes = 34
# model = load_model(modelname)
model = modelconstruction(use_step, pre_step, num_classes, usefunction)
history = modeltraining(model, modelname, batch_size, epochs, num_classes, trainlist, vallist)
# 获取程序结束时间
end_time = time.time()

# 计算程序运行时间
run_time = end_time - start_time
print(f"训练运行时间为：{run_time} 秒")

# 检查目标文件夹是否存在，如果不存在则创建
folder_path = os.path.join("results", root_p)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# plt.plot(history.history['loss'], label='train_loss')
# plt.plot(history.history['val_loss'], label='val_loss')
# np.savetxt(os.path.join(folder_path, 'lstm_loss.csv'), history.history['loss'], delimiter=",")
# np.savetxt(os.path.join(folder_path, 'lstm_val_loss.csv'), history.history['val_loss'], delimiter=",")
# plt.legend()
# plt.savefig(os.path.join(folder_path, 'lstm_loss.png'))
# plt.show()


# 加载模型
# model = load_model(modelname)

# model = load_model(modelname, custom_objects={'QuantileLoss': QuantileLoss})
# model = load_model(modelname, custom_objects={'HuberLoss': HuberLoss})

# 加载预测数据

# 预测

time_step = use_step
test_feat, test_label = create_dataset(testlist, time_step)

y_true = scaler.inverse_transform(test_label.reshape(-1, 34)).reshape(-1, pre_step, 34)

y_pred = scaler.inverse_transform(model.predict(test_feat).reshape(-1, 34)).reshape(-1, pre_step, 34)

# 保存numpy数组到文件
save_path = os.path.join("results", root_p)
os.makedirs(save_path, exist_ok=True)
# 保存y_true和y_pred数组
np.save(os.path.join(save_path, 'true.npy'), y_true)
np.save(os.path.join(save_path, 'pred.npy'), y_pred)