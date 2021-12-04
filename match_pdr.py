import utils as pdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from parameters import WALKING_DATA_FILE

# real_trace = pd.read_csv(REAL_TRACE_FILE).values  # 真实轨迹
df_walking = pd.read_csv(WALKING_DATA_FILE)

# 获得线性加速度、重力加速度、旋转向量的numpy.ndarray数据
linear = df_walking[[col for col in df_walking.columns if 'linear' in col]].values
gravity = df_walking[[col for col in df_walking.columns if 'gravity' in col]].values
rotation = df_walking[[col for col in df_walking.columns if 'rotation' in col]].values
eu_g_yaw = df_walking[[col for col in df_walking.columns if 'eu_g_yaw' in col]].values

orientation = df_walking[[col for col in df_walking.columns if 'orientation' in col]].values
eu_c_yaw = df_walking[[col for col in df_walking.columns if 'eu_c_yaw' in col]].values

pdr = pdr.Model(linear, gravity, rotation)

# Demo1：显示垂直方向合加速度与步伐波峰分布
# frequency：数据采集频率
# walkType：行走方式（normal为正常走路模式，abnormal为做融合定位实验时走路模式）
# pdr.show_steps(frequency=100, walkType='normal')


# # Demo2：显示数据在一定范围内的分布情况，用来判断静止数据呈现高斯分布
# # 传入参数为静止状态x（y或z）轴线性加速度
# acc_z = linear[:,2]
# pdr.show_gaussian(acc_z, True)


# # Demo3：显示三轴线性加速度分布情况
# pdr.show_data('linear')


# # Demo4/5：获取步伐信息
# # 返回值steps为字典类型，index为样本序号，acceleration为步伐加速度峰值
# steps = pdr.step_counter(frequency=100, walkType='normal')
# print(steps)
# print('steps:', len(steps))
#
# stride = pdr.step_stride # 步长推算函数
# # 计算步长推算的平均误差
# accuracy = []
# for v in steps:
#     a = v['acceleration']
#     print(stride(a))
#     accuracy.append(
#         np.abs(stride(a)-0.8)
#     )
# square_sum = 0
# for v in accuracy:
#     square_sum += v*v
# acc_mean = (square_sum/len(steps))**(1/2)
# print("mean: %f" % acc_mean) # 平均误差
# print("min: %f" % np.min(accuracy)) # 最小误差
# print("max: %f" % np.max(accuracy)) # 最大误差
# print("sum: %f" % np.sum(accuracy)) # 累积误差


# # Demo6：获取航向角
# # 实验过程中手机采用HOLDING模式，即手握手机放在胸前，并且x轴与地面平行，
# # `step_heading`直接返回每一时刻的偏航角yaw值，初始值默认设为0，这里的都是相对值。
# theta = pdr.step_heading()[:10]
# temp = theta[0]
# for i,v in enumerate(theta):
#     # 弧度制=>角度制
#     theta[i] = np.abs(v-temp)*360/(2*np.pi)
#     print(theta[i])
# print("mean: %f" % np.mean(theta))


# Demo7/8：显示PDR预测轨迹
# 注意：PDR不清楚初始位置与初始航向角
# pdr.show_trace(frequency=100, walkType='normal', offset=-MAG_DECLINATION)

x, y, step = pdr.show_trace(frequency=100, walkType='normal', initPosition=(0, 0), offset=0)
# ini_angle1 = angle1[1] * 360 / (2 * np.pi)
# print("IMU初始航向角：", ini_angle1, "° (与地理北向的夹角，顺时针为正方向)")
# ini_angle2 = angle2[1] * 360 / (2 * np.pi)
# print("Orientation初始航向角：", ini_angle2, "° (与地理北向的夹角，顺时针为正方向)")
# ini_angle3 = angle3[1] * 360 / (2 * np.pi)
# print("Android初始航向角：", ini_angle3, "° (与地理北向的夹角，顺时针为正方向)")


# pdr_predictions = [([0] * 2) for i in range(steps+1)]
# for i in range(steps+1):
#     pdr_predictions[i][0] = x_a3[i]
#     pdr_predictions[i][1] = y_a3[i]
#
# real_trace = real_trace[0:steps+1,:]
# pdr_acc = accuracy(np.array(pdr_predictions), np.array(real_trace))
# print("PDR_accuracy: ", round(pdr_acc, 3), "m")
#
# res_plot(real_trace, pdr_predictions, title="PDR Loction")
# errors = calculate_error(real_trace, pdr_predictions)

# 误差数据导出
# test = pd.DataFrame(data=errors)
# test.to_csv('C:/Users/KouMengya/Desktop/test.csv',encoding='gbk')
# cdf_plot(np.reshape(errors, (len(errors),)).tolist(), x_range=15, title="PDR Loction")