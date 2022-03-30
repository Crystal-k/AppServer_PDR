'''
实验前提：采集数据时手机x轴与重力方向垂直

1.Model
参数列表（2个参数）：
线性加速度矩阵（x轴加速度、y轴加速度、z轴加速度）；
方向矩阵（yaw、pitch、roll）;

2.Model参数类型：
numpy.ndarray
'''

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

from parameters import RESULT_FIGURE

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Model(object):
    def __init__(self, linear, orientation):
        self.linear = linear
        self.orientation = orientation

        if(len(linear)!= len(orientation)):
            print("数据不合格！")


    # == 步伐检测（计步） ============================
    '''
        功能：
          获得垂直方向上的合加速度（除重力加速度外）
    '''
    def coordinate_conversion(self):
        linear = self.linear
        pitch = self.orientation[:, 1]  # [-180, 180]，平放为0°，顶端朝下为正

        if -45 < np.average(pitch) <= 45:
            print("手机为屏幕朝上的水平状态。")
            a_vertical = linear[:, 2]
        elif -135 < np.average(pitch) <= -45:
            print("手机为摄像头朝前的竖直状态。")
            a_vertical = linear[:, 1]
        elif -180 <= np.average(pitch) <= -135 or 135 < np.average(pitch) <= 180:
            print("手机为屏幕朝下的水平状态。")
            a_vertical = -linear[:, 2]
        elif 45 < np.average(pitch) <= 135:
            print("手机为摄像头朝用户的竖直状态。")
            a_vertical = -linear[:, 1]

        return a_vertical

    '''
        功能：
          步数检测函数（峰值检测）
    
        walkType取值：
          normal：正常行走模式
          abnormal：融合定位行走模式（每一步行走间隔大于1s）

        返回值：
          steps：字典型数组，每个字典保存了峰值位置（index）与该点的合加速度值（acceleration）
    '''
    def step_counter(self, frequency=100, walkType='normal'):
        offset = frequency / 100  # 自动转化为浮点数
        g = 9.807
        a_vertical = self.coordinate_conversion()
        slide = 50 * offset  # 滑动窗口（100Hz的采样数据）=>不管采样频率多少，大小为0.4s内采集到的样本数
        frequency = 100 * offset

        # 行人加速度阈值（小程序采集到的是负值 ）
        min_acceleration = -0.85 * g  # 0.2g
        max_acceleration = -0.02 * g  # 2g

        # 峰值间隔(s)
        # min_interval = 0.4
        min_interval = 0.5 if walkType == 'normal' else 3  # 'abnormal
        # max_interval = 1

        # 计算步数
        steps = []
        peak = {'index': 0, 'acceleration': -100}

        # 以40*offset为滑动窗检测峰值
        # 条件1：峰值在0.25g~2g之间
        for i, v in enumerate(a_vertical):
            if v >= peak['acceleration'] and v >= min_acceleration and v <= max_acceleration:
                peak['acceleration'] = v
                peak['index'] = i
            if i % slide == 0 and peak['index'] != 0:
                steps.append(peak)
                peak = {'index': 0, 'acceleration': -100}

        # 条件2：两个峰值之前间隔至少大于0.4s*offset
        # del使用的时候，一般采用先记录再删除的原则
        if len(steps) > 0:
            lastStep = steps[0]
            dirty_points = []
            for key, step_dict in enumerate(steps):
                # print(step_dict['index'])
                if key == 0:
                    continue
                if step_dict['index'] - lastStep['index'] < min_interval * frequency:
                    # print('last:', lastStep['index'], 'this:', step_dict['index'])
                    if step_dict['acceleration'] <= lastStep['acceleration']:
                        dirty_points.append(key)
                    else:
                        lastStep = step_dict
                        dirty_points.append(key - 1)
                else:
                    lastStep = step_dict

            counter = 0  # 记录删除数量，作为偏差值
            for key in dirty_points:
                del steps[key - counter]
                counter = counter + 1

        return steps


    # == 步长估计 ============================
    # 目前的方法不具备科学性，临时使用
    # crystal: 经验模型L=C.((a_max-a_min)**(1/4))（C取0.4; a_min取0）
    def step_stride(self, max_acceleration):
        return np.power(abs(max_acceleration), 1 / 4) * 0.4


    # == 航向估计 ============================
    '''
        功能：
          利用小程序的方向传感器采集到的姿态角（单位：角度0~360，顺时针为正）
    '''
    def step_heading_orientation(self):
        yaw = self.orientation[:, 0] * (np.pi / 180)
        return yaw


    # == 轨迹恢复 ============================
    '''
        功能：
          计算步行轨迹的每一个相对坐标位置

        参数：
          frequency：数据采集频率
          walkType：行走模式
          offset：初始航向角大小
          initPosition：初始位置，格式为两个元素的元组形式，分别表示x与y

        返回值：
          每一步的x、y坐标值，以及每一步的步长和航向角
    '''
    def pdr_position_orientation(self, frequency=100, walkType='normal', initPosition=(0, 0), offset=0):
        yaw = self.step_heading_orientation()
        steps = self.step_counter(frequency=frequency, walkType=walkType)
        position_x = []
        position_y = []
        x = initPosition[0]
        y = initPosition[1]
        position_x.append(x)
        position_y.append(y)
        strides = []
        angle = [offset]

        for v in steps:
            index = v['index']

            length = self.step_stride(v['acceleration'])
            strides.append(length)

            theta = yaw[index] + offset
            angle.append(theta)

            x = x + length * np.sin(theta)
            y = y + length * np.cos(theta)
            position_x.append(x)
            position_y.append(y)

        # 步长计入一个状态中，最后一个位置没有下一步，因此步长记为0
        return position_x, position_y, strides + [0], angle


    '''
        功能：
          显示PDR运动轨迹图
          
        参数：
          frequency：数据采集频率
          walkType：行走模式
          initPosition：初始位置，格式为两个元素的元组形式，分别表示x与y
          offset：轨迹偏差角度（指上北下南地图中的轨迹旋转到输出轨迹的角度值，实验过程使用了安卓的旋转矢量软件传感器，
                  它集成了加速度计、陀螺仪和磁力计的数据，最终输出了一个以地球坐标系为基础的绝对信息，
                  但实验输出图有时候会分析一个相对定位的情景，所以可以用该偏差值进行修正）
          realTrace：两列的numpy.ndarray格式数据，表示真实轨迹坐标，主要是为了方便轨迹的对比（可选）
    '''
    def show_trace(self, frequency=100, walkType='normal', initPosition=(0, 0), **kw):
        handles = []
        labels = []
        stepsAxis = []
        if 'real_trace' in kw:
            real_trace = kw['real_trace'].T
            trace_x = real_trace[0]
            trace_y = real_trace[1]
            l0, = plt.plot(trace_x, trace_y, color='b')
            handles.append(l0)
            labels.append('real tracks')
            plt.scatter(trace_x, trace_y, color='b')
            for k in range(0, len(trace_x)):
                plt.annotate(k, xy=(trace_x[k], trace_y[k]), xytext=(trace_x[k] + 0.1, trace_y[k] + 0.1))

        if 'offset' in kw:
            offset = kw['offset']
        else:
            offset = 0
        x1, y1, _, angle1 = self.pdr_position_orientation(frequency=frequency, walkType=walkType, initPosition=initPosition,
                                                  offset=offset)

        plt.cla() #清除当前图像，若不清除则前面画的图保留
        l1, = plt.plot(x1, y1, '*-')
        handles.append(l1)
        labels.append('IMU predicting')

        for k in range(0, len(x1)):
            if (k == 0):
                plt.annotate("起点", xy=(x1[k], y1[k]), xytext=(x1[k] + 0.1, y1[k] + 0.1))
                stepsAxis.append([x1[k], y1[k]])
            else:
                plt.annotate(k, xy=(x1[k], y1[k]), xytext=(x1[k] + 0.1, y1[k] + 0.1))
                stepsAxis.append([x1[k], y1[k]])

        plt.legend(handles=handles, labels=labels, loc='best')
        plt.xlabel("单位：m")
        plt.ylabel("单位：m")
        plt.grid(True)
        plt.axis('equal')

        plt.pause(0.1)
        # plt.show()
        plt.savefig(RESULT_FIGURE + "result.jpg")

        steps = len(x1) - 1

        return x1, y1, steps, stepsAxis



    # == demo测试 ============================
    '''
        功能：
          显示步伐检测图像及检测步数结果

        walkType取值：
          normal：正常行走模式
          abnormal：融合定位行走模式（每一步行走间隔大于1s）
    '''
    def show_steps(self, frequency=100, walkType='normal'):
        a_vertical = self.coordinate_conversion()
        steps = self.step_counter(frequency=frequency, walkType=walkType)

        index_test = []
        value_test = []
        for v in steps:
            index_test.append(v['index'])
            value_test.append(v['acceleration'])

        textstr = '='.join(('steps', str(len(steps))))  # 显示字符"steps=18"
        _, ax = plt.subplots()
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        plt.plot(a_vertical)

        plt.scatter(index_test, value_test, color='r')
        plt.xlabel('samples')
        plt.ylabel('vertical acceleration')
        # plt.show()
        plt.savefig(RESULT_FIGURE + "show_steps.jpg")


    '''
        功能：
          输出一个数据分布散点图, 用来判断某一类型数据的噪声分布情况, 通常都会是高斯分布，可以用来分析静止惯性数据。

        参数：
          data：某一轴加速度数据
          fit：布尔值，是否进行高斯拟合
    '''
    def show_gaussian(self, data, fit):
        wipe = 150
        data = data[wipe:len(data) - wipe]
        division = 100
        acc_min = np.min(data)
        acc_max = np.max(data)
        interval = (acc_max - acc_min) / division
        counter = [0] * division
        index = []

        for k in range(division):
            index.append(acc_min + k * interval)

        for v in data:
            for k in range(division):
                if v >= (acc_min + k * interval) and v < (acc_min + (k + 1) * interval):
                    counter[k] = counter[k] + 1

        textstr = '\n'.join((
            r'$max=%.3f$' % (acc_max,),
            r'$min=%.3f$' % (acc_min,),
            r'$mean=%.3f$' % (np.mean(data),)))
        _, ax = plt.subplots()
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        plt.scatter(index, counter, label='distribution')

        if fit == True:
            length = math.ceil((acc_max - acc_min) / interval)
            counterArr = length * [0]
            for value in data:
                key = int((value - acc_min) / interval)
                if key >= 0 and key < length:
                    counterArr[key] += 1
            normal_mean = np.mean(data)
            normal_sigma = np.std(data)
            normal_x = np.linspace(acc_min, acc_max, 100)
            normal_y = norm.pdf(normal_x, normal_mean, normal_sigma)
            normal_y = normal_y * np.max(counterArr) / np.max(normal_y)
            ax.plot(normal_x, normal_y, 'r-', label='fitting')

        plt.xlabel('acceleration')
        plt.ylabel('total samples')
        plt.legend()
        # plt.show()
        plt.savefig(RESULT_FIGURE + "show_gaussian.jpg")


    '''
        功能：
          显示三轴加速度的变化情况

        dataType取值：
          linear：查看三轴线性加速度的分布情况 
          gravity：查看三轴重力加速度的分布情况 
          rotation：查看旋转四元数的数据分布情况
    '''
    def show_data(self, dataType):
        if dataType == 'linear':
            linear = self.linear
            x = linear[:, 0]
            y = linear[:, 1]
            z = linear[:, 2]
            index = range(len(x))

            ax1 = plt.subplot(3, 1, 1)  # 第一行第一列图形
            ax2 = plt.subplot(3, 1, 2)  # 第一行第二列图形
            ax3 = plt.subplot(3, 1, 3)  # 第二行
            plt.sca(ax1)
            plt.title('x')
            plt.scatter(index, x)
            plt.sca(ax2)
            plt.title('y')
            plt.scatter(index, y)
            plt.sca(ax3)
            plt.title('z')
            plt.scatter(index, z)
            # plt.show()
            plt.savefig(RESULT_FIGURE + "show_data.jpg")
        elif dataType == 'gravity':
            gravity = self.gravity
            x = gravity[:, 0]
            y = gravity[:, 1]
            z = gravity[:, 2]
            index = range(len(x))

            ax1 = plt.subplot(3, 1, 1)  # 第一行第一列图形
            ax2 = plt.subplot(3, 1, 2)  # 第一行第二列图形
            ax3 = plt.subplot(3, 1, 3)  # 第二行
            plt.sca(ax1)
            plt.title('x')
            plt.scatter(index, x)
            plt.sca(ax2)
            plt.title('y')
            plt.scatter(index, y)
            plt.sca(ax3)
            plt.title('z')
            plt.scatter(index, z)
            plt.show()
        else:  # rotation
            rotation = self.rotation
            x = rotation[:, 0]
            y = rotation[:, 1]
            z = rotation[:, 2]
            w = rotation[:, 3]
            index = range(len(x))

            ax1 = plt.subplot(4, 1, 1)  # 第一行第一列图形
            ax2 = plt.subplot(4, 1, 2)  # 第一行第二列图形
            ax3 = plt.subplot(4, 1, 3)  # 第二行
            ax4 = plt.subplot(4, 1, 4)  # 第二行
            plt.sca(ax1)
            plt.title('x')
            plt.scatter(index, x)
            plt.sca(ax2)
            plt.title('y')
            plt.scatter(index, y)
            plt.sca(ax3)
            plt.title('z')
            plt.scatter(index, z)
            plt.sca(ax4)
            plt.title('w')
            plt.scatter(index, w)
            plt.show()
