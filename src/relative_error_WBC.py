import os
import numpy as np
from scipy.integrate import solve_ivp
import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd
from pyswarm import pso
import re
import csv
from tqdm import tqdm

# 读取血常规数据文件
# data = "Cleaned_Blood_routine_data.xlsx"  # 请使用您的CSV文件路径替换
# data1 = pd.read_excel(data, engine='openpyxl')
data1 = "combined_Blood_routine_data.csv"
data1 = pd.read_csv(data1, sep=',')

# 读取 用药方案 文件
# file_path = "updated_drug_dosage_plan.xlsx"  # 请使用您的文件路径替换
# data2 = pd.read_excel(file_path, engine='openpyxl')
data2 = "combined_drug_dosage_plan.csv"
data2 = pd.read_csv(data2, sep=',')


def calculate_error_for_patient(patient_id, B, gamma, ktr, slopeA, slopeD, data1, data2):
    # 筛选出病人ID为patient_id的数据
    patient_data1 = data1[data1['id'] == patient_id]
    # 筛选 Start Time 小于等于 25 的数据行
    patient_data2 = data2[(data2['id'] == patient_id) & (data2['Start Time'] <= 25)]

    # 从筛选出的数据中提取'time'和'WBC'两列
    time_wbc_data = patient_data1[['time', 'WBC']]
    # 将DataFrame转换为NumPy数组
    time_wbc_array = time_wbc_data.to_numpy()
    timeWBC = time_wbc_array[:, 0]
    xma_data = time_wbc_array[:, 1]
    # 筛选出时间小于等于0的数据
    time_le_0_data = time_wbc_data[time_wbc_data['time'] <= 0]
    # 检查是否存在 time 为 0 的数据点
    time_0_data = time_wbc_data[time_wbc_data['time'] == 0]

    if not time_0_data.empty:
        # 如果存在 time 为 0 的数据点，提取对应的 WBC 值并赋给 B
        B0 = time_0_data['WBC'].values[0]
    elif not time_le_0_data.empty:
        # 如果不存在 time 为 0 的数据点，但存在 time <= 0 的数据点，
        # 找到 time 最接近 0 的前后两个数据点
        time_before_0 = time_le_0_data.loc[time_le_0_data['time'].idxmax()]
        # Find the data point just after time = 0
        time_after_0 = time_wbc_data[time_wbc_data['time'] > 0].iloc[0]

        # 对它们的 WBC 值进行插值，得到 time 为 0 时刻的 WBC 值，并赋给 B
        B0 = np.interp(0, [time_before_0['time'], time_after_0['time']], [time_before_0['WBC'], time_after_0['WBC']])
    else:
        # 如果不存在 time <= 0 的数据点，找到 time 最接近且大于0的数据点
        time_after_0 = time_wbc_data.loc[time_wbc_data['time'].idxmin()]
        B0 = time_after_0['WBC']

    kma = 2.3765

    patient_data2 = data2[data2['id'] == patient_id]

    def preprocess_dosage_data(patient_data2):
        from collections import defaultdict
        # 初始化药物的剂量字典，其中键是(药物, 时间点)元组，值是累积剂量
        dosage_timeline = defaultdict(float)

        # 遍历给药记录，累积每天的剂量
        for _, row in patient_data2.iterrows():
            drug = row['Drug']
            start_time = row['Start Time']
            end_time = min(row['End Time'], 25)  # 如果End Time大于25，将其设置为25
            dosage_str = row['Dosage']

            # 使用正则表达式匹配剂量字符串中的数字，包括可能的小数点和空格
            match = re.search(r'(\d+\.?\d*)\s*mg', dosage_str)
            if match:
                # 转换剂量为浮点数
                dosage_value = float(match.group(1))
                # 将每个时间点的用药量累加
                for time_point in range(start_time, end_time + 1):
                    key = (drug, time_point)  # 使用药物和时间点作为键
                    dosage_timeline[key] += dosage_value

        return dosage_timeline

    def get_u(t, dosage_timeline):
        uA, uD = 0, 0
        # 遍历dosage_timeline累积特定时间点的剂量
        for (drug, time_point), dosage in dosage_timeline.items():
            if time_point == t:
                if drug == 'Ara-C':
                    uA += dosage * 2
                elif drug == 'Daunorubicin':
                    uD += dosage
        return uA, uD

    # 使用此函数预处理数据
    dosage_timeline = preprocess_dosage_data(patient_data2)

    # 定义时间范围
    t_start = 0
    t_end = 30  # 35
    num_points = 100
    t_eval = np.linspace(t_start, t_end, num_points)

    def ode_system(x, t, p, get_u):
        # x 是一个包含 x1, x2, ..., xn 的向量
        # p 是一个包含参数的向量
        xpr, xtr1, xtr2, xtr3, xma = x
        B, gamma, ktr, slopeA, slopeD = p

        # 计算 Cv 值
        MMArac = 243.217
        MMDaun = 527.52  # 543.519
        Vc = 37.33
        BSA = 1.78  # 范围[1.61, 2.07]
        dur = 1  # day
        CvA = 1 / (Vc * MMArac)
        CvD = 1 / (Vc * MMDaun)
        uA, uD = get_u(t, dosage_timeline)
        # 根据 uA 和 uD 分别计算 E 值
        x1A = uA * BSA / dur
        x1D = uD * BSA / dur
        EA = slopeA * x1A * CvA
        ED = slopeD * x1D * CvD

        # E 的总和是两种药物效果的总和
        E = EA + ED
        dur = 1  # day

        # 确保 B/xma 非负，避免无效的幂运算
        ratio = max(B / max(xma, 1e-8), 0)

        # 计算各个方程的导数
        dxpr_dt = ktr * xpr * (1 - E) * ratio ** gamma - ktr * xpr
        dxtr1_dt = ktr * (xpr - xtr1)
        dxtr2_dt = ktr * (xtr1 - xtr2)
        dxtr3_dt = ktr * (xtr2 - xtr3)
        dxma_dt = ktr * xtr3 - kma * xma

        # 返回导数向量
        return [dxpr_dt, dxtr1_dt, dxtr2_dt, dxtr3_dt, dxma_dt]

    # 使用 p 中的参数
    p = B, gamma, ktr, slopeA, slopeD
    Bbm = B * kma / ktr

    # 更新 x0
    x0_updated = [Bbm, Bbm, Bbm, Bbm, B0]

    # 使用 solve_ivp 求解 ODE 系统
    solution = solve_ivp(lambda t, x: ode_system(x, t, p, get_u), (t_start, t_end), x0_updated, method='RK45')

    # 提取我们关心的变量（例如 xma）
    xma_values_at_timeWBC = np.interp(timeWBC, solution.t, solution.y[-1])

    relative_error = np.sum(np.abs(xma_data - xma_values_at_timeWBC) / xma_data) / len(xma_data)
    # # 计算相对误差
    # relative_error = np.sum(((xma_values_at_timeWBC - xma_data) / np.maximum(np.abs(xma_data), 1e-8)) ** 2)
    # # 计算归一化的均方误差（NMSE）
    # nmse = relative_error / len(xma_data)

    # # 计算Root Mean Squared Error均方根误差
    rmse = np.sqrt(np.mean((xma_values_at_timeWBC - xma_data) ** 2))
    # 计算NRMSE(Normalized Root Mean Squared Error)
    y_max = np.max(xma_data)
    y_min = np.min(xma_data)
    nrmse = rmse / (y_max - y_min)

    return patient_id, relative_error,nrmse

def analyze_and_optimize(args):
    patient_id, B, gamma, ktr, slopeA, slopeD = args
    # 直接使用全局变量 data1 和 data2
    global data1, data2
    return calculate_error_for_patient(patient_id, B, gamma, ktr, slopeA, slopeD, data1, data2)

merged_patient_data = "optimized_parameters_and_R2_WBC.csv"  # 请使用您的CSV文件路径替换
optimized_params = pd.read_csv(merged_patient_data, sep=',')

def main():
    patient_data_list = optimized_params[['patient_id',	'B_opt',	'gamma_opt',	'ktr_opt',	'slopeA_opt',	'slopeD_opt']].values.tolist()

    results = []
    with mp.Pool(processes=min(90, mp.cpu_count()-1)) as pool:  # 使用服务器的核心数或90个进程，取较小值
        for result in tqdm(pool.imap_unordered(analyze_and_optimize, patient_data_list), total=len(patient_data_list)):
            results.append(result)

    # 将病人编号取整
    results = [(int(patient_id), error, nrmse) for patient_id, error, nrmse in results]

    # 将结果写入新的CSV文件
    with open('relative_error_WBC.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['patient_id', 'relative_error','nrmse'])
        for result in results:
            writer.writerow(result)


if __name__ == '__main__':
    main()