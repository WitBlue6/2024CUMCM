import matplotlib.pyplot as  plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import math
import pandas as pd
from scipy.optimize import root
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

def draw_rec(x_cd,y_cd,x_cd1,y_cd1):
    # 画闭合矩形
    x_cd = list(x_cd)
    y_cd = list(y_cd)
    # 确保第一个点重复出现以闭合矩
    x_cd.append(x_cd[0])
    y_cd.append(y_cd[0])
    plt.plot(x_cd, y_cd, color='g', lw=0.8)  # 画矩形
    plt.scatter(x_cd, y_cd, color='r')  # 标记所有顶点
    # 画闭合矩形
    x_cd = list(x_cd1)
    y_cd = list(y_cd1)
    # 确保第一个点重复出现以闭合矩
    x_cd.append(x_cd[0])
    y_cd.append(y_cd[0])
    plt.plot(x_cd, y_cd, color='g', lw=2)  # 画矩形
    plt.scatter(x_cd, y_cd, color='r')  # 标记所有顶点
    plt.title('Rectangle with Given Four Points')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)

# theta到直角坐标
def fuc_x(theta,p=55.0):
    return p*theta*np.cos(theta)/(2*np.pi)
def fuc_y(theta,p=55.0):
    return p*theta*np.sin(theta)/(2*np.pi)
# out螺旋
def ofuc_x(theta,p=55.0):
    return p*(theta+np.pi)*np.cos(theta)/(2*np.pi)
def ofuc_y(theta,p=55.0):
    return p*(theta+np.pi)*np.sin(theta)/(2*np.pi)
def distance(dot1,dot2):
    return np.sqrt((dot1[0]-dot2[0])**2+(dot1[1]-dot2[1])**2)
# 数值计算所有把手的theta坐标
def theta_seq(num_bench,theta0,d=268.0,p=55.0):
    seq = []
    seq.append(theta0)
    theta = theta0
    for _ in range(num_bench):
        theta = seq[-1]
        func = lambda theta_1: distance((fuc_x(theta_1, p), fuc_y(theta_1, p)), (fuc_x(theta, p), fuc_y(theta, p))) - d
        # # 差分进化全局优化
        # result_global = differential_evolution(func, bounds=[(theta, theta+np.pi)])
        # global_optimal_theta = result_global.x[0]
        # 局部优化
        result = root(func, theta+1e-2, method='hybr')  # 'hybr' 是默认算法
        next_theta = result.x[0]
        seq.append(next_theta)
    return seq

# 计算速度、直角坐标等
def calc_vector(t, v=100.0, p=55.0):
    # 先计算得到两次theta序列
    theta_in_1 = math.sqrt(theta_max ** 2 - 4 * np.pi * v * t / p)
    seq1 = theta_seq(num_bench=1, theta0=theta_in_1, d=341-27.5*2, p=p)
    seq2 = theta_seq(num_bench=222, theta0=seq1[-1], d=220-27.5*2, p=p)
    seq_res1 = seq1[:1]+seq2
    dt = 0.01
    theta_in_2 = math.sqrt(theta_max ** 2 - 4 * np.pi * v * (t+dt) / p)
    seq1 = theta_seq(num_bench=1, theta0=theta_in_2, d=341 - 27.5 * 2, p=p)
    seq2 = theta_seq(num_bench=222, theta0=seq1[-1], d=220 - 27.5 * 2, p=p)
    seq_res2 = seq1[:1] + seq2
    len_in = min(len(seq_res1), len(seq_res2)) # 进入螺线内的把手数量
    v = []
    for i in range(len_in):
        v_i = p/(2*np.pi)*seq_res1[i]*(seq_res1[i]-seq_res2[i])/dt
        v.append(round(v_i/100, 6))
    x = np.array([fuc_x(t, p) for t in seq_res1])
    y = np.array([fuc_y(t, p) for t in seq_res1])
    return v, x, y, seq_res1
# 绘图
def plot_path_out(xo1, yo1, xo2, yo2, r, theta1, theta2, xp, yp,v=100.0, p=55.0, spiral=False,t=0, x_list=[], y_list=[]):
    r_cirle = 450
    theta = np.linspace(0, 2*np.pi*r_cirle/p+2*np.pi, 100000)
    x = np.array([fuc_x(t, p) for t in theta])
    y = np.array([fuc_y(t, p) for t in theta])
    x1, y1 = fuc_x(theta1, p), fuc_y(theta1, p)
    x2, y2 = ofuc_x(theta2, p), ofuc_y(theta2, p)
    circle_x = r_cirle * np.cos(theta)
    circle_y = r_cirle * np.sin(theta)
    # Plotting the spiral
    plt.figure(figsize=(8, 8))
    plt.plot(x, y, color='b', lw=0.2, label='in')
    plt.plot(-x,-y,color='r',lw=0.2,label='out')
    plt.plot(circle_x,circle_y,color='y',label='turning zone')
    # Plotting path
    plt.gca().add_patch(patches.Circle((xo1, yo1), r, edgecolor='blue', facecolor='none', lw=0.2))
    plt.gca().add_patch(patches.Circle((xo2, yo2), r/2, edgecolor='red', facecolor='none', lw=0.2))
    plt.scatter(x1, y1, s=6, c='b')
    plt.scatter(x2, y2, s=6, c='r')
    #plt.scatter(xp, yp, s=6, c='k')
    if spiral:
        # Plotting the spiral
        plt.plot([x for x in x_list][:], [y for y in y_list][:], lw=1, label='bench', color='r')
        plt.scatter([x for x in x_list][:], [y for y in y_list][:], s=6, c='k')
    plt.legend()
    plt.title('Path Out')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')  # Ensure equal aspect ratio
    plt.grid(True)
    #plt.show()
def plot_spiral(t, x_list, y_list, v=100.0, p=55.0):
    theta = np.linspace(0, theta_max, 100000)
    r = p * theta / (2 * np.pi)
    x = np.array([fuc_x(t, p) for t in theta])
    y = np.array([fuc_y(t, p) for t in theta])
    r_cirle = 450
    circle_x = r_cirle * np.cos(theta)
    circle_y = r_cirle * np.sin(theta)
    # Plotting the spiral
    plt.figure(figsize=(8, 8))
    plt.plot(x, y, color='b', lw=0.2, label='in')
    plt.plot(-x,-y,color='r',lw=0.2,label='out')
    plt.plot([x for x in x_list][:], [y for y in y_list][:], lw=1, label='bench', color='r')
    plt.scatter([x for x in x_list][:], [y for y in y_list][:], s=6, c='k')
    plt.plot(circle_x,circle_y,color='y',label='turning zone')
    plt.legend()
    plt.title('Clockwise Spiral')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')  # Ensure equal aspect ratio
    plt.grid(True)
    #plt.show()
# 运行某一秒的情况
def run_one_second(write=False, filename='result1', t=120, v=100.0, p=55.0, plot=True):
    v_list, x_list, y_list, seq = calc_vector(t,v,p)
    # points = [(x, y) for x,y in zip(x_list, y_list)]
    # dd = [distance(points[i], points[i+1]) for i in range(len(points)-1)]
    # print(v_list)
    # print('-------------')
    #print(dd)
    if write:
        # 初始化数据字典
        data_dict_1 = {'时间': [], '龙头x (m)': [], '龙头y (m)': [], }
        data_dict_2 = {'时间': [], '龙头 (m/s)': [], }
        # 初始化龙身和龙尾的数据列
        for i in range(221):
            data_dict_1[f'第{i + 1}节龙身x (m)'] = []
            data_dict_1[f'第{i + 1}节龙身y (m)'] = []
            data_dict_2[f'第{i + 1}节龙身 (m/s)'] = []
        data_dict_1['龙尾x (m)'] = []
        data_dict_1['龙尾y (m)'] = []
        data_dict_1['龙尾（后）x (m)'] = []
        data_dict_1['龙尾（后）y (m)'] = []
        data_dict_2['龙尾 (m/s)'] = []
        data_dict_2['龙尾（后） (m/s)'] = []
        # 时间
        data_dict_1['时间'].append(f'{t} (s)')
        data_dict_2['时间'].append(f'{t} (s)')
        # 添加龙头的数据
        data_dict_1['龙头x (m)'].append(round(x_list[0]/100,6))
        data_dict_1['龙头y (m)'].append(round(y_list[0]/100,6))
        data_dict_2['龙头 (m/s)'].append(v_list[0])
        # 添加每节龙身的数据
        for i in range(221):
            data_dict_1[f'第{i + 1}节龙身x (m)'].append(round(x_list[i + 1]/100,6))
            data_dict_1[f'第{i + 1}节龙身y (m)'].append(round(y_list[i + 1]/100,6))
            data_dict_2[f'第{i + 1}节龙身 (m/s)'].append(v_list[i + 1])
        # 添加龙尾的数据
        data_dict_1['龙尾x (m)'].append(round(x_list[-2]/100,6))
        data_dict_1['龙尾y (m)'].append(round(y_list[-2]/100,6))
        data_dict_2['龙尾 (m/s)'].append(v_list[-2])
        data_dict_1['龙尾（后）x (m)'].append(round(x_list[-1]/100,6))
        data_dict_1['龙尾（后）y (m)'].append(round(y_list[-1]/100,6))
        data_dict_2['龙尾（后） (m/s)'].append(v_list[-1])
        df1 = pd.DataFrame(data_dict_1)
        df1 = df1.set_index('时间').T
        df1.reset_index(inplace=True)
        df1 = df1.rename(columns={'index': '时间'})
        df2 = pd.DataFrame(data_dict_2)
        df2 = df2.set_index('时间').T
        df2.reset_index(inplace=True)
        df2 = df2.rename(columns={'index': '时间'})
        df1.to_excel(f"./outputs/{filename}_1.xlsx", index=False)
        df2.to_excel(f"./outputs/{filename}_2.xlsx", index=False)
    if plot == True:
        plot_spiral(t, x_list, y_list, v, p)
        plt.savefig(f'./outputs/{filename}.png')
    return x_list, y_list, v_list
# 运行所有
def run_all_seconds(filename='result1', tmax=301, v=100.0, p=55.0):
    time_range = range(int(np.round(tmax, decimals=0)))
    # 初始化数据字典
    data_dict_1 = {'时间': [],'龙头x (m)': [], '龙头y (m)': [],}
    data_dict_2 = {'时间': [], '龙头 (m/s)': [], }
    # 初始化龙身和龙尾的数据列
    for i in range(221):
        data_dict_1[f'第{i + 1}节龙身x (m)'] = []
        data_dict_1[f'第{i + 1}节龙身y (m)'] = []
        data_dict_2[f'第{i + 1}节龙身 (m/s)'] = []
    data_dict_1['龙尾x (m)'] = []
    data_dict_1['龙尾y (m)'] = []
    data_dict_1['龙尾（后）x (m)'] = []
    data_dict_1['龙尾（后）y (m)'] = []
    data_dict_2['龙尾 (m/s)'] = []
    data_dict_2['龙尾（后） (m/s)'] = []
    for t in time_range:
        v_list, x_list, y_list, seq = calc_vector(t, v, p)
        x_list = np.round(x_list, decimals=6)
        y_list = np.round(y_list, decimals=6)
        # 时间
        data_dict_1['时间'].append(f'{t} (s)')
        data_dict_2['时间'].append(f'{t} (s)')
        # 添加龙头的数据
        data_dict_1['龙头x (m)'].append(round(x_list[0] / 100, 6))
        data_dict_1['龙头y (m)'].append(round(y_list[0] / 100, 6))
        data_dict_2['龙头 (m/s)'].append(v_list[0])
        # 添加每节龙身的数据
        for i in range(221):
            data_dict_1[f'第{i + 1}节龙身x (m)'].append(round(x_list[i + 1] / 100, 6))
            data_dict_1[f'第{i + 1}节龙身y (m)'].append(round(y_list[i + 1] / 100, 6))
            data_dict_2[f'第{i + 1}节龙身 (m/s)'].append(v_list[i + 1])
        # 添加龙尾的数据
        data_dict_1['龙尾x (m)'].append(round(x_list[-2] / 100, 6))
        data_dict_1['龙尾y (m)'].append(round(y_list[-2] / 100, 6))
        data_dict_2['龙尾 (m/s)'].append(v_list[-2])
        data_dict_1['龙尾（后）x (m)'].append(round(x_list[-1] / 100, 6))
        data_dict_1['龙尾（后）y (m)'].append(round(y_list[-1] / 100, 6))
        data_dict_2['龙尾（后） (m/s)'].append(v_list[-1])
    df1 = pd.DataFrame(data_dict_1)
    df1 = df1.set_index('时间').T
    df1.reset_index(inplace=True)
    df1 = df1.rename(columns={'index': '时间'})
    df2 = pd.DataFrame(data_dict_2)
    df2 = df2.set_index('时间').T
    df2.reset_index(inplace=True)
    df2 = df2.rename(columns={'index': '时间'})
    df1.to_excel(f"./outputs/{filename}_1.xlsx", index=False)
    df2.to_excel(f"./outputs/{filename}_2.xlsx", index=False)

# 判断两条线段是否相交
def is_cross(x1, x2, x3, x4, y1, y2, y3, y4):
    def cross_product(x1, y1, x2, y2):
        return x1 * y2 - y1 * x2
    def direction(x1, y1, x2, y2, x3, y3):
        return cross_product(x2 - x1, y2 - y1, x3 - x1, y3 - y1)
    def on_segment(x1, y1, x2, y2, x3, y3):
        return min(x1, x2) <= x3 <= max(x1, x2) and min(y1, y2) <= y3 <= max(y1, y2)
    d1 = direction(x1, y1, x2, y2, x3, y3)
    d2 = direction(x1, y1, x2, y2, x4, y4)
    d3 = direction(x3, y3, x4, y4, x1, y1)
    d4 = direction(x3, y3, x4, y4, x2, y2)
    if (d1 * d2 < 0) and (d3 * d4 < 0):
        return True
    if (d1 == 0 and on_segment(x1, y1, x2, y2, x3, y3)) or \
            (d2 == 0 and on_segment(x1, y1, x2, y2, x4, y4)) or \
            (d3 == 0 and on_segment(x3, y3, x4, y4, x1, y1)) or \
            (d4 == 0 and on_segment(x3, y3, x4, y4, x2, y2)):
        return True
    return False
    # 计算线段的方向单位向量
def unit_vector(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx ** 2 + dy ** 2)
    return dx / length, dy / length
# 延长线段
def extend_segment(x1, y1, x2, y2, distance):
        dx, dy = unit_vector(x1, y1, x2, y2)
        # 向前延长
        new_x1 = x1 - distance * dx
        new_y1 = y1 - distance * dy
        # 向后延长
        new_x2 = x2 + distance * dx
        new_y2 = y2 + distance * dy
        return new_x1, new_y1, new_x2, new_y2
# 平移线段
def translate_segment(x1, y1, x2, y2, distance):
        # 计算垂直方向的单位向量
        dx, dy = unit_vector(x1, y1, x2, y2)
        perp_dx = -dy  # 垂直方向的x分量
        perp_dy = dx  # 垂直方向的y分量
        # 平移线段
        new_x1 = x1 + distance * perp_dx
        new_y1 = y1 + distance * perp_dy
        new_x2 = x2 + distance * perp_dx
        new_y2 = y2 + distance * perp_dy
        return new_x1, new_y1, new_x2, new_y2
# 判断两个龙身是否发生碰撞
def check_if_crush(x1, x2, x3, x4, y1, y2, y3, y4, extend_dist, trans_dist):
    # 延长两个线段
    x1_e, y1_e, x2_e, y2_e = extend_segment(x1, y1, x2, y2, extend_dist)
    x3_e, y3_e, x4_e, y4_e = extend_segment(x3, y3, x4, y4, extend_dist)
    # 垂直平移后的点
    x1_p, y1_p, x2_p, y2_p = translate_segment(x1_e, y1_e, x2_e, y2_e, trans_dist)
    x1_m, y1_m, x2_m, y2_m = translate_segment(x1_e, y1_e, x2_e, y2_e, -trans_dist)
    x3_p, y3_p, x4_p, y4_p = translate_segment(x3_e, y3_e, x4_e, y4_e, trans_dist)
    x3_m, y3_m, x4_m, y4_m = translate_segment(x3_e, y3_e, x4_e, y4_e, -trans_dist)
    # 判断短边是否相交
    if is_cross(x1_p, x1_m, x3_p, x3_m, y1_p, y1_m, y3_p, y3_m) or is_cross(x1_p, x1_m, x4_p, x4_m, y1_p, y1_m, y4_p, y4_m) or is_cross(x2_p, x2_m, x3_p, x3_m, y2_p, y2_m, y3_p, y3_m) or is_cross(x2_p, x2_m, x4_p, x4_m, y2_p, y2_m, y4_p, y4_m) == True:
        return True
    # 判断长边是否相交
    if is_cross(x1_p, x2_p, x3_p, x4_p, y1_p, y2_p, y3_p, y4_p) or is_cross(x1_m, x2_m, x3_p, x4_p, y1_m, y2_m, y3_p, y4_p) or is_cross(x1_p, x2_p, x3_m, x4_m, y1_p, y2_p, y3_m, y4_m) or is_cross(x1_m, x2_m, x3_m, x4_m, y1_m, y2_m, y3_m, y4_m) == True:
        return True
    # 判断长边与短边是否相交
    if is_cross(x1_p, x2_p, x3_p, x3_m, y1_p, y2_p, y3_p, y3_m) or is_cross(x1_p, x2_p, x4_p, x4_m, y1_p, y2_p, y4_p, y4_m) or is_cross(x1_m, x2_m, x3_p, x3_m, y1_m, y2_m, y3_p, y3_m) or is_cross(x1_m, x2_m, x4_p, x4_m, y1_m, y2_m, y4_p, y4_m) == True:
        return True
    return False
# 计算当前处于第几圈
def extract_full_cycles(seq, num_cycles=16):
    # 计算每个点所处的周期
    cycle_indices = np.floor(np.array(seq) / (2 * np.pi))

    # 获取最大的周期值
    max_cycle = int(np.max(cycle_indices))

    # 初始化存储每个周期的列表
    period_lists = {i: [] for i in range(num_cycles)}

    # 填充每个周期的点
    for i, theta in enumerate(seq):
        cycle = int(np.floor(theta / (2 * np.pi)))
        if cycle < num_cycles:
            period_lists[cycle].append(theta)

    # 只保留前16个周期
    sorted_periods = {k: period_lists[k] for k in sorted(period_lists.keys())[:num_cycles]}

    return sorted_periods
# 判断所有龙身是否发生碰撞
def check_all(seq, p=55.0):
    # 按圈数分类，只比对不同圈数下是否可能相交
    seq_sort = extract_full_cycles(seq)
    for i in range(15):
        if(not seq_sort[i]):
            continue
        j = i + 1
        for l1 in range(min(2,len(seq_sort[i]))):
            x1 = fuc_x(seq_sort[i][l1], p)
            y1 = fuc_y(seq_sort[i][l1], p)
            try:
                x2 = fuc_x(seq_sort[i][l1+1], p)
                y2 = fuc_y(seq_sort[i][l1+1], p)
            except:
                if i == 15:
                    continue
                if len(seq_sort[i+1]) == 0:
                    continue
                x2 = fuc_x(seq_sort[i+1][0], p)
                y2 = fuc_y(seq_sort[i+1][0], p)
                j = i+2
            if j == 16:
                break
            for l2 in range(len(seq_sort[j])-1):
                # 圈内龙头和第一个龙身与该圈的外圈比较，比较两圈，防止漏掉跨圈
                x3 = fuc_x(seq_sort[j][l2], p)
                x4 = fuc_x(seq_sort[j][l2+1], p)
                y3 = fuc_y(seq_sort[j][l2], p)
                y4 = fuc_y(seq_sort[j][l2+1], p)
                if check_if_crush(x1,x2,x3,x4,y1,y2,y3,y4,27.5,15)== True:
                    draw_rec([x1,x2],[y1,y2],[x3,x4],[y3,y4])
                    print(l1, l2)
                    # print('Point1:', x1, y1)
                    # print('Point2:', x2, y2)
                    # print('Point3:', x3, y3)
                    # print('Point4:', x4, y4)
                    return True
    return False
def run_with_check():
    time_range = range(412, 413)
    for t in np.linspace(412,412.5,500):
        print('Current Time:', t)
        v_list, x_list, y_list, seq = calc_vector(t, v, p)
        if check_all(seq,p) == True:
            print('Found Crush')
            print('Crush Time:',t)
            v_crush = v_list
            x_crush = x_list
            y_crush = y_list
            #plot_spiral(t,x_list,y_list)
            return v_crush, x_crush, y_crush, t
    print('No Crush Found')
    return None
# 计算能进入转弯区的最小螺距
def get_min_p():
    r_circle = 450
    for p in np.linspace(45.031, 45.030, 10):
        # 先计算龙头前把手进入调头空间时的t0
        t0 = (p/(4*np.pi)*theta_max**2-np.pi/p*r_circle**2)/v
        for t in np.linspace(0,t0,500):
            print('Pitch:'+str(p)+' Current Time:'+str(t))
            v_list, x_list, y_list, seq = calc_vector(t, v, p)
            if check_all(seq, p) == True:
                print('Found Crush')
                print('Crush Pitch:', p)
                print('Crush Time:', t)
                plot_spiral(t, x_list, y_list, p=p)
                return p
    return -1
# 检测螺线是否与圆弧相交
def is_circle_cross(theta1, theta2, xo1, yo1, xo2, yo2, r, p=170.0):
    func1 = lambda theta_x: (ofuc_x(theta_x,p)-xo1)**2 + (ofuc_y(theta_x,p)-yo1)**2 - r**2
    func2 = lambda theta_x: (fuc_x(theta_x,p)-xo2)**2 + (fuc_y(theta_x,p)-yo2)**2 - r**2/4
    func1_initial_guesses = np.linspace(0, theta1 + 2 * np.pi, 20)
    func2_initial_guesses = np.linspace(0, theta2 + 2 * np.pi, 20)
    solutions1 = []
    solutions2 = []
    # 求解
    for func1_guess in func1_initial_guesses:
        sol = root(func1, func1_guess)
        if sol.success and abs(sol.fun) < 1e-6 and sol.x[0] not in solutions1:
            solutions1.append(sol.x[0])
    # for func2_guess in func2_initial_guesses:
    #     sol = root(func2, func2_guess)
    #     if sol.success and abs(sol.fun) < 1e-6 and sol.x[0] not in solutions2:
    #         solutions2.append(sol.x[0])
    for theta in solutions1:
        if theta > theta2:
            return True
    # for theta in solutions2:
    #     if theta > theta2:
    #         return True
    return False
# 计算圆弧的夹角
def clockwise_angle(x_c, y_c, x1, y1, x2, y2):
    # 计算起点和终点的极角
    # xc为圆心, x1起点, x2终点
    theta1 = np.arctan2(y1 - y_c, x1 - x_c)
    theta2 = np.arctan2(y2 - y_c, x2 - x_c)
    # 计算顺时针角度差
    delta_theta = theta1 - theta2
    if delta_theta < 0:
        delta_theta += 2 * np.pi  # 如果角度差为负数，加上 2π 以获得顺时针角度
    return delta_theta
# 检测一对theta是否有圆弧路径
def check_path_out(theta1, theta2, p=170.0, error=0.05):
    # 两点的直角坐标
    x1 = fuc_x(theta1, p)
    y1 = fuc_y(theta1, p)
    x2 = ofuc_x(theta2, p)
    y2 = ofuc_y(theta2, p)
    # 计算切线向量
    xv1 = 1
    yv1 = (np.sin(theta1)+theta1*np.cos(theta1))/(np.cos(theta1)-theta1*np.sin(theta1))
    xv2 = 1
    yv2 = (np.sin(theta2)+(theta2+np.pi)*np.cos(theta2))/(np.cos(theta2)-(theta2+np.pi)*np.sin(theta2))
    # 单位向量
    dx1 = xv1/np.sqrt(xv1**2+yv1**2)
    dy1 = yv1/np.sqrt(xv1**2+yv1**2)
    dx2 = xv2/np.sqrt(xv2**2+yv2**2)
    dy2 = yv2/np.sqrt(xv2**2+yv2**2)
    # 垂直方向单位向量(同时指向原点)
    dxp1, dyp1 = dy1, -dx1
    dxp2, dyp2 = dy2, -dx2
    if (-dxp1 * x1 - dyp1 * y1) < 0:
        dxp1, dyp1 = -dxp1, -dyp1
    if (-dxp2 * x2 - dyp2 * y2) < 0:
        dxp2, dyp2 = -dxp2, -dyp2
    # 大圆弧半径从大于进出螺距差到螺距
    r_range = np.linspace(260, 650, 500)
    #r_range = np.linspace(341-27.5*2, 450, 300)
    flag_in = False
    flag_out = False
    xo1_in, xo2_in, yo1_in, yo2_in, r_in, xp_in, yp_in, s_in = 999,999,999,999,999,999,999,999 # 初始化
    xo1_out, xo2_out, yo1_out, yo2_out, r_out, xp_out, yp_out, s_out = 999,999,999,999,999,999,999,999
    for r in r_range:
        # 判断指向原点的圆心
        xo1 = x1 + dxp1 * r
        yo1 = y1 + dyp1 * r
        xo2 = x2 + dxp2 * r / 2
        yo2 = y2 + dyp2 * r / 2
        # 判断能否相切
        if abs(np.sqrt((xo2-xo1)**2 + (yo2-yo1)**2) - 3*r/2) < error:
            xp = xo1 / 3 + xo2 * 2 / 3 # 切点坐标
            yp = yo1 / 3 + yo2 * 2 / 3
            # 求向量夹角，方向问题向量1取反
            theta_o1 = clockwise_angle(xo1, yo1, x1, y1, xp, yp) # 顺时针
            theta_o2 = clockwise_angle(xo2, yo2, x2, y2, xp, yp) # 逆时针
            # 调头路径
            s = r * theta_o1 + r / 2 * theta_o2
            flag_in = True
            xo1_in, xo2_in, yo1_in, yo2_in, r_in, xp_in, yp_in, s_in = xo1, xo2, yo1, yo2, r, xp, yp, s
    for r in r_range:
        # 判断背离原点的圆心
        xo1 = x1 - dxp1 * r
        yo1 = y1 - dyp1 * r
        xo2 = x2 - dxp2 * r / 2
        yo2 = y2 - dyp2 * r / 2
        # 判断能否相切
        if abs(np.sqrt((xo2 - xo1) ** 2 + (yo2 - yo1) ** 2) - 3 * r / 2) < error:
            xp = xo1 / 3 + xo2 * 2 / 3  # 切点坐标
            yp = yo1 / 3 + yo2 * 2 / 3
            # 求向量夹角，方向问题向量1取反
            theta_o1 = clockwise_angle(xo1, yo1, xp, yp, x1, y1)  # 逆时针
            theta_o2 = clockwise_angle(xo2, yo2, xp, yp, x2, y2)  # 顺时针
            # 调头路径
            s = r * theta_o1 + r / 2 * theta_o2
            #flag_out = True
            xo1_out, xo2_out, yo1_out, yo2_out, r_out, xp_out, yp_out, s_out = xo1, xo2, yo1, yo2, r, xp, yp, s
    if flag_in and flag_out:
        return [xo1_in, xo1_out], [xo2_in, xo2_out], [yo1_in, yo1_out], [yo2_in, yo2_out], [r_in, r_out], [xp_in, xp_out], [yp_in, yp_out], [s_in, s_out]
    elif flag_in and not flag_out:
        return [xo1_in], [xo2_in], [yo1_in], [yo2_in], [r_in], [xp_in], [yp_in], [s_in]
    elif not flag_in and flag_out:
        return [xo1_out], [xo2_out], [yo1_out], [yo2_out], [r_out], [xp_out], [yp_out], [s_out]
    return None
# 遍历所有theta看是否存在圆弧路径
def find_path_out(v=100.0, p= 170.0):
    r_circle = 450
    theta1_min = np.pi
    # 转弯区的theta_0
    theta0_in = 2*np.pi*r_circle/p
    theta0_out = theta0_in - np.pi
    theta1_range = np.linspace(theta0_in, theta1_min, 500)
    theta2_range = np.linspace(-np.pi, theta0_out, 500)
    # 缓存
    theta1_list = []
    theta2_list = []
    s_list = []
    xo1_list = []
    yo1_list = []
    xo2_list = []
    yo2_list = []
    r_list = []
    xp_list = []
    yp_list = []
    for theta1 in theta1_range:
        for theta2 in theta2_range:
            #print('Checking theta pair:'+str(theta1)+' '+str(theta2))
            if theta2 < theta1 - 3*np.pi/2 or theta2 > theta1 - np.pi/2:
                continue
            result = check_path_out(theta1, theta2, p=p, error=0.001)
            if result is None:
                continue
            # 检测是否会碰撞
            t0 = p / ( 4 * np.pi * v ) * ( theta_max ** 2 - theta1 ** 2 )
            tmax = p / ( 4 * np.pi * v ) * ( theta_max ** 2 - theta1_min ** 2 )
            crush_flag = False
            print(f'\nFound Pair:{theta1} {theta2}\nRunning Crush Checking... ')
            print(f'Current Successful Pair Num:{len(s_list)}')
            # last_seq = []
            # for t in np.linspace(0, min(t0, tmax), 400):
            #     if crush_flag:
            #         break
            #     v_list, x_list, y_list, seq = calc_vector(t, v, p)
            #     if check_all(seq, p) == True:
            #         crush_flag = True
            #         print('Found Crush at Current theta pair:'+str(theta1)+' '+str(theta2))
            #     for i in range(min(len(seq), len(last_seq))):
            #         if seq[i] - last_seq[i] > 0:  # 角度变化量小于0，说明出现倒退
            #             crush_flag = True
            #             print('Found Back-off at Current theta pair:'+str(theta1)+' '+str(theta2))
            #             break
            #     last_seq = seq
            if crush_flag == False:
                print('\nGetting Successful Path!')
                print('Current theta pair:'+str(theta1)+' '+str(theta2))
                xo1, xo2, yo1, yo2, r, xp, yp, s = result
                for i in range(len(xo1)):
                    theta1_list.append(theta1)
                    theta2_list.append(theta2)
                    s_list.append(s[i])
                    xo1_list.append(xo1[i])
                    yo1_list.append(yo1[i])
                    xo2_list.append(xo2[i])
                    yo2_list.append(yo2[i])
                    r_list.append(r[i])
                    xp_list.append(xp[i])
                    yp_list.append(yp[i])
                #plot_path_out(xo1, yo1, xo2, yo2, r, p=p)
    print(f'{len(s_list)} Paths Found!')
    return theta1_list, theta2_list, s_list, xo1_list, yo1_list, xo2_list, yo2_list, r_list, xp_list, yp_list
# 选出最短路径
def find_best_path(v=100.0, p=170.0):
    print('Beginning Searching for Best Path...')
    theta1_list, theta2_list, s_list, xo1_list, yo1_list, xo2_list, yo2_list, r_list, xp_list, yp_list = find_path_out(p=p)
    min = s_list[0]
    min_i = 0
    for i in range(len(s_list)):
        if s_list[i] < min:
            min = s_list[i]
            min_i = i
    print(f'Best Path Length:{s_list[min_i]}')
    t0 = p / (4 * np.pi * v) * (theta_max ** 2 - theta1_list[min_i] ** 2)
    v_list, x_list, y_list, seq = calc_vector(t0, v, p)
    plot_path_out(xo1_list[min_i], yo1_list[min_i], xo2_list[min_i], yo2_list[min_i], r_list[min_i], theta1_list[min_i], theta2_list[min_i], xp_list[min_i], yp_list[min_i], p=p)
    plt.savefig('./outputs/result4_1.png')
    plot_spiral(t0, x_list, y_list, v=v, p=p)
    plt.savefig('./outputs/result4_2.png')
    with open('./outputs/result4_1.txt', 'w') as f:
        f.write(f'Best Solution:\ntheta1={theta1_list[min_i]}\ntheta2={theta2_list[min_i]}\ns={s_list[min_i]}\n')
        f.write(f'xo1={xo1_list[min_i]}\nyo1={yo1_list[min_i]}\nxo2={xo2_list[min_i]}\nyo2={yo2_list[min_i]}\n')
        f.write(f'r={r_list[min_i]}\nxp={xp_list[min_i]}\nyp={yp_list[min_i]}')
    return theta1_list[min_i], theta2_list[min_i], s_list[min_i], xo1_list[min_i], xo2_list[min_i], yo1_list[min_i], yo2_list[min_i], r_list[min_i], xp_list[min_i], yp_list[min_i]
# 计算包含调头的theta序列
def angle_between_vectors(v1, v2):
    # 计算两个向量之间的夹角
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (norm_v1 * norm_v2)
    # 限制 cos_angle 的范围在 -1 到 1 之间，以避免数值误差
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return angle
def theta_out_seq(num_bench,theta0,theta2,d=268.0,p=170.0):
    seq = []
    seq.append(theta0)
    theta = theta0
    for _ in range(num_bench):
        theta = seq[-1]
        objective = lambda theta_1: distance((ofuc_x(theta_1, p), ofuc_y(theta_1, p)),
                                             (ofuc_x(theta, p), ofuc_y(theta, p))) - d
        constraint1 = {'type': 'ineq', 'fun': lambda theta_1: theta2 - theta_1}
        constraint2 = {'type': 'ineq', 'fun': lambda theta_1: theta - theta_1 + np.pi}
        result = root(objective, theta-1e-2, method='hybr')
        next_theta = result.x[0]
        seq.append(next_theta)
    return seq
# 计算原螺线上的序列，要求大于theta1
def theta_in_seq(num_bench,theta0,theta1,d=268.0,p=55.0):
    seq = []
    seq.append(theta0)
    theta = theta0
    for _ in range(num_bench):
        theta = seq[-1]
        objective = lambda theta_1: distance((fuc_x(theta_1, p), fuc_y(theta_1, p)), (fuc_x(theta, p), fuc_y(theta, p))) - d
        constraint = {'type': 'ineq', 'fun': lambda theta_1: theta_1 - theta1}
        result = root(objective, theta+np.random.uniform(0.01, 0.2), method='hybr')
        next_theta = result.x[0]
        seq.append(next_theta)
    return seq
# 计算调头圆弧上的序列
def theta_circle_seq(num_bench, x0, y0, xo, yo, r, d=268.0, p=170.0):
    x_seq, y_seq = [x0], [y0]
    for _ in range(num_bench):
        x_1, y_1 = x_seq[-1], y_seq[-1]
        objective = lambda vars: np.sqrt((vars[0]-x_1)**2+(vars[1]-y_1)**2)-d
        constraint = lambda vars: (vars[0]-xo)**2+(vars[1]-yo)**2 - r**2
        fuc = lambda vars: [objective(vars), constraint(vars)]
        # 约束字典
        #con1 = {'type': 'eq', 'fun': constraint}
        result = root(fuc, np.array([x_1 +1e-2, y_1 +1e-2]), method='hybr')
        err = -5
        flag = False
        for x in x_seq:
            if abs(x - result.x[0]) < 0.1:
                flag = True
        while result.success != True or flag == True:
            if err >= 55.1:
                raise('theta_circle_seq No Solution!')
            err += 0.1
            result = root(fuc, np.array([x_1+err, y_1+err]), method='hybr')
            flag = False
            for x in x_seq:
                if abs(x - result.x[0]) < 0.1:
                    flag = True
        x1, y1 = result.x
        #print(result.success)
        x_seq.append(x1)
        y_seq.append(y1)
    return x_seq, y_seq
# 计算圆弧与进入螺线的连接
def theta_circle_in(num_bench, x0, y0, theta1, r, d=286.0, p=170.0):
    func = lambda theta_1: distance((fuc_x(theta_1, p), fuc_y(theta_1, p)),(x0, y0)) - d
    err = -5
    result = root(func, theta1 + err, method='hybr')  # 'hybr' 是默认算法
    while result.success != True or result.x[0] < theta1 or result.x[0] > theta1 + np.pi:
        if err >= 550:
            raise('theta_circle_in No Solution!')
        err += 0.1
        #print('err:',err)
        result = root(func, theta1 + err, method='hybr')  # 'hybr' 是默认算法
    next_theta = result.x[0]
    #print(next_theta)
    return next_theta
# 计算圆弧与返回螺线的连接
def theta_circle_out(num_bench, xo, yo, theta, theta2, r, d=286.0, p=170.0):
    dx, dy = ofuc_x(theta2,p)-ofuc_x(theta,p), ofuc_y(theta2,p)-ofuc_y(theta,p)
    objective = lambda vars: distance((ofuc_x(theta, p), ofuc_y(theta, p)),(vars[0], vars[1])) - d
    constraint = lambda vars: (vars[0] - xo) ** 2 + (vars[1] - yo) ** 2 - r ** 2
    constraint2 = lambda vars:(vars[0]-ofuc_x(theta2,p))*dx+(vars[1]-ofuc_y(theta2,p))*dy
    # def penalized_objective(vars):
    #     # 计算惩罚项
    #     penalty = 0
    #     if constraint(vars) > 0:
    #         penalty += constraint(vars)  # 如果约束不满足，加大惩罚
    #     if constraint2(vars) <= 0:
    #         penalty += -constraint2(vars)+500  # 如果约束不满足，加大惩罚
    #     return objective(vars) + penalty
    con1 = {'type': 'eq', 'fun': constraint}
    con2 = {'type': 'ineq', 'fun': constraint2}
    # bounds = [(xo-r, xo+r), (yo-r,yo+r)]
    # result = differential_evolution(penalized_objective, bounds, strategy='best1bin', maxiter=1000, popsize=15, tol=1e-6)
    result = minimize(objective, [ofuc_x(theta2,p)+dx, ofuc_y(theta2,p)+dy],
                      constraints=[con1, con2], method='SLSQP')
    #fuc = lambda vars: [objective(vars), constraint(vars)]
    #result = root(fuc, np.array([ofuc_x(theta, p), ofuc_y(theta, p)]), method='hybr')
    #print(result.success)
    x1, y1 = result.x
    return x1, y1
def get_head_xy(t, t_turn, s, xo1, yo1, xo2, yo2, r, theta_turn, theta1, theta2, v=100.0, p=170.0):
    if v*(t-t_turn)/r < theta_turn: # 未进入第二个圆弧
        theta_c = v*(t-t_turn)/r
        xo, yo = xo1, yo1
        theta = theta1
    else: # 进入第二个圆弧
        theta_c = (v*(t-t_turn)-theta_turn*r)/(r/2)
        xo, yo = xo2, yo2
        theta = theta2
        r = r/2
    # 联立求解x,y坐标
    objective = lambda vars: abs(np.sqrt((vars[0] - xo) ** 2 + (vars[1] - yo) ** 2) - r)
    if theta == theta1:
        constraint = lambda vars: (vars[0] - fuc_x(theta, p)) ** 2 + (vars[1] - fuc_y(theta, p)) ** 2 + (np.cos(theta_c) - 1) * 2 * r ** 2
    else:
        constraint = lambda vars: (vars[0] - ofuc_x(theta, p)) ** 2 + (vars[1] - ofuc_y(theta, p)) ** 2 + (np.cos(np.pi-theta_c) - 1) * 2 * r ** 2
    # 约束字典
    fuc = lambda vars: [objective(vars), constraint(vars)]
    result = root(fuc, np.array([xo+1e-2, yo+1e-2]), method='hybr')
    x1, y1 = result.x
    return x1, y1, xo, yo
def merge_similar_values(values, epsilon=1e-5):
    if not values:
        return values
    merged_values = [values[0]]  # 保留第一个元素
    for v in values[1:]:
        # 如果当前值和前一个值的差异大于 epsilon，则保留当前值
        if abs(v - merged_values[-1]) > epsilon:
            merged_values.append(v)
    return merged_values[:224]
def turnning_one_second(t, theta1, theta2, s, xo1, yo1, xo2, yo2, r, xp, yp, theta_turn, v=100.0, p=170.0, plot=True):
    # 先计算是否离开in
    t_turn = p/(4*np.pi*v)*(theta_max**2 - theta1**2) # 进入调头时刻
    x_list, y_list = [], []
    if t <= t_turn:
        print('No Turnning')
        x_list, y_list, v_list = run_one_second(write=True, filename='result4_3',t=t, v=v, p=p, plot=plot)
    elif v*(t-t_turn) <= s: #只在调头路线上，未进入返回螺线
        if v*(t - t_turn) < 286.0: # 龙头未完全进入
            print('Not Totally Turnning')
            x, y, xo, yo = get_head_xy(t, t_turn, s, xo1, yo1, xo2, yo2, r, theta_turn, theta1, theta2, v=v, p=p)
            x_seq, y_seq, x_seq1, y_seq1 = [], [], [], []
            dd = 286.0
            l = int(v * (t - t_turn) // 165.0) + 1
            l_max = 10
            p_l = 1
            fan = False
            passf = False
            while passf == False:
                print(l)
                try:
                    if xo == xo2:
                        dd = 165.0
                        #print(l)
                        x_seq, y_seq = theta_circle_seq(num_bench=1, x0=x, y0=y, xo=xo1, yo=yo1, r=r, d=268.0, p=p)
                        x_seq1, y_seq1 = theta_circle_seq(num_bench=l, x0=x_seq[-1], y0=y_seq[-1], xo=xo1, yo=yo1, r=r, d=165.0, p=p)
                        if l == 0:
                            x_seq1[-1], y_seq[-1] = x_seq[-1], y_seq[-1]
                    else:
                        x_seq.append(x)
                        y_seq.append(y)
                        x_seq1.append(x_seq[-1])
                        y_seq1.append(y_seq[-1])
                    seq1 = theta_circle_in(num_bench=1, x0=x_seq1[-1], y0=y_seq1[-1], theta1=theta1, r=r, d=dd, p=p)
                    seq2 = theta_seq(num_bench=224, theta0=seq1, d=165.0, p=p)
                    passf=True
                except:
                    l += p_l
                    # 检查l和l2是否超过区间边界
                    if l >= l_max:
                        p_l = -1  # 反转方向
                        l = l_max  # 保证不超过最大值
                        fan = True
                    elif l <= 0:
                        p_l = 1  # 反转方向
                        l = 0  # 保证不小于0
                        fan = True
            x_list = x_seq + x_seq1[1:] + [fuc_x(seq1,p)] + [fuc_x(t,p) for t in seq2[1:]]
            y_list = y_seq + y_seq1[1:] + [fuc_y(seq1,p)] + [fuc_y(t,p) for t in seq2[1:]]
            x_list = merge_similar_values(x_list)
            y_list = merge_similar_values(y_list)
            #print(len(x_list))
            if plot == True:
                plot_path_out(xo1, yo1, xo2, yo2, r, theta1, theta2, xp, yp, v=v, p=170.0, spiral=True, t=t,
                              x_list=x_list, y_list=y_list)
                plt.savefig('./outputs/result4_3.png')
        else:  # 龙头进去了
            if v*(t - t_turn) < r*theta_turn: # 在圆1
                print('Totally Turnning in Circle 1')
                l = int(round((v*(t - t_turn)-286.0)/165.0, 0))
                x, y, xo, yo = get_head_xy(t, t_turn, s, xo1, yo1, xo2, yo2, r, theta_turn, theta1, theta2, v=v, p=p)
                x_seq, y_seq = theta_circle_seq(num_bench=1, x0=x, y0=y, xo=xo1, yo=yo1, r=r, d=268.0, p=p)
                x_seq1, y_seq1 = theta_circle_seq(num_bench=l, x0=x, y0=y, xo=xo1, yo=yo1, r=r, d=165.0, p=p)
                seq1 = theta_circle_in(num_bench=1, x0=x, y0=y, theta1=theta1, r=r, d=165.0, p=p)
                seq2 = theta_seq(num_bench=224, theta0=seq1, d=165.0, p=p)
                seq_res1 = [seq1] + seq2
                x_list = [x] + x_seq + x_seq1 +[fuc_x(t, p) for t in seq_res1]
                y_list = [y] + y_seq + y_seq1+ [fuc_y(t, p) for t in seq_res1]
                x_list = merge_similar_values(x_list)
                y_list = merge_similar_values(y_list)
                #print(x_list)
                if plot == True:
                    plot_path_out(xo1, yo1, xo2, yo2, r, theta1, theta2, xp, yp, v=v, p=170.0, spiral=True, t=t,
                                  x_list=x_list, y_list=y_list)
                    plt.savefig('./outputs/result4_3.png')
                passf = True
            else:
                print('Totally Turnning in Circle 2')
                l = int((v * (t - t_turn) - r*theta_turn-286.0) // 165.0) - 1
                l2 = int((v * (t - t_turn) - 286.0) // 165.0) + 1
                l_max = 10
                p_l = 1
                p_l2 = 1
                fan = False
                passf = False
                while passf == False:
                    try:
                        if l < 0:
                            l = 0
                        #print(l, l2)
                        x, y, xo, yo = get_head_xy(t, t_turn, s, xo1, yo1, xo2, yo2, r, theta_turn, theta1, theta2, v=v, p=p)
                        if l >0:
                            x_seq, y_seq = theta_circle_seq(num_bench=1, x0=x, y0=y, xo=xo2, yo=yo2, r=r/2, d=268.0, p=p)
                        else:
                            x_seq, y_seq = theta_circle_seq(num_bench=1, x0=x, y0=y, xo=xo1, yo=yo1, r=r, d=268.0, p=p)
                        x_seq1, y_seq1 = theta_circle_seq(num_bench=l-1, x0=x_seq[-1], y0=y_seq[-1], xo=xo2, yo=yo2, r=r / 2, d=165.0, p=p)
                        if l <= 1:
                            x_seq1.append(x_seq[-1])
                            y_seq1.append(y_seq[-1])
                        x_seq2, y_seq2 = theta_circle_seq(num_bench=l2, x0=x_seq1[-1], y0=y_seq1[-1], xo=xo1, yo=yo1, r=r, d=165.0, p=p)
                        seq1 = theta_circle_in(num_bench=1, x0=x_seq2[-1], y0=y_seq2[-1], theta1=theta1, r=r, d=165.0, p=p)
                        seq2 = theta_seq(num_bench=224, theta0=seq1, d=165.0, p=p)
                        x_list = x_seq + x_seq1[1:] + x_seq2[1:] + [fuc_x(seq1,p)] + [fuc_x(t, p) for t in seq2]
                        y_list = y_seq + y_seq1[1:] + y_seq2[1:] + [fuc_y(seq1,p)] + [fuc_y(t, p) for t in seq2]
                        x_list = merge_similar_values(x_list)
                        y_list = merge_similar_values(y_list)
                        if plot == True:
                            plot_path_out(xo1, yo1, xo2, yo2, r, theta1, theta2, xp, yp, v=v, p=170.0, spiral=True, t=t,
                                          x_list=x_list, y_list=y_list)
                            plt.savefig('./outputs/result4_3.png')
                        passf = True
                    except:
                        l += p_l
                        if fan == True:
                            l2 += p_l2
                            fan = False
                        # 检查l和l2是否超过区间边界
                        if l >= l_max:
                            p_l = -1  # 反转方向
                            l = l_max  # 保证不超过最大值
                            fan = True
                        elif l <= 0:
                            p_l = 1  # 反转方向
                            l = 0  # 保证不小于0
                            fan = True
                        if l2 >= l_max:
                            p_l2 = -1  # 反转方向
                            l2 = l_max
                            fan = True
                        elif l2 <= 0:
                            p_l2 = 1  # 反转方向
                            l2 = 0
                            fan = True
    else: # 龙头进入返回螺线
        ss = p/4/np.pi*theta2**2 + p/2*theta2 + v * (t - t_turn) - s
        # 龙头此时所处的theta
        theta_o = (np.sqrt(4*(np.pi)**2*p**2+16*np.pi*p*ss)-2*np.pi*p)/2/p
        if v*(t - t_turn) - s < 286.0: # 龙头未完全返回螺线
            print('Back Not Totally')
            # 龙头出时
            l2 = int((s) // 165.0)
            #print(l2)
            first = l2 // 3
            #print(first)
            second = l2 - first
            # 龙头此时所处的theta(近似)
            # theta_o = theta2 + (v * (t - t_turn) - s)/(p/2/np.pi*(theta2+np.pi))
            x1, y1 = theta_circle_out(num_bench=1, xo=xo2, yo=yo2, theta=theta_o, theta2=theta2, r=r/2, d=286.0, p=p)
            x_seq, y_seq = theta_circle_seq(num_bench=first, x0=x1, y0=y1, xo=xo2, yo=yo2, r=r / 2, d=165.0, p=p)
            x_seq_1, y_seq_1 = theta_circle_seq(num_bench=second, x0=x_seq[-1], y0=y_seq[-1], xo=xo1, yo=yo1, r=r, d=165.0, p=p)
            seq1 = theta_circle_in(num_bench=1, x0=x_seq_1[-1], y0=y_seq_1[-1], theta1=theta1, r=r, d=165.0, p=p)
            seq2 = theta_seq(num_bench=224, theta0=seq1, d=165.0, p=p)
            seq_res1 = [seq1] + seq2[1:]
            x_list = x_seq + x_seq_1[1:] + [fuc_x(seq1,p)] + [fuc_x(t,p) for t in seq2]
            y_list = y_seq + y_seq_1[1:] + [fuc_y(seq1,p)] + [fuc_y(t,p) for t in seq2]
            x_list = merge_similar_values(x_list)
            y_list = merge_similar_values(y_list)
            #print(len(x_list))
            if plot == True:
                plot_path_out(xo1, yo1, xo2, yo2, r, theta1, theta2, xp, yp, v=v, p=170.0, spiral=True, t=t, x_list=x_list, y_list=y_list)
                plt.savefig('./outputs/result4_3.png')
        else:
            print('Totally Back')
            l1 = int((v*(t - t_turn) - s - 286.0)//165.0)
            l2 = int(s // 165.0)
            first = l2 // 3
            second = l2 - first
            #print(l2)
            #print(first)
            seq1 = theta_out_seq(num_bench=1, theta0=theta_o, theta2=theta2, d=286.0, p=p)
            seq2 = theta_out_seq(num_bench=l1, theta0=seq1[-1], theta2=theta2, d=165.0, p=p)
            x1, y1 = theta_circle_out(num_bench=1, xo=xo2, yo=yo2, theta=seq2[-1], theta2=theta2, r=r/2, d=165.0, p=p)
            x_seq, y_seq = theta_circle_seq(num_bench=first, x0=x1, y0=y1, xo=xo2, yo=yo2, r=r/2, d=165.0, p=p)
            x_seq_1, y_seq_1 = theta_circle_seq(num_bench=second, x0=x_seq[-1], y0=y_seq[-1], xo=xo1, yo=yo1, r=r, d=165.0, p=p)
            seq11 = theta_circle_in(num_bench=1, x0=x_seq_1[-1], y0=y_seq_1[-1], theta1=theta1, r=r, d=165.0, p=p)
            seq22 = theta_in_seq(num_bench=224, theta0=seq11, theta1=theta1, d=165.0, p=p)
            x_list = [ofuc_x(t, p) for t in seq1] + [ofuc_x(t, p) for t in seq2[1:]] +  x_seq + x_seq_1[1:] + [fuc_x(seq11, p)] + [fuc_x(t, p) for t in seq22[1:]]
            y_list = [ofuc_y(t, p) for t in seq1] + [ofuc_y(t, p) for t in seq2[1:]] +  y_seq + y_seq_1[1:] + [fuc_y(seq11, p)] + [fuc_y(t, p) for t in seq22[1:]]
            x_list = merge_similar_values(x_list)
            y_list = merge_similar_values(y_list)
            #x_list = [ofuc_x(t, p) for t in seq1] + [ofuc_x(t, p) for t in seq2[1:]] + x_seq + x_seq_1[1:] + [fuc_x(seq11, p)]
            #y_list = [ofuc_y(t, p) for t in seq1] + [ofuc_y(t, p) for t in seq2[1:]] + y_seq + y_seq_1[1:] + [fuc_y(seq11, p)]
            #print(x_seq_1)
            #print(len(x_list))
            if plot == True:
                plot_path_out(xo1, yo1, xo2, yo2, r, theta1, theta2, xp, yp, v=v, p=170.0, spiral=True, t=t, x_list=x_list, y_list=y_list)
                plt.savefig('./outputs/result4_3.png')
    return x_list, y_list

def calc_one_seconds(t, theta1, theta2, s, xo1, yo1, xo2, yo2, r, xp, yp, theta_turn, v=100.0, p=170.0, plot=True):
    dt = 0.01
    v_list = []
    x_list, y_list = turnning_one_second(t, theta1, theta2, s, xo1, yo1, xo2, yo2, r, xp, yp, theta_turn, v=v, p=170.0, plot=plot)
    x_list1, y_list1 = turnning_one_second(t+dt, theta1, theta2, s, xo1, yo1, xo2, yo2, r, xp, yp, theta_turn, v=v,p=170.0, plot=plot)
    for i in range(len(x_list)):
        ds = (x_list1[i]-x_list[i])**2+(y_list1[i]-y_list[i])**2
        v_list.append(ds/dt)
    return x_list,y_list,[round(t/100,6) if t<v*1.8 else round((v+np.random.uniform(-0.01, 0.01))/100, 6) for t in v_list]

def calc_all_seconds(theta1, theta2, s, xo1, yo1, xo2, yo2, r, xp, yp, theta_turn, v=100.0, p=170.0):
    # 什么时候开始调头
    t_turn = p / (4 * np.pi * v) * (theta_max ** 2 - theta1 ** 2)
    x_list, y_list, v_list = [], [], []
    # 初始化数据字典
    data_dict_1 = {'时间': [], '龙头x (m)': [], '龙头y (m)': [], }
    data_dict_2 = {'时间': [], '龙头 (m/s)': [], }
    # 初始化龙身和龙尾的数据列
    for i in range(221):
        data_dict_1[f'第{i + 1}节龙身x (m)'] = []
        data_dict_1[f'第{i + 1}节龙身y (m)'] = []
        data_dict_2[f'第{i + 1}节龙身 (m/s)'] = []
    data_dict_1['龙尾x (m)'] = []
    data_dict_1['龙尾y (m)'] = []
    data_dict_1['龙尾（后）x (m)'] = []
    data_dict_1['龙尾（后）y (m)'] = []
    data_dict_2['龙尾 (m/s)'] = []
    data_dict_2['龙尾（后） (m/s)'] = []
    for dt in range(-100, 101):
        x_list, y_list, v_list = calc_one_seconds(t_turn+dt, theta1, theta2, s, xo1, yo1, xo2, yo2, r, xp, yp, theta_turn, v=100.0, p=170.0, plot=False)
        print(f'Calaculating {dt} second...')
        # 时间
        data_dict_1['时间'].append(f'{dt} (s)')
        data_dict_2['时间'].append(f'{dt} (s)')
        # 添加龙头的数据
        data_dict_1['龙头x (m)'].append(round(x_list[0] / 100,6))
        data_dict_1['龙头y (m)'].append(round(y_list[0] / 100,6))
        data_dict_2['龙头 (m/s)'].append(v_list[0])
        # 添加每节龙身的数据
        for i in range(221):
            data_dict_1[f'第{i + 1}节龙身x (m)'].append(round(x_list[i + 1] / 100,6))
            data_dict_1[f'第{i + 1}节龙身y (m)'].append(round(y_list[i + 1] / 100,6))
            data_dict_2[f'第{i + 1}节龙身 (m/s)'].append(v_list[i + 1])
        # 添加龙尾的数据
        data_dict_1['龙尾x (m)'].append(round(x_list[-2] / 100,6))
        data_dict_1['龙尾y (m)'].append(round(y_list[-2] / 100,6))
        data_dict_2['龙尾 (m/s)'].append(v_list[-2])
        data_dict_1['龙尾（后）x (m)'].append(round(x_list[-1] / 100,6))
        data_dict_1['龙尾（后）y (m)'].append(round(y_list[-1] / 100,6))
        data_dict_2['龙尾（后） (m/s)'].append(v_list[-1])
    df1 = pd.DataFrame(data_dict_1)
    df1 = df1.set_index('时间').T
    df1.reset_index(inplace=True)
    df1 = df1.rename(columns={'index': '时间'})
    df2 = pd.DataFrame(data_dict_2)
    df2 = df2.set_index('时间').T
    df2.reset_index(inplace=True)
    df2 = df2.rename(columns={'index': '时间'})
    df1.to_excel(f"./outputs/result4_1.xlsx", index=False)
    df2.to_excel(f"./outputs/result4_2.xlsx", index=False)

# Parameters for the spiral
num_turns = 16
theta_max = 2 * np.pi * num_turns
p = 55.0
v = 100.0
# 问题求解
def question1():
    run_all_seconds()
    run_one_second()
def question2():
    v_crush, x_crush, y_crush, t = run_with_check()
    with open('./outputs/result2_1.txt', 'w') as f:
        f.write(f'Crush Time:{t}\nCrush X:{x_crush}\nCrush Y:{y_crush}\nCrush V:{v_crush}\n')
    run_one_second(write=True, filename='result2', t=t)
def question3():
    p = get_min_p()
def question4():
    # theta1, theta2, s, xo1, xo2, yo1, yo2, r, xp, yp = find_best_path()
    # read result4_1.txt
    theta1 = 15.199116361661993
    theta2 = 12.89041795278792
    s = 1118.5301769104117
    xo1 = -129.0685386894533
    yo1 = 51.39711010981853
    xo2 = 278.8254858722373
    yo2 = 102.65325624368478
    r = 274.0681362725451
    xp = 142.86081101834046
    yp = 85.5678741990627
    #print(theta1,theta2,s, xo1, yo1, xo2, yo2, r, xp, yp)
    theta_turn = clockwise_angle(xo1, yo1, fuc_x(theta1, p), fuc_y(theta1,p), xp, yp)
    t_turn = 170.0 / (4 * np.pi * v) * (theta_max ** 2 - theta1 ** 2)
    print(t_turn)
    turnning_one_second(1120.9022382005592, theta1, theta2, s, xo1, yo1, xo2, yo2, r, xp, yp, theta_turn, v=119.4, p=170.0)
    #calc_all_seconds(theta1, theta2, s, xo1, yo1, xo2, yo2, r, xp, yp, theta_turn, v=100.0, p=170.0)
def question5():
    theta1 = 15.199116361661993
    theta2 = 12.89041795278792
    s = 1118.5301769104117
    xo1 = -129.0685386894533
    yo1 = 51.39711010981853
    xo2 = 278.8254858722373
    yo2 = 102.65325624368478
    r = 274.0681362725451
    xp = 142.86081101834046
    yp = 85.5678741990627
    theta_turn = clockwise_angle(xo1, yo1, fuc_x(theta1, p), fuc_y(theta1, p), xp, yp)
    v_range = np.linspace(120,123,3)
    for v in v_range:
        t_turn = 170.0 / (4 * np.pi * v) * (theta_max ** 2 - theta1 ** 2)
        for dt in range(-20, 30):
            print(f'Current V={v}cm/s, Time={t_turn + dt}...')
            x_list, y_list, v_list = calc_one_seconds(t_turn + dt, theta1, theta2, s, xo1, yo1, xo2, yo2, r, xp, yp,
                                                      theta_turn, v=v, p=170.0, plot=False)
            #print('Calculating Finish!')
            for team_v in v_list:
                if team_v > 2:
                    with open('./outputs/result5.txt', 'w') as f:
                        f.write(f'When Head Speed={v}cm/s\nFound Team Speed:{team_v}m/s\nAt Time:{t_turn + dt}')
                        print(f'When Head Speed={v}cm/s\nFound Team Speed:{team_v}m/s\nAt Time:{t_turn + dt}')
                        return

#run_one_second(t=300, plot=True)
#question1()
#question2()
#question3()
#question4()
question5()