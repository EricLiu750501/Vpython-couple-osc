from vpython import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

g = vector(0, 9.8, 0)  # 重力加速度(m/s^2)
size, m, d = 0.1, 0.001, 0.3  # 設定大小/joint質量/距離常數
K, k = 1e5, 5e4  # 彈力係數
L = 0.96  # 擺長長度
M = 100  # m1,m2質量
theta0 = 50 * np.pi / 180  # 初始擺角
t = 0  # 時間
dt = 0.0001
joint2_data = joint_data = t_data = m1_data = m2_data = np.array([]) # 數據紀錄
p_m1 = np.array([-0.71106195,  0.13471959,  0.00557993,  2.84337717,  3.07789602,
        0.01445426])  # m1 運動預測
p_m2 = np.array([-0.71106195,  0.13471959,  0.00557993,  2.84337717,  3.07789602,
        0.01445426])  # m2 運動預測
TIME = 300  # 停止時間

"""
初始位置設定
"""
fixed1_pos = vector(-3 * d, 0.5, 0.0)
fixed2_pos = vector(3 * d, 0.5, 0.0)
joint1_pos = vector(-d, 0, 0)
joint2_pos = vector(d, 0, 0)
m1_pos = vector(-d, -L * np.cos(theta0), L * np.sin(theta0))
m2_pos = vector(d, -L, 0)


scene = canvas(width=1260, height=700, center=vector(0, 0, 0), title='couple osc')

fixed1 = simple_sphere(radius=size / 5, color=color.black, pos=fixed1_pos)    # 左固定球
fixed2 = simple_sphere(radius=size / 5, color=color.black, pos=fixed2_pos)    # 右固定球

joint1 = simple_sphere(radius=size / 10, color=color.white, pos=joint1_pos)   # 左節點
joint2 = simple_sphere(radius=size / 10, color=color.white, pos=joint2_pos)   # 右節點

m1 = simple_sphere(radius=size, color=color.red, pos=m1_pos)   # 球1
m2 = simple_sphere(radius=size, color=color.blue, pos=m2_pos)  # 球2

"""
初始速度
"""
m1_v = vector(0, 0, 0)
m2_v = vector(0, 0, 0)
joint1_v = vector(0, 0, 0)
joint2_v = vector(0, 0, 0)

d_fixed = (fixed1_pos - joint1_pos).mag  # fixed到joint的長度

"""
連結線(不重要)
"""
string1 = cylinder(radius=size / 10)
string2 = cylinder(radius=size / 10)
string3 = cylinder(radius=size / 10)
string4 = cylinder(radius=size / 10)
string5 = cylinder(radius=size / 10)

"""
設定繩長
"""
string1.pos = fixed1.pos
string1.axis = joint1.pos - fixed1.pos
string2.pos = joint1.pos
string2.axis = joint2.pos - joint1.pos
string3.pos = joint2.pos
string3.axis = fixed2.pos - joint2.pos
string4.pos = joint1.pos
string4.axis = m1.pos - joint1.pos
string5.pos = joint2.pos
string5.axis = m2.pos - joint2.pos


def error1(p, x, y):
    """
    a*cos(bx+c)*sin(dx+e)+f
    擬和m1
    :param p: p[a,b,c,d,e,f]
    :param x: x軸
    :param y: y軸
    :return: 最小平方法
    """
    return p[0] * np.cos(p[1] * x + p[2]) * np.sin(p[3] * x + p[4]) + p[5] - y


def error2(p, x, y):
    """
    a*sin(bx+c)*sin(dx+e)+f
    擬和m2
    :param p: p[a,b,c,d,e,f]
    :param x: x軸
    :param y: y軸
    :return: 最小平方法
    """
    return p[0] * np.sin(p[1] * x + p[2]) * np.sin(p[3] * x + p[4]) + p[5] - y


def fs(a, b, l, k):
    """
    :param a:受力物位置
    :param b: 施力物位置
    :param l: 兩者之間長度
    :param k: 彈性常數
    :return: 力
    """
    s_axis = b - a
    tmp = s_axis - l*s_axis.norm()
    f = k * tmp
    return f if tmp.mag > 0 else vector(0, 0, 0)


scene.waitfor('click')
while t <= TIME:
    rate(1 / dt)
    t += dt

    """
    減重力加速度, 乘時間(delta t), ==> 速度
    """
    m1_v -= g * dt
    m2_v -= g * dt
    joint1_v -= g * dt
    joint2_v -= g * dt

    """
    算出力, 除質量, 乘時間(delta t), ==> 速度
    """
    joint1_v += fs(joint1_pos, fixed1_pos, d_fixed, k) / m * dt
    joint1_v += fs(joint1_pos, joint2_pos, 2 * d, K) / m * dt
    joint1_v += fs(joint1_pos, m1_pos, L, k) / m * dt

    joint2_v += fs(joint2_pos, joint1_pos, 2 * d, K) / m * dt
    joint2_v += fs(joint2_pos, fixed2_pos, d_fixed, k) / m * dt
    joint2_v += fs(joint2_pos, m2_pos, L, k) / m * dt

    m1_v += fs(m1_pos, joint1_pos, L, k) / m * dt

    m2_v += fs(m2_pos, joint2_pos, L, k) / m * dt

    """
    位移 = 速度乘時間
    """
    joint1_pos += joint1_v * dt
    joint2_pos += joint2_v * dt
    m1_pos += m1_v * dt
    m2_pos += m2_v * dt

    T = t % (300 * dt)
    if T + dt > 300 * dt and T < 300 * dt:
        m1.pos = m1_pos
        m2.pos = m2_pos
        joint1.pos = joint1_pos
        joint2.pos = joint2_pos
        """
        繩子晃動
        """
        string1.pos = fixed1.pos
        string1.axis = joint1.pos - fixed1.pos
        string2.pos = joint1.pos
        string2.axis = joint2.pos - joint1.pos
        string3.pos = joint2.pos
        string3.axis = fixed2.pos - joint2.pos
        string4.pos = joint1.pos
        string4.axis = m1.pos - joint1.pos
        string5.pos = joint2.pos
        string5.axis = m2.pos - joint2.pos

        # 紀錄數據
        t_data = np.append(t_data, t)
        m1_data = np.append(m1_data, m1_pos.z)
        m2_data = np.append(m2_data, m2_pos.z)
        joint_data = np.append(joint_data, joint1_pos.y)
        joint2_data = np.append(joint2_data, joint2_pos.y)

"""
擬和
"""
ret1 = leastsq(error1, p_m1, args=(t_data, m1_data+joint_data))
ret2 = leastsq(error2, p_m2, args=(t_data, m2_data+joint2_data))

a1, b1, c1, d1, e1, f1 = ret1[0]
a2, b2, c2, d2, e2, f2 = ret2[0]
t_fit = np.linspace(0, t, len(t_data))
m1_fit = a1 * np.cos(b1 * t_fit + c1) * np.sin(d1 * t_fit + e1) + f1
m2_fit = a2 * np.sin(b2 * t_fit + c2) * np.sin(d2 * t_fit + e2) + f2
m1_fit_big = a1 * np.cos(b1 * t_fit + c1) + f1
m2_fit_big = a2 * np.sin(b2 * t_fit + c2) + f2

print('ret1', ret1)
print('ret2', ret2)

plt.figure(figsize=(8, 6))
#plt.scatter(t_data, m1_data+joint_data, color="green", label="m1", linewidth=0.01)
plt.plot(t_data, m1_fit, color='red', label='m1_fit', linewidth=2)
plt.plot(t_data, m1_fit_big,  color='red', label='m1_fit_big', linewidth=0.5)
plt.legend()
plt.xlabel('time (s)')
plt.ylabel('z coordinate')
plt.show()

plt.figure(figsize=(8, 6))
#plt.scatter(t_data, m2_data+joint2_data, color="blue", label="m2", linewidth=0.01)
plt.plot(t_data, m2_fit, color='orange', label='m2_fit', linewidth=2)
plt.plot(t_data, m2_fit_big,  color='red', label='m2_fit_big', linewidth=0.5)

plt.legend()
plt.xlabel('time (s)')
plt.ylabel('z coordinate')
plt.show()


