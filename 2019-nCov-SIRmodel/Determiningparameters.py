# coding:utf-8
# SIR模型预测新型冠状病毒肺炎数据


import scipy.integrate as spi
import numpy as np
import pandas as pd

beta = 8e-6
gamma = 0.04
TS = 1.0
ND = 60.0
I0 = 41
count = 0



if __name__ == "__main__":
    # 读取数据
    #一月十六日至二月十六日
    data = pd.read_csv("/Users/zzbj/Desktop/jianmo/2019-nCov-SIRmodel-master/truedata.csv", index_col=["date"])
    #γ值设定为0.04，即一般病程25天
    #用最小二乘法估计β值和初始易感人数
    gamma = 0.04
    t_start = 0.0
    t_end = ND
    t_inc = TS
    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    S0 = [i for i in range(30000, 1000000, 1000)]
    beta = [f for f in np.arange(1e-7, 1e-4, 1e-7)]


    # 定义偏差函数
    def error(res):
        data["现有感染者"] = data["感染者"] - data["死亡"] - data["治愈"]
        err = (data["现有感染者"] - res)**2
        errsum = sum(err)
        return errsum

    # 穷举法，找出与实际数据差的平方和最小的S0和beta值
    # 结果 S0 = 102000, β = 3.5e-6
    minSum = 1e10
    minS0 = 0.0
    minBeta = 0.0
    bestRes = None
    for S in S0:
        for b in beta:
            # 模型的差分方程
            def diff_eqs_2(INP, t):
                Y = np.zeros((3))
                V = INP
                Y[0] = -b * V[0] * V[1]
                Y[1] = b * V[0] * V[1] - gamma * V[1]
                Y[2] = gamma * V[1]
                return Y
            # 数值解模型方程
            INPUT = [S, I0, 0.0]
            RES = spi.odeint(diff_eqs_2, INPUT, t_range)
            errsum = error(RES[:33, 1])

            if errsum < minSum:
                minSum = errsum
                minS0 = S
                minBeta = b
                bestRes = RES
                count = count+1
                print("count=%d S0=%d beta=%.8f minErr=%f" % (count,S, b, errsum))

    print("S0 = %d β = %.8f" % (minS0, minBeta))

