# coding:utf-8
# SIR模型预测新型冠状病毒肺炎数据


import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd


beta = 4.2e-6
gamma = 0.04
TS = 1.0
ND = 60.0
S0 = 85000
I0 = 41
INPUT = [S0, I0, 0.0]


# 模型的差分方程
def diff_eqs(INP, t):
	Y = np.zeros((3))
	V = INP
	#print(V)
	Y[0] = -beta * V[0] * V[1]
	Y[1] = beta * V[0] * V[1] - gamma * V[1]
	Y[2] = gamma * V[1]
	return Y


if __name__ == "__main__":
	t_start = 0.0
	t_end = ND
	t_inc = TS
	t_range = np.arange(t_start, t_end+t_inc, t_inc)
	RES = spi.odeint(diff_eqs, INPUT, t_range)
	#print(S0,I0)
	#print(RES)
	#print(len(RES))
	# 数据做图 画出预测数据
	fig = pl.figure()
	pl.subplot(111)
	pl.plot(RES[:, 1], "-r", label = "Infectious")
	pl.plot(RES[:, 0], "-g", label = "Susceptibles")
	pl.plot(RES[:, 2], "-k", label = "Recovereds")
	pl.legend(loc = 0)
	pl.title("SIR model")
	pl.xlabel("Time")
	pl.ylabel("Infectious Susceptibles")
	pl.savefig("/Users/zzbj/Desktop/jianmo/2019-nCov-SIRmodel-master/result.png")
	
	# 读取数据
	data = pd.read_csv("/Users/zzbj/Desktop/jianmo/2019-nCov-SIRmodel-master/latestdata.csv", index_col = ["date"])
	data["现有感染者"] = data["感染者"] - data["死亡"] - data["治愈"]
	print(data)
	
	# 数据作图
	fig = pl.figure()
	pl.subplot(111)
	pl.plot(data["现有感染者"], "-r", label = "infected")
	pl.plot(data["疑似者"], "-g", label = "undecided")
	pl.plot(data["死亡"], "-b", label = "death")
	pl.plot(data["治愈"], "-k", label = "healed")
	pl.plot(data["现有感染者"]-data["现有感染者"].shift(1), "-y", label = "increase")
	pl.legend(loc = 0)
	pl.title("real data")
	pl.xlabel("Time")
	pl.ylabel("Infectious Susceptibles")
	pl.xticks(rotation=-90)
	fig.tight_layout()
	pl.savefig("/Users/zzbj/Desktop/jianmo/2019-nCov-SIRmodel-master/realdata.png")

			
	print("预测最大感染人数:%d 天数:%d" % (RES[:,1].max(), np.argmax(RES[:, 1])))
	# 将预测值与真实值画到一起
	fig = pl.figure()
	pl.subplot(111)
	pl.plot(RES[:, 1], "-r", label = "Infectious")
	pl.plot(data["现有感染者"], "o", label = "realdata")
	pl.plot(data["现有感染者"]-data["现有感染者"].shift(1), "-y", label = "increase")
	pl.legend(loc = 0)
	pl.title("SIR model")
	pl.xlabel("Time")
	pl.ylabel("Infectious Susceptibles")
	pl.xticks(rotation=-90)
	fig.tight_layout()
	pl.savefig("/Users/zzbj/Desktop/jianmo/2019-nCov-SIRmodel-master/contrastdata.png")
		
	
