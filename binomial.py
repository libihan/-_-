# -*- coding: utf-8 -*-
"""
# 二项分布 (binomial distribution)
# 前提：独立重复试验、有放回、只有两个结果
# 二项分布指出，随机一次试验出现事件A的概率如果为p，那么在重复n次试验中出现k次事件A的概率为：
# f(n,k,p) = choose(n, k) * p**k * (1-p)**(n-k)
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#==============================================================================
# # ①定义二项分布的基本信息
#==============================================================================
p = 0.4 # 事件A概率0.4
n = 5   # 重复实验5次
k = np.arange(n+1) # 6种可能出现的结果:你可能观测到A发生0次,1次...6次
#k = np.linspace(stats.binom.ppf(0.01,n,p), stats.binom.ppf(0.99,n,p), n+1) #另一种方式


# ②计算二项分布的概率质量分布 (probability mass function)
# 之所以称为质量，是因为离散的点，默认体积（即宽度）为1
# P(X=x) --> 是概率
probs = stats.binom.pmf(k, n, p) 
#array([ 0.07776,  0.2592 ,  0.3456 ,  0.2304 ,  0.0768 ,  0.01024])
#plt.plot(k, probs)


# ③计算二项分布的累积概率 (cumulative density function)
# P(X<=x) --> 也是概率
cumsum_probs = stats.binom.cdf(k, n, p)
#array([ 0.07776,  0.33696,  0.68256,  0.91296,  0.98976,  1.     ])


# ④根据累积概率得到对应的k，这里偷懒，直接用了上面的cumsum_probs
k2 = stats.binom.ppf(cumsum_probs, n, p)
#array([0, 1, 2, 3, 4, 5])


# ⑤伪造符合二项分布的随机变量 (random variates)
X = stats.binom.rvs(n,p,size=20)
#array([2, 3, 1, 2, 2, 2, 1, 2, 2, 3, 3, 0, 1, 1, 1, 2, 3, 4, 0, 3])

#⑧作出上面满足二项分布随机变量的频数直方图（类似group by）
plt.hist(X)

#⑨作出上面满足二项分布随机变量的频率分布直方图
plt.hist(X, normed=True)
plt.show()