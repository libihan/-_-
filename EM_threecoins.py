# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 11:37:24 2018

@author: Administrator
"""
import numpy as np
#==============================================================================
# (1)一次迭代
#==============================================================================
def em_single(observations,priors):
    """
    # 1)第一次先投掷A，若出现正面则投掷B，否则投掷C
    # 2)记录第二次投掷的硬币出现的结果，正面记作1，反面记作0
    """
    π = priors[0]
    p = priors[1]
    q = priors[1]
    
    #E步
    sum_B = 0   #数据来自B的概率求和
    sum_C = 0
    B_H = 0     #数据来自B且出现正面的概率求和
    C_H = 0
    
    for y in observations:
        #数据y来自B/C的概率
        contribute_B = π * p**y * (1-p)**(1-y)    #u
        contribute_C = (1-π) * q**y * (1-q)**(1-y)
        #归一化
        weighted_B =contribute_B / (contribute_B +contribute_C)
        weighted_C = 1 - weighted_B     #contribute_C / (contribute_B +contribute_C)
        
        sum_B += weighted_B
        sum_C += weighted_C
        
        #数据来自B且出现正面
        B_H += weighted_B * y
        C_H += weighted_C * y

    #M步
    new_π = sum_B / len(observations)
    new_p = B_H / sum_B
    new_q = C_H / sum_C
    
    return[new_π,new_p,new_q]

#==============================================================================
# (2)EM算法主循环    
#==============================================================================
def em(observations,priors,tol=1e-4,iters=10000):
    """
    priors:模型初值
    """
    for iter in range(iters):
        new_priors = em_single(observations,priors)
        
        error = sum([(x-y)**2 for x,y in zip(new_priors,priors)])
        if np.sqrt(error) < tol:
            break
        priors = new_priors
    return new_priors,iter+1

if __name__ == '__main__':
    """
    observations --- 采集数据:用1表示H(正面),用0表示T(反面)
    初始值
    [0.5,0.5,0.5] --- ([0.5, 0.6, 0.6], 2)
    [0.4,0.6,0.7] --- ([0.39999999999999997, 0.6000000000000001, 0.6000000000000001], 2)
    """
    observations = [1,1,0,1,0,0,1,0,1,1]
    ans = em(observations,[0.5, 0.6, 0.6])
    print(ans)