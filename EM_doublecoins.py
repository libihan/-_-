# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 09:39:59 2018

@author: Administrator
"""
import numpy as np
from scipy import stats

#==============================================================================
# (1)一次迭代
#==============================================================================
def em_single(observations,priors):
    """
    #stats.binom.pmf(k,n,p):
        n---投掷次数;p---单次抛掷正面朝上次数;k---观测到k次正面朝上
        return---抛掷n次,观测到k次正面朝上的概率
    """
    theta_A = priors[0]
    theta_B = priors[1]
    
    #E步
    counts = {'A':{'H':0,'T':0},'B':{'H':0,'T':0}}    
    for observation in observations:
        len_observation = len(observation)     #每轮抛掷10次
        num_heads= sum(observation)    #正面朝上次数
        num_tails= len_observation - sum(observation)
        
        contribute_A = stats.binom.pmf(num_heads,len_observation,theta_A)
        contribute_B = stats.binom.pmf(num_heads,len_observation,theta_B)
        
        #归一化概率:计算数据来自硬币A/B的概率
        weighted_A = contribute_A / (contribute_A + contribute_B)
        weighted_B = contribute_B / (contribute_A + contribute_B)
        
        counts['A']['H'] += weighted_A * num_heads
        counts['A']['T'] += weighted_A * num_tails
        counts['B']['H'] += weighted_B * num_heads
        counts['B']['T'] += weighted_B * num_tails
        
    #M步:更新theta_A 和 theta_B
    new_theta_A = counts['A']['H'] / (counts['A']['H'] + counts['A']['T'])
    new_theta_B = counts['B']['H'] / (counts['B']['H'] + counts['B']['T'])
    
    return [new_theta_A,new_theta_B]

#==============================================================================
# (2)EM算法主循环    
#==============================================================================
def em(observations,priors,tol=1e-4,iters=10000):
    """
    priors:模型初值
    """
    for iter in range(iters):
        new_priors = em_single(observations,priors)
       
        if abs(new_priors[0] - priors[0]) < tol:
            break
        priors = new_priors
    return new_priors,iter+1

if __name__ == '__main__':
    """
    observations --- 采集数据:用1表示H(正面),用0表示T(反面)
    """
    observations = np.array([[1,0,0,0,1,1,0,1,0,1],
                             [1,1,1,1,0,1,1,1,1,1],
                             [1,0,1,1,1,1,1,0,1,1],
                             [1,0,1,0,0,0,1,1,0,0],
                             [0,1,1,1,0,1,1,1,0,1]])
    ans = em(observations,[0.6,0.5])
    print(ans)