# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 20:49:43 2017

@author: Administrator
"""
import numpy as np
import random
from sklearn.metrics import accuracy_score

study_step = 0.0001
study_total = 10000
n = 324  #特征数
object_num = 0   # 分类的数字

m = len(train_labels)

#(1)初始化w,b
w = np.zeros((n,1))
b = 0

study_count = 0                         # 学习次数记录，只有当分类错误时才会增加
nochange_count = 0                      # 统计连续分类正确数，当分类错误时归为0
nochange_upper_limit = 100000           # 连续分类正确上界，当连续分类超过上界时，认为已训练好，退出训练

while True:
    nochange_count += 1
    if nochange_count > nochange_upper_limit:
        break
    
    #(2)随机选择数据
    index = random.randint(0,n-1)
    xi = trainset[index]
    label = train_labels[index]
    
    #(3)计算yi(w*xi+b)
    yi = (label != object_num) * 2 - 1
    result = yi * (xi * w + b)
    
    #判断
    if result <= 0:
        w += study_step * xi.T * yi
        b += study_step * yi
        
        study_count += 1
        if study_count > study_total:
            break
        nochange_count = 0

predict = []
for x in testset:
    result = x * w + b
    predict.append(result > 0)

score =  accuracy_score(test_labels,np.array(predict))  
        
        
        
        