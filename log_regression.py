# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:47:32 2017

@author: Administrator
Note1:
当对array进行约简运算时,经常二维数组变成一维array;运算时会broadcast效果,可能报错;所以此处用matrix类型.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split


def loadData():
    train_x = []
    train_y = []

    with open('test_set.txt') as f:
        text = f.readlines()

    for line in text:
        train_x.append([1] + line.split()[:-1])   #此处float便于后续计算
        train_y.append([line.split()[-1]]) #此处float便于后续train_y - output,[[]]是为了方便转为数组
    return np.mat(train_x).astype(float),np.mat(train_y).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))    
    
def trainLogRegres(train_x,train_y,opts):
    numSample,numFeature = train_x.shape    #样本数numSample，特征数numFeature
    weights = np.ones((numFeature,1))
    
    maxIter = opts['maxIter']
    alpha = opts['alpha']
    for k in range(maxIter):
        if opts['optimizeType'] == 'gradDescent':   ## gradient descent algorilthm 
            output = sigmoid(train_x * weights) #train_x:(numSample,numFeature)
            error = train_y - output    #error:(numSample,1)
            weights += alpha * train_x.transpose() * error
        elif opts['optimizeType'] == 'stocGradDescent':     #stochastic gradient descent
            for i in range(numSample):
                alpha = 4 / (100 + k + i)   #约束alpha>稍大的常数项
                output = sigmoid(train_x[i] * weights)    #train_x[i]:(1,numFeature)  weights:(numFeature,1)
                error = train_y[i] - output     #error：(1,1)
                weights += alpha * train_x[i].transpose() * error
        elif opts['optimizeType'] == 'min-batch-gradDescent':   #set b = 10
            b = 10
            for i in range(0,b,numSample):
                output = sigmoid(train_x[i:i+b] * weights)   #train_x[i]:(b,numFeature)
                error = train_y[i:i+b] - output    #error:(numSample,1)
                weights += alpha * train_x[i:i+b].transpose() * error
        else:
            raise NameError('Not support optimize method type!') 
    return weights

def testLogRegres(weights, test_x, test_y):
      
    output = sigmoid(test_x * weights)
    predict = output > 0.5
    
    accuracy = sum(predict == test_y)/len(test_y)
    return accuracy

def showLogRegres(weights, train_x, train_y):
    for i in range(len(train_y)):
        if train_y[i] == 1:
            plt.plot(train_x[i,1], train_x[i,2],'or')
        else:
            plt.plot(train_x[i,1], train_x[i,2],'ob')
        
    min_x = float(min(train_x[:,1]))
    max_x = float(max(train_x[:,1]))
    min_y = (- weights[0,0] - weights[1,0] * min_x) / weights[2,0]  #注意此处x2和x1的关系
    max_y = (- weights[0,0] - weights[1,0] * max_x) / weights[2,0]
    plt.plot([min_x,max_x],[min_y,max_y],'g-')  #绘制分割线
        
    plt.xlabel('x1')
    plt.ylabel('x2')
     
if __name__ == '__main__':
    ##step1：load data
    print('step1:Start loading data...')
    time_1 = time.time()
    
    train_set,train_label = loadData()
    train_x,test_x,train_y,test_y = train_test_split(train_set,train_label,test_size=0.33, random_state=23323)
    
    time_2 = time.time()
    print('      Loading data costs %f seconds.\n'%(time_2 - time_1))
    
    ##step2：training...
    print('step2:Start training...')
    opts = {'alpha': 0.01, 'maxIter': 20, 'optimizeType': 'stocGradDescent'}  
    optimalWeights = trainLogRegres(train_x,train_y,opts)
    
    time_3 = time.time()
    print('      Training data costs %f seconds.\n'%(time_3 - time_2))
    ##step3：testing...
    print('step3:Start testing...')
    accuracy = testLogRegres(optimalWeights, test_x, test_y)
    print('      accuracy:%f'%accuracy)
    
    time_4 = time.time()
    print('      Testing data costs %f seconds.\n'%(time_4 - time_3))
    
    ##step4：plot the figure...
    print('step4:Start plotting the figure...')
    showLogRegres(optimalWeights, train_x, train_y)
    
    time_4 = time.time()
    print('      Plotting the figure costs %f seconds.\n'%(time_4 - time_3))
    
    
    