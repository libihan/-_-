# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:45:59 2017

@author: Administrator
#问题1：Y是一个集合
#问题2：注意read_table将FALSE自动转为False,pd.read_table(filename,header = None,dtype = str)！！
#问题3：
    特征函数fi(x,y)≠1，而是等于样本中各个特征出现的count和;fi(x,y)的和为14*4=56
"""
import pandas as pd
import math
#from collections import defaultdict

class maxEnt(object):
    def __init__(self,threshold,max_iter):
        self.samples = []    #样本集,元素为(y,x1,x2...)的元组
        self.X = []
        self.Y = set()
        self.max_iter = max_iter    #迭代次数
        self.threshold = threshold
        
        self.W = []
        self.last_W = []    #上一轮迭代的权值
        
        self.xy_list = []   #!!!!!保存全部特征对(xi,y)的列表,xi为x中所有的值
        self.n = 0
        self.C = 0   #样本最大的特征数量？
    
    def load_data(self,filename):
        self.data = pd.read_table(filename,header = None,dtype = str)
        self.samples = [tuple(self.data.loc[i]) for i in range(len(self.data))]
        self.X = [sample[1:] for sample in self.samples]
        self.Y = {sample[0] for sample in self.samples} 
        
    def init_params(self):
       """
       初始化参数
       """
       xy_set = set()
       for sample in self.samples:
           x = sample[1:]
           y = sample[0]
           
           for xi in x:
               xy_set.add((xi,y))
       self.xy_list = list(xy_set)
       self.n = len(self.xy_list)  #特征对的总个数→特征函数的个数
       self.C = max( [len(xi) - 1 for xi in x] )
       
       self.W = [0] * self.n
       self.last_W = self.W[:]  #deep-copy 你变我不变
       
       self.calcu_EPxy()        #只跟Pxy有关,固定
    
#    def calcu_Pxy_Px(self,X,Y): #len和特征数相同
#        self.Pxy = defaultdict(int)
#        self.Px = defaultdict(int)
#        
#        for i in range(len(X)):
#            x = X[i]
#            y = Y[i]
#            for xi in x:
#                self.Pxy[(xi,y)] += 1
#                self.Px[xi] += 1
             
    def calcu_EPxy(self):   #1.计算EPxy_i
        """
        EPxy = sum_x_y(Pxy * f(x,y))    #Pxy未知
        EPxy=[EPxy_i...] EPxy_i = 第i个特征对的EPxy,i=1..n
        """
        self.EPxy = [0] * self.n
        
        for sample in set(self.samples):            
            x = sample[1:]
            y = sample[0]
            
            self.Pxy = 1 / len(self.samples)
            for xi in x:
                id = self.xy_list.index((xi,y))
                self.EPxy[id] += self.Pxy

    def calcu_Pyx(self,x):  #2-1 计算EPx_i,需要计算Pyx
        """
        zw = sum_y( exp( sum_i(wi * fi(x,y)) ) )
        pyx_top = exp( sum_i(wi * fi(x,y)) )
        pyx[y] = pyx_top[y] / zw
        """
        pyx = {}   #对于不同的y值有不同的pyx
        pyx_top = {}  #分子,{y：值}
        zw = 0  #分母，因为对y求和了→所有是一个固定的值
        for y in self.Y:
            sum = 0                        
            for xi in x:
                if (xi,y) in self.xy_list:
                    id = self.xy_list.index((xi,y))
                    sum += self.W[id]
            pyx_top[y] = math.exp(sum)
            zw += pyx_top[y]
        for y in self.Y:
            pyx[y] = pyx_top[y] / zw
        return pyx
                   
    def calcu_EPx(self):    #2.计算EPx_i
        """
        求EPx需要calcu_Px和calcu_pyx
        EPxy = sum_x_y(Pxy * f(x,y))
        EPxy=[每一个特征对的Pxy]
        EPx = sum_x_y(Px * Py|x * f(x,y))
        """       
        self.EPx = [0] * self.n
        for sample in set(self.samples):
            x = sample[1:]
            y = sample[0]
            Px = 1 / len(self.X)  # 计算p(X)的经验分布
            pyx = self.calcu_Pyx(x)
            
            for xi in x:
                id = self.xy_list.index((xi,y))  #特征对在list中的索引id
                self.EPx[id] += Px * pyx[y]
    
    def judge_convergence(self,W,last_W):
        """
        如果不是所有wi都收敛
        """
        for i in range(self.n):
            if math.fabs(W[i] - last_W[i]) >= self.threshold:
                return False    #否则存在不收敛的wi
        return True #全部运行完了,都收敛！
                
    def train(self):
        self.init_params()  #计算了EPxy

        for iter in range(self.max_iter):  #如果不是所有wi都收敛,则再次运行
            self.last_W = self.W[:]
#            self.calcu_Pxy_Px(self.X,self.Y)
            self.calcu_EPx()    #因为pyx随着wi变化
            for i in range(self.n):        
                delta = 1 / self.C * math.log( self.EPxy[i] / self.EPx[i] )
                self.W[i] += delta
            #检查是否收敛
            if self.judge_convergence(self.W,self.last_W):
                break
            
    def predict(self):
#        test = test.strip().split('\t')
        results = []
        for test in self.X:
            prob = self.calcu_Pyx(test)
            result = max(prob,key = prob.get)
            results.append(result)
            print('Testset: %s,\nprob: %s,\nresult: %s\n'%(test,prob,result))
        return results
    
    def calcu_accuracy(self,results):
        return sum(results == self.data[0]) / len(results)

if __name__ == '__main__':
    maxent = maxEnt(threshold = 1e-7,max_iter = 1000)
    maxent.load_data('maxEnt_data.txt')
    maxent.train()
    
    #测试集
    results = maxent.predict()
    print('Accuracy: %f'%maxent.calcu_accuracy(results))