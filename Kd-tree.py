# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:32:46 2017

@author: Administrator
"""
import numpy as np

class KD_node:
    def __init__(self,elt=None,split=None,LL=None,RR=None):
        """
        elt:样本点，即根节点
        split:划分域
        LL,RR:左右子空间内数据点构成的kd-tree
        """
        self.elt = elt
        self.split = split
        self.left = LL
        self.right = RR

def computeVariance(arr):
    """
    var = E[X^2] - (E(X))^2
    """
    sum1 = sum(arr)
    mean = sum1/arr.size
    arr2 = arr * arr  #list不能相乘
    sum2 = sum(arr2)
    
    return sum2/arr.size - mean ** 2

def createKDTree(root,data_list):
    """
    root:当前树的根节点  
    data_list:数据点的集合(无序)  
    return:构造的KDTree的树根
    
    LEN:数据点的个数
    demension:数据点的维数
    """
    if len(data_list) == 0:
        return
    
    LEN = len(data_list)
    demension = len(data_list[0])
    
    var_max = 0
    split = 0
    for d in range(demension):
        arr1 = np.array(demension)[:,d]
        var = computeVariance(arr1)
        
        #找出方差最大的那一维,记为split
        if var_max < var:
            var_max = var
            split = d
        
    #根据选择的划分域对数据点进行排序,方便分配左右儿子
    data_list = sorted(data_list,key = lambda x:x[split])
    elt = data_list[LEN//2]
    
    root = KD_node(elt,split)
    root.left = createKDTree(root.left,data_list[:LEN//2])
    root.right = createKDTree(root.right,data_list[LEN//2+1:])
    
    return root

def findNN(root,query):
    """
    root:KDTree的树根 
    query:查询点 
    return:返回距离data最近的点NN，同时返回最短距离min_dist 
    
    nodeList:用来存储二分查找经过的节点，先进后出，方便回溯查找
    back_elt:回溯查找找弹出的节点
    ss:分割的维度
    """
    NN = root.elt
    min_dist = np.linalg.norm(np.array(NN)-np.array(query))
    
    nodeList = []
    temp_root = root
    #二叉查找,直到根节点
    while temp_root:
        nodeList.append(temp_root)
        dist = np.linalg.norm(np.array(temp_root.elt)-np.array(query))
        
        if dist < min_dist:
            NN = temp_root.elt
            min_dist = dist
            
        #当前节点的划分域  
        ss = temp_root.split
        if query[ss] < temp_root[ss]:
            temp_root = temp_root.left
        else:
            temp_root = temp_root.right
    
    #回溯查找,将nodeList遍历一遍
    while nodeList:
        back_root = nodeList.pop()
        back_elt = back_root.elt
        ss = back_root.split
        
        """
        if min大于query到分割面的距离:
            相邻子空间可能存在更近的点
        """
        if abs(query[ss] - back_elt[ss]) < min_dist: 
            if query[ss] > back_elt[ss]:
                temp_root = back_root.left
            else:
                temp_root = back_root.right
            #类似二分查找
            if temp_root:
                nodeList.append(temp_root)
        
                #相邻子空间找更近的点
                dist = np.linalg.norm(np.array(temp_root.elt)-np.array(query))
            
                if dist < min_dist:
                    NN = temp_root.elt
                    min_dist = dist
    return NN,min_dist
                
def preorder(root):
    print(root.elt)
    
    if root.left:
        preorder(root.left)
    if root.right:
        preorder(root.right)

def knn(list,query):
    NN = list[0]
    min_dist = np.linalg.norm(np.array(query) - np.array(NN))
    
    for l in list:
        dist = np.linalg.norm(np.array(query) - np.array(l))
        
        if dist < min_dist:
            NN = l
            min_dist = dist
    return NN,min_list
          
        
        
        
    
    
    