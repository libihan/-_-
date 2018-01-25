# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 22:09:04 2017

@author: Administrator
"""
import numpy as np
import math

#树用嵌套的字典表示

#创建训练集
def createTrainset():
    dataSet = [['长', '粗', '男'],
               ['短', '粗', '男'],
               ['短', '粗', '男'],
               ['长', '细', '女'],
               ['短', '细', '女'],
               ['短', '粗', '女'],
               ['长', '粗', '女'],
               ['长', '粗', '女']]
    train_set = [sample[:-1] for sample in dataSet]
    train_label = [sample[-1] for sample in dataSet] #['男', '男', '男', '女', '女', '女', '女', '女']
    features = ['头发','声音']
    return np.array(train_set),np.array(train_label),features

#统计label的的个数
def createclass_count(train_label):
    class_count = {}
    for label in train_label:
        if label in class_count:
            class_count[label] += 1
        else:
            class_count[label] = 1
    return class_count

#计算信息熵
def calc_ent(train_label):
    hd = 0
    class_count = createclass_count(train_label)
    for label in class_count.keys():
        prob = class_count[label] / len(train_label)
        hd -= prob * math.log(prob,2)
    return hd

#计算各特征对数据集的信息熵
def calc_condition_ent(feature,train_label):
    hda = 0
    for feature_value in set(feature):
        #对于某个特征feature_value,只考虑包含此特征的train_label
        sub_train_label = train_label[feature == feature_value]
        tempt_ent = calc_ent(sub_train_label)
        hda += len(sub_train_label) / len(train_label) * tempt_ent
    return hda

#递归创建决策树
def createTree(train_set,train_label,features):
    # 步骤1——如果train_set中的所有实例都属于同一类Ck[递归停止条件]
    label_set = set(train_label)

    if len(label_set) == 1:
        return label_set.pop()

    # 步骤2——如果特征集features为空[递归停止条件]
    class_count = createclass_count(train_label)
    # 类标记为train_set中实例数最大的类
    max_class = max(class_count, key=class_count.get)

    if len(features) == 0:
        return max_class

    # 步骤3——计算信息增益
    #计算原始的信息熵
    hd = calc_ent(train_label)

    max_gda = 0
    max_feature_idx = -1
    #计算各特征对数据集D的信息增益
    for i in range(len(features)):
        gda = hd - calc_condition_ent(train_set[:,i],train_label)

        if gda > max_gda:
            max_gda = gda
            max_feature_idx = i

    # 步骤4——小于阈值
    if max_gda < epsilon:
        return max_class

    # 步骤5——构建非空子集
    max_feature = features[max_feature_idx]
    tree = {max_feature:{}}

    #删除此特征
    del(features[max_feature_idx])

    feature_value = set(train_set[:,max_feature_idx])   #构建子树,遍历feature的值
    for value in feature_value:
        #子树的训练集和label只包含-含有当前feature-的行
        sub_train_set = train_set[train_set[:,max_feature_idx] == value]
        sub_train_label = train_label[train_set[:,max_feature_idx] == value]
        tree[max_feature][value] = createTree(sub_train_set,sub_train_label,features)
    return tree

def classify(myTree,features,sample):
    #找到最优特征-根节点['声音']
    firstfeature =list(myTree.keys())[0]
    #找到最优特征的索引
    feature_idx = features.index(firstfeature)  #此处'声音'的索引为1
    
    #判断最优特征的值
    secondDic = myTree[firstfeature]
    for key in secondDic.keys():
        if key == sample[feature_idx]:  #和sample对应处相同
            if type(secondDic[key]) == dict:#还没有结束
                mylabel = classify(secondDic[key],features,sample)
            else:
                mylabel = secondDic[key]
    return mylabel
           
if __name__ == '__main__':
    epsilon = 1e-6
    train_set,train_label,features = createTrainset()
    myTree = createTree(train_set,train_label,features)
    print(myTree) #{'声音': {'粗': {'头发': {'短': '男', '长': '女'}}, '细': '女'}}
    
    train_set,train_label,features = createTrainset()
    sample = ['短','粗']
    mylabel = classify(myTree,features,sample)
    print('特征为%s的同学性别为: %s'%(str(sample),mylabel)) # 男





