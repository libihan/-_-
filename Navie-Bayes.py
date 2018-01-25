# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:21:33 2017

@author: Administrator
"""
import numpy as np

def loadDataSet():
    """
    return: 单词列表postingList, 所属类别classVec 
    """
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                  ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                  ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                  ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]#1 侮辱性文字 ， 0 代表正常言论
    return postingList,classVec

def createVocabList(dataSet):
    """
    dataSet: 数据集 return: vocabList不含重复元素的单词列表
    """
    vocabSet = set()
    for data in dataSet:
        vocabSet |= set(data)
    return list(vocabSet)

#遍历vocabList，查看该单词是否出现在input，出现该单词则将该单词置1  
def wordsToVector(vocabList, inputSet):
    """
    vocabList: 所有单词集合列表,inputSet: 输入数据集 
    return: returnVec匹配列表[0,1,0,1...]，其中 1与0 表示词汇表中的单词是否出现在输入的数据集中 
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('The word: %s is not in the vocabList!'%word)
    return returnVec

def train_bayes(trainset, trainCategory):
    """
    先验概率和条件概率的求解
    return:
        pAbusive:负类的概率 = 负类的数量/训练集的数目
        pabu_vector:类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表   
        pnorm_vector:类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    """
    num_train = len(trainset)
    num_vocab = len(trainset[0])
    
    pAbusive = sum(trainCategory) / num_train
    
    #求条件概率：构造单词出现次数列表
    pnorm_vector = np.ones(num_vocab)
    pabu_vector = np.ones(num_vocab)
    #初始化整个数据集单词出现总数2
    pnorm_denom = 2
    pabu_denom = 2
    
    for i in range(num_train):
        if trainCategory[i] == 1:
            pabu_vector += trainset[i]
            pabu_denom += sum(trainset[i])
        else:
            pnorm_vector += trainset[i]
            pnorm_denom += sum(trainset[i])
    
    pabu_vector = np.log(pabu_vector / pabu_denom)
    pnorm_vector = np.log(pnorm_vector / pnorm_denom)
    
    return pAbusive,pabu_vector,pnorm_vector     
    
def classify(testVector, pnorm_vector, pabu_vector, pabusive):
    """
    testVector:待测数据[0,1,1,1,1...],即要分类的向量 
    pnorm_vector:类别0,pabu_vector:类别1,pabusive:类别1，侮辱性文件的出现概率 
    return:类别1 or 0
    # 计算公式 log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    testVector*pnorm_vector:单词在词汇表中的条件下，文件是good 类别的概率        
    """
    pabu = sum(testVector * pabu_vector) + np.log(pabusive)
    pnorm = sum(testVector * pnorm_vector) + np.log(1-pabusive)
    if pabu > pnorm:
        return 1
    else:
        return 0
            
"""
trainMat已经将原始的输入转换成[[0,0,1,...],[1,0]...]
"""    
if __name__ == '__main__':
        # 1. 加载数据集  
    posts_List, classes_list = loadDataSet()  
        # 2. 创建单词集合  
    vocab_List = createVocabList(posts_List)  
        # 3. 计算单词是否出现并创建数据矩阵  
    trainset = []  
    for post in posts_List:  
            # 返回m*len(myVocabList)的矩阵， 记录的都是0，1信息  
        trainset.append(wordsToVector(vocab_List, post))  
        # 4. 训练数据  
    pAbusive,pabu_vector,pnorm_vector = train_bayes(trainset,classes_list)  
        # 5. 测试数据  
    testEntry1 = ['love', 'my', 'dalmation']  
    testVector1 = wordsToVector(vocab_List, testEntry1)
    print (testEntry1, 'classified as: ', classify(testVector1, pnorm_vector, pabu_vector, pAbusive))  
    
    testEntry2 = ['stupid', 'garbage']  
    testVector2 = wordsToVector(vocab_List, testEntry2)
    print (testEntry2, 'classified as: ', classify(testVector2, pnorm_vector, pabu_vector, pAbusive))      