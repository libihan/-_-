# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:02:57 2018

@author: Administrator
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class Optstruct(object):
    def __init__(self,dataMat,labelMat,C,toler):
        self.dataMat = dataMat
        self.labelMat = labelMat
        self.C = C
        self.toler = toler

        self.m = len(labelMat)
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0

        self.eCache = np.mat(np.zeros((self.m,2)))    #两列:第一列代表此Ek是否有效,第二列是E值

def calcuEk(k,oS):
    fXk = np.multiply(oS.alphas,oS.labelMat).T * oS.dataMat * oS.dataMat[k,:].T + oS.b
    Ek = fXk - oS.labelMat[k]
    return Ek

def updateEk(k,oS):
    Ek = calcuEk(k,oS)
    oS.eCache[k] = [1,Ek]

def selectJrand(i, m):
    j = i  #排除i
    while j == i:   #当j！=i时跳出循环
        j = random.randint(0,m-1)
    return j

def selectJ(i,Ei,oS):	#挑选使abs(Ei – Ej)波动最大的j
    updateEk(i, oS)
    maxK = -1
    Ej = 0
    maxDeltaE = 0
    validEcacheList = np.nonzero(oS.eCache[:,0])[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            #如果k！=i
            Ek = calcuEk(k,oS)
            DeltaE = abs(Ek - Ei)
            if DeltaE > maxDeltaE:
                maxK = k
                maxDeltaE = DeltaE
                Ej = Ek
        return maxK, Ej
    else:   #否则,随机选择j
        j = selectJrand(i, oS.m)
        Ej = calcuEk(j,oS)
        return j,Ej

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    elif aj < L:
        aj = L
    return aj

def innerLK(i, oS):
    """
    变量的选择方法
    :param i:
    :param oS:
    :return:
    """
    Ei = calcuEk(i,oS)
    if (oS.alphas[i] < oS.C and Ei * oS.labelMat[i] < -oS.toler) \
            or (oS.alphas[i] > 0 and Ei * oS.labelMat[i] > oS.toler): #如果不满足KKT条件
        j,Ej = selectJ(i,Ei,oS) #选出使αj波动最大的j
        alphasIOld = oS.alphas[i].copy()
        alphasJOld = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:  #确定L和H的值
            L = max(0,alphasJOld - alphasIOld)
            H = min(oS.C,oS.C + alphasJOld - alphasIOld)
        else:
            L = max(0, alphasJOld + alphasIOld - oS.C)
            H = min(oS.C, alphasJOld + alphasIOld)
        if L == H:
            print('L == H')
            return 0
        eta = oS.dataMat[i,:] * oS.dataMat[i,:].T + oS.dataMat[j,:] * oS.dataMat[j,:].T \
              - 2 * oS.dataMat[i,:] * oS.dataMat[j,:].T
        if eta <= 0:
            print('eta <= 0')
            return 0
        oS.alphas[j] = alphasJOld + oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        #判断αj波动是不是够大
        if abs(oS.alphas[j] - alphasJOld) < 0.00001:
            print('j not moving enough')
            return 0
        updateEk(j, oS)
        oS.alphas[i] = alphasIOld + oS.labelMat[i] * oS.labelMat[j] * (alphasJOld - oS.alphas[j])
        updateEk(i, oS)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.dataMat[i,:] * oS.dataMat[i,:].T) * (oS.alphas[i] - alphasIOld) \
             - oS.labelMat[j] * (oS.dataMat[j,:] * oS.dataMat[i,:].T) * (oS.alphas[j] - alphasJOld)
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.dataMat[i,:] * oS.dataMat[j, :].T) * (oS.alphas[i] - alphasIOld) \
             - oS.labelMat[j] * (oS.dataMat[j, :] * oS.dataMat[j, :].T) * (oS.alphas[j] - alphasJOld)
        if 0 < oS.alphas[i] < oS.C:
            oS.b = b1
        elif 0 < oS.alphas[j] < oS.C:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2
        return 1
    else:
        return 0

def SmoP(dataMat,labeleMat,C,toler,maxIter):
    oS = Optstruct(dataMat,labelMat,C,toler)
    iter = 0
    alphaPairsChanged = 0
    entireSet = True    #第一轮需要遍历所有Set,确定不满足KKT条件的α
    while iter < maxIter and (alphaPairsChanged > 0 or entireSet):
        alphaPairsChanged = 0 #!!每次iter开始,记得置0
        if entireSet:   #遍历所有Set
            for i in range(oS.m):
                alphaPairsChanged += innerLK(i, oS)
                print('fullSet,iter:%d,i:%d,Pairs Changed:%d'%(iter,i,alphaPairsChanged))
            iter += 1
        else:   #只遍历间隔边界即0<α<C
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < oS.C))[0]    #!!
            for i in nonBoundIs:
                alphaPairsChanged += innerLK(i, oS)
                print('nonBoundSet,iter:%d,i:%d,Pairs Changed:%d' % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:   #第一轮后/正常情况,只遍历间隔边界
            entireSet = False
        elif alphaPairsChanged == 0: #当entireSet = False且alpha对未变化时，需要遍历整个边界
            entireSet = True
        print('iter:%d'%iter)
    return oS.b,oS.alphas

def calcWs(alphas,dataMat,labelMat):
    m,n = dataMat.shape
    w = np.mat(np.zeros((n,1)))
    for i in range(m):
        w += np.multiply(labelMat[i] * alphas[i],dataMat[i,:].T)
    return w

def loadDataSet(filename):
    data = pd.read_table(filename,header = None)
    dataSet = data.loc[:,:1]
    labelSet = data[2]

    dataMat = np.mat(dataSet.loc[:,:1])
    labelMat = np.mat(labelSet).T

    return dataMat,labelMat

def plotSVM(dataMat,labelMat,svData,w,b):
    #分别存储正负类的x-y坐标
    xcord0 = []
    ycord0 = []
    xcord1 = []
    ycord1 = []

    for i in range(len(labelMat)):
        xPt = dataMat[i,0]
        yPt = dataMat[i,1]
        if labelMat[i] == -1:
            xcord0.append(xPt)
            ycord0.append(yPt)
        else:
            xcord1.append(xPt)
            ycord1.append(yPt)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord0,ycord0,marker='s',s=90)
    ax.scatter(xcord1,ycord1,marker='o',s=50,c='red')
    plt.title('Support Vector Circled')

    for sv in svData:
        circle = Circle((sv[0,0],sv[0,1]),0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)#
        ax.add_patch(circle)

    w0 = w[0,0]
    w1 = w[1,0]
    b = float(b)
    x = np.arange(-2.0,12.0,0.1)
    y = (-w0 * x - b) / w1
    ax.plot(x,y)
    ax.axis([-2,12,-8,6])
    plt.show()


if __name__ == '__main__':
    dataMat,labelMat = loadDataSet('SVM_data.txt')
    # SmoP(dataMat, lableMat, C, toler, maxIter)
    b,alphas = SmoP(dataMat, labelMat, 0.6, 0.001, 40)
    print('b:%f'%b)
    w = calcWs(alphas,dataMat,labelMat)
    print(w)

    #取α>0的样本点为支持向量
    svInd = np.nonzero(alphas > 0)[0]   #大于0的索引
    svData = dataMat[svInd]
    svLabel = labelMat[svInd]
    print('There are %d Support Vectors.'%len(svLabel))
    for x,y in zip(svData,svLabel):
        print(x,float(y))

    #预测误差
    m,n = dataMat.shape
    predict = dataMat * w + b
    errorCount = sum(np.sign(predict) != labelMat)
    print("The training error rate is: %f" % (float(errorCount) / m))

    #图形化
    plotSVM(dataMat,labelMat,svData,w,b)






