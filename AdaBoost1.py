import numpy as np
import matplotlib.pyplot as plt

#==============================================================================
# (1)载入数据
#==============================================================================
def loadSimpleData():
    dataMat = np.mat([[1,2.1],
                     [2.1,1],
                     [1.3,1],
                     [1,1],
                     [2,1]])
    classLabels = np.mat([1,1,-1,-1,1]).T
    return dataMat,classLabels

#==============================================================================
# (2)可视化数据
#==============================================================================
def plotData(dataMat,classLabels):
    xcard0 = []
    ycard0 = []
    xcard1 = []
    ycard1 = []
    
    for i in range(len(classLabels)):
        if classLabels[i] == 1:
            xcard1.append(dataMat[i,0])
            ycard1.append(dataMat[i,1])
        else:
            xcard0.append(dataMat[i,0])
            ycard0.append(dataMat[i,1])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcard0,ycard0,marker='s',s=90)
    ax.scatter(xcard1,ycard1,marker='o',s=90,c='r')
    plt.title('Decision Stump test data')
    plt.show()
    
#==============================================================================
# (3)已知特征维度dimen,分类边界thre,符号Ineq----求得预测的label
#==============================================================================
def stumpClassify(dataMat,dimen,thre,ineq):
    reMat = np.mat(np.ones((np.shape(dataMat)[0],1)))
    if ineq == 'lt':    #≤thre判为-1
        reMat[dataMat[:,dimen] <= thre] = -1
    else:
        reMat[dataMat[:,dimen] >= thre] = -1
    return reMat

#==============================================================================
# (4)已知权值矩阵D和训练数据dataMat和classLabels----求得最优的Gm(x)
#==============================================================================
def bulidStump(dataMat,classLabels,D):
    """
    遍历特征维度/遍历分类边界/遍历Ineq
    D维度m*1
    #return: Gm(x)[dimen/thre/ineq] 分类误差率minError bestClass
    """
    m,n = dataMat.shape
    numSteps = 10   #分类边界10步
    
    minError = float('inf')
    bestStump = {}
    bestClass = np.mat(np.zeros((m,1)))
    #遍历特征维度
    for dimen in range(n):   
        rangemax = max(dataMat[:,dimen])
        rangemin = min(dataMat[:,dimen])
        stepSize = (rangemax - rangemin) / numSteps
        #遍历分类边界
        for i in range(-1,numSteps+1):
            thre = rangemin + i * stepSize  #分类边界,从min-stepsize---max+stepsize
            for ineq in ('lt','gl'):
                #利用(3)求得此时预测的label
                predictLabels = stumpClassify(dataMat,dimen,thre,ineq)
                errorArr = np.mat(np.ones((m,1)))
                errorArr[predictLabels == classLabels] = 0  #errorArr中预测正确为0,预测错误为1
                #分类误差率
                weightedError = D.T * errorArr
                #找到最优的Gm(x)
                if weightedError < minError:
                    minError = weightedError
                    bestStump['dimen'] = dimen
                    bestStump['thre'] = thre
                    bestStump['ineq'] = ineq
                    bestClass = predictLabels
    return bestStump,float(minError),bestClass

#==============================================================================
# (5)训练AdaBoost---给定分类器数目M,返回弱分类器weakClass和fx
#==============================================================================
def adaBoostTrain(dataMat,classLabels,M=40):
    m,n = dataMat.shape
    
    weakClass = []
    fx = np.mat(np.zeros((m,1)))
    #1.初始化训练数据集的权值分布
    D = np.mat(np.ones((m,1)) * 1/m)
    
    for i in range(M):
        bestStump,minError,bestClass = bulidStump(dataMat,classLabels,D)
        
        #2.c--Gm(x)的系数alpha
        alpha = 1/2 * np.log((1 - minError) / max(minError,1e-16))   #【注意】：防止发生除零错误
        bestStump['alpha'] = alpha
        weakClass.append(bestStump)
        #2.d---更新权值分布D
        expon = -alpha * np.multiply(classLabels,bestClass)
        D = np.multiply(D,np.exp(expon))
        D = D / D.sum()
        #3.计算fx
        fx += alpha * bestClass
        #如果误差为0,则结束循环
        aggError = np.multiply(np.sign(fx) != classLabels,np.mat(np.ones((m,1))))
        errorRate = aggError.sum() / m
        print('errorRate:%f'%errorRate)
        if errorRate == 0:
            break
    return weakClass,fx

#==============================================================================
# (6)提升树分类----输入分类器和测试数据集datToclass,输出类别
#==============================================================================
def adaClassify(datToclass,weakClass):
    dataMat = np.mat(datToclass)
    m = dataMat.shape[0]
    
    aggClassEst = np.zeros((m,1))
    for i in range(len(weakClass)):
        classEst = stumpClassify(dataMat,weakClass[i]['dimen'],weakClass[i]['thre'],weakClass[i]['ineq'])
        aggClassEst += weakClass[i]['alpha'] * classEst
        print('labelValue:%f'%aggClassEst)
    return np.sign(aggClassEst)

#==============================================================================
# 主函数
#==============================================================================
if __name__ == '__main__':
    dataMat,classLabels = loadSimpleData()
    plotData(dataMat,classLabels)
    
    weakClass,fx = adaBoostTrain(dataMat,classLabels,30)
    label = adaClassify([0,0],weakClass)
    print('The label of [0,0] is:%d'%label)
    
    """
    [分析结果]:
    分类过程中，随着加法模型的不断叠加，
    对于（0，0）这个点，其累加结果是“负”得越来越厉害的，
    最终取符号输出-1类别。
    """
    
            