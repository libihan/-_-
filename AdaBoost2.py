import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#==============================================================================
# (1)载入数据
#==============================================================================
def loadSimpleData(filename):
    data = pd.read_table(filename,header = None)
    dataSet = data.iloc[:,:-1]
    labelSet = data.iloc[:,-1]
    
    dataMat = np.mat(dataSet)
    classLabels = np.mat(labelSet).T
    
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
        print('errorTrain:%f'%errorRate)
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
#        print('labelValue:%s'%aggClassEst)
    return aggClassEst

#==============================================================================
# (7)查准率/召回率
#==============================================================================
def evaluate(fx,classLabels):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for i in range(len(classLabels)):
        if classLabels[i] == 1: #真实为1
            if np.sign(fx)[i] == 1:
                TP += 1     #正确预测为1
            else:
                FN += 1     #错误预测为-1
        else:                   #真实为-1
            if np.sign(fx)[i] == -1:
                TN += 1     #正确预测为-1
            else:
                FP += 1     #错误预测为1
    return TP / (TP + FP),TP / (TP + FN)

#==============================================================================
# (8)输出查准率/召回率 ---- 输入训练集和测试集，输出prTrain,prTest
#==============================================================================
def train_test(dataMat,classLabels,dataMatTest,classLabelsTest,M):
    #训练集
    weakClass,fx = adaBoostTrain(dataMat,classLabels,M)
    prTrain = evaluate(fx,classLabels)
    #测试集
    fx_label = adaClassify(dataMatTest,weakClass)
    prTest = evaluate(fx_label,classLabelsTest)
    
    return prTrain,prTest

#==============================================================================
# ROC曲线: FP--x轴,TP--y轴---【没彻底看懂】
# fx --  一个Numpy数组或者一个行向量组成的矩阵，该参数代表的是分类器的预测强度，
# 在分类器和训练函数将这些数值应用到sign()函数之前，它们就已经产生
#==============================================================================
def plotROC(fx,classLabels):
    """
    cur:保留的是绘制光标的位置
    """
    cur = (1.0, 1.0)  # 中间变量,初始状态为右上角
    ySum = 0.0  # variable to calculate AUC
    numPosClas = sum(np.array(classLabels) == 1.0)  # TP
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = fx.argsort()  # 按元素值排序后的下标,逆序
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # 遍历所有的值
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:  # 预测错误
            delX = 0  # 真阳率不变
            delY = yStep  # 假阳率减小
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]  # ROC面积的一个小长条
        # 从 cur 到 (cur[0]-delX,cur[1]-delY) 画一条线
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')  # 随机猜测的ROC线
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("the Area Under the Curve is: ", ySum * xStep)
    

#==============================================================================
# 主函数
#==============================================================================
if __name__ == '__main__':
    dataMat,classLabels = loadSimpleData('horseColicTraining2.txt')
    dataMatTest,classLabelsTest = loadSimpleData('horseColicTest2.txt')
    
    weakClass,fx = adaBoostTrain(dataMat,classLabels,10)
    fx_test = adaClassify(dataMatTest,weakClass)
    fx_label = np.sign(fx_test)
    errorTest = sum(fx_label != classLabelsTest) / len(classLabelsTest)
    print('errorTest:%f'%errorTest)
    
    plotROC(fx_test.T,classLabelsTest)
    """
    查准率和召回率
    prTrain,prTest = train_test(dataMat,classLabels,dataMatTest,classLabelsTest,10)
    print(prTrain,prTest)
    """
    
            