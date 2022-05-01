# -*- coding: utf-8 -*-
"""
Created on Sun May  1 09:18:30 2022

@author: QAJ
"""

### 决策树
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

clf = DecisionTreeClassifier() # 建立决策树分类器
iris = load_iris() # 获取iris数据
cross_val_score(clf, iris.data, iris.target, cv=10) #交叉验证，第一个参数是分类器



data = iris.data

a = pd.DataFrame(range(150))
select = a.sample(frac = 0.8)

exerData = data[select]
exer = exerData.reshape(120,4)
exerName = iris.target[select]
name = exerName[:,0]

clf.fit(exer, name)  # 训练决策树模型


c = list(select[0])
b = a.drop(a.index[c]) # b是a的补集

test = data[b[0]] #得到测试集

t = clf.predict(test) # 利用决策树模型，进行测试集的预测


### 下面把测试集预测值和真实值用图示的方法表示出来，借用上面的PCA代码。

y = iris.target[b[0]] #使用y表示原来的标签
X = data #使用 X表示数据集中的属性数据

pca = PCA(n_components=2)
#加载 PCA 算法，设置降维后主成分数目为 2
reduced_X = pca.fit_transform(X)
# 对原始数据进行降维，保存在 reduced_X 中

red_x, red_y = [], [] # 第1类数据点
blue_x, blue_y = [], [] # 第2类数据点
green_x, green_y = [], [] # 第3类数据点


points = pd.DataFrame(reduced_X)
testPoints = points.iloc[b[0]]
selectPoints = testPoints.values


### 第一个子图
plt.subplot(1, 2, 1)
plt.title("原图")
for i in range(len(b)):
    if y[i] == 0:
        red_x.append(selectPoints[i][0])
        red_y.append(selectPoints[i][1])        
    elif y[i] == 1:
        blue_x.append(selectPoints[i][0])
        blue_y.append(selectPoints[i][1])        
    else:
        green_x.append(selectPoints[i][0])
        green_y.append(selectPoints[i][1])

plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')

# 参考线
plt.axvline(x=0, c = 'k',ls = '--', lw = 1)
plt.axhline(y=0, c = 'k',ls = '--', lw = 1)

### 第二个子图
plt.subplot(1, 2, 2)

plt.title("预测图")
for i in range(len(b)):
    if t[i] == 0:
        red_x.append(selectPoints[i][0])
        red_y.append(selectPoints[i][1])        
    elif t[i] == 1:
        blue_x.append(selectPoints[i][0])
        blue_y.append(selectPoints[i][1])
    else:
        green_x.append(selectPoints[i][0])
        green_y.append(selectPoints[i][1])
        
### 因为要标注错误的预测点，所以需要逐个画点。
for i in range(len(b)):
    if t[i] == 0:
        if t[i] == y[i]:
            plt.scatter(selectPoints[i][0], selectPoints[i][1], c = 'r', marker = 'x')
        else:
            plt.scatter(selectPoints[i][0], selectPoints[i][1], c = 'm', marker = '+')      
    elif t[i] == 1:
        if t[i] == y[i]:
            plt.scatter(selectPoints[i][0], selectPoints[i][1], c = 'b', marker = 'D')
        else:
            plt.scatter(selectPoints[i][0], selectPoints[i][1], c = 'y', marker = '1')
    else:
        if t[i] == y[i]:
            plt.scatter(selectPoints[i][0], selectPoints[i][1], c = 'g', marker = '.')
        else:
            plt.scatter(selectPoints[i][0], selectPoints[i][1], c = 'k', marker = '*')
# 参考线
plt.axvline(x=0, c = 'k',ls = '--', lw = 1)
plt.axhline(y=0, c = 'k',ls = '--', lw = 1)

plt.show()