[toc]

# 一、特征工程
## 1.1 特征处理
### 特征清洗  
- 删除异常数据

### 去量纲
- Max-Min
$$
X_{norm} = \frac{X-X_{min}}{X_{max}-X_{min}}
$$

- z-score
$$
z = \frac{x-\mu}{\sigma}
$$

- 归一化
$$
x^{'} = \frac{x}{\sqrt{\sum_{j=1}^{n}(x_j)^2}}
$$

### 离散化
- 序列编码  
序列转换成数字，保留优先级，如高中低 ==> 3、2、1

- one-hot  
序列转换成数字，不保留优先级，如血型A、B、O、AB分别对应(1,0,0,0)、(0,1,0,0)、(0,0,1,0)、(0,0,0,1)  
维度太高的时候可以使用hash来降维，或者改成稀疏向量的形式

- 二进制编码  
如血型A、B、O、AB分别对应(0,0,1)、(0,1,0)、(0,1,1)、(1,0,0)  

- word2vec

### 缺失值处理
- 平均值填充
- 0填充
- 删除

### 特征变换
- 生成多项式特征
- 图像随机旋转、平移、缩放、裁剪、填充、翻转
- 图像像素增加噪声扰动
- 图像颜色变换
- 图像改变亮度、清晰度、对比度、锐度

## 1.2 特征选择
特征选择的两个标准  
1. 特征是否发散
   如果特征不发散，方差接近于0，样本在这个特征上基本上没有差异，所以特征并没有什么用  
2. 特征与目标的相关性
   与目标相关性高的特征，应当优选选择  

### 过滤法(filter)  
  按照发散性或者相关性对各个特征进行评分，选择特征  
- 方差选择，方差越大越好
- 相关系数，和label的相关系数越高越好
- 卡方检验
- 信息增益，信息增益越大越好

### 包装法(wrapper)
通过目标函数(AUC/MSE)来决定是否要新加入一个变量

### 集成法(embedded)
先使用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，根据系数从大到小选择特征。类似于Filter方法，但是是通过训练来确定特征的优劣

## 1.3 特征降维
- PCA（主成分分析，Principal Components Analysis）  
  寻找合适的超平面，能有最大化的投影方差，让映射后的样本具有最大的发散性，无监督的方法
- LDA（线性判别分析，Linear Discriminant Analysis）  
  找到一个合适的投影超平面，让映射后的样本有最好的分类性能，有监督的方法


# 二、模型评估
## 2.1 分类模型评估

真实\预测 | 0 | 1
---|---|---
0 | TN | FP 
1 | FN | TP

所有的，都是在描述模型输出结果  
- T ： True 表示判断正确 ；F ： False 表示判断错误
- P ： PostIve 表示判断该样本为正样本； N ： Negative 表示判断该样本为负样本
- TP ： (T)该判断正确，(P)判断该样本为正样本（事实上样本为正）
- FP ： (F)该判断错误，(P)判断该样本为正样本（事实上样本为负）
- TN ： (T)该判断正确，(N)判断该样本为负样本（事实上样本为负）
- FN ： (F)该判断错误，(N)判断该样本为负样本（事实上样本为正）

$$
Precision = \frac{TP}{TP+FP}
$$
$$
Recall = \frac{TP}{TP+FN}
$$
$$
F1 = \frac{2}{\frac{1}{P}+\frac{1}{R}}=\frac{2PR}{P+R}
$$
$$
Accurate = \frac{TP+TN}{TN+FN+FP+TP}
$$

- P-R曲线  
横坐标：召回率  
纵坐标：准确率  
P-R曲线是通过将阈值从高到低移动而生成的  

- ROC曲线  
横坐标：FPR 假阳性率  
纵坐标：TPR 真阳性率、召回率  
ROC曲线是通过将阈值从高到低移动而生成的  
当正负样本的比例发生变化时，ROC曲线相比于P-R曲线会更为稳定

- AUC  
  是ROC曲线的面积，表示任意取两个样本，正样本的score大于负样本的score的概率

## 2.2 回归模型评估
- SSE 和方差
$$
SSE = \sum_{i=1}^{n}(y^{(i)}-p^{(i)})^2
$$
- MSE 均方方差
$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(y^{(i)}-p^{(i)})^2
$$
- RMSE 均方根
$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y^{(i)}-p^{(i)})^2}
$$
- MAPE 平均绝对百分比误差(mean absolute percent error)
$$
MAPE = \frac{100}{n} \times \sum_{i=1}^{n}| \frac{y^{(i)}-p^{(i)}}{y^{(i)}} |
$$

## 2.3 聚类算法评估
- 评估聚类随机程度 —— 霍普金斯统计量H  
  从所有样本中随机寻找n个点，P，再寻找离每个点最近的点x，计算px的距离x；  
  从所有样本中寻找n个点，q，再寻找最近点y，计算qy的距离y
$$
H = \frac{\sum_{i=1}^{n}y_i}{\sum_{i=1}^{n}x_i + \sum_{i=1}^{n}y_i}
$$
  随机分布H=0.5，H值越大聚类效果越好
- 评估聚类质量 —— 轮廊系数  
  a( p )表示p与同簇其他点之间的平均距离  
  b( p )表示里p最近的一个族中所有点到p的平均距离  
  轮廊系数可以衡量族与簇的距离、族自己的紧凑程度
$$
s(p) = \frac{b(p)-a(p)}{max{\{ b(p),a(p) \}}}
$$

- 均方根标准偏差
- R方

## 2.4 过拟合和欠拟合
### 过拟合，降低复杂度
- 获得更多的数据
- 降低模型复杂度
- 增加正则项
- 集成学习方法

### 欠拟合，增加复杂度
- 增加新特征
- 增加模型复杂度
- 减小正则项

# 三、经典机器学习
## 3.1 liner regression 
- 假设函数：
$$
    h_\theta(x) = \theta^TX = \sum_{i=0}^{n}\theta_ix_i = \theta_0 + \theta_1x_1 + \cdots + \theta_nx_n
$$

- 损失函数：
$$
J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h(x^{(i)}) - y^{(i)})^2
$$

- 正规方程求解：
$$
X = \begin{pmatrix}
            x_0^{(1)} & x_1^{(1)} & \cdots & x_n^{(1)} \\ 
            x_0^{(2)} & x_1^{(2)} & \cdots & x_n^{(2)} \\ 
            \cdots & \cdots & \cdots & \cdots \\ 
            x_0^{(m)} & x_1^{(m)} & \cdots & x_n^{(m)} \\
        \end{pmatrix}
$$
$$
\theta = (X^TX)^{-1}X^TY
$$

- 梯度下降求解：
$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}J(\theta_0, \theta_1, ...) \qquad 
    (simulate - for \quad j=0,1,2,...)
$$

开始梯度下降之前：
1. 梯度下降对特征的值敏感，不同的特征取值差别太大会影响收敛效率，所以需要对特征进行归一化
2. 初始点不同可能导致的求解的结果不一样
3. 学习率太小，则学习时间太长；学习率太大，可能会错过最低点，最终在最低点来回摆动而无法到达最优

梯度下降中：
1. 求解的最低点导数为0，越接近最低点倒数绝对值越小，所以theta的变化也就越小
2. 每次迭代一个theta都需要用全量的数据，消耗资源多

梯度下降后：
1. 绘制**梯度下降曲线**，横坐标是迭代次数，纵坐标是损失函数的值。正常情况下曲线应该是单调递减，最后趋近稳定

**几种梯度下降方法**：

- Batch Gradient Descent  
    1) 每一次迭代都需要使用全量的训练数据X
    2) 超参数  
        学习率eta, 可以用grid search去找一个    
        总迭代次数n_iterations, 用两次迭代之间的cost变化值，来衡量最大迭代次数选择的好坏
- Stochastic Gradient Descent  
    随机的选择一个实例进行梯度下降
    1)  优点  
        much faster  
        makes it possible to train on huge training sets  
        has a better chance of finding the global minimum than Batch Gradient Descent does 因为随机，所以更有可能找到全局最优解，而不是局部最优
    2) 缺点  
        much less regular than Batch Gradient Descent  
        最后只会在最小值附近徘徊，并不会停在最优解上  
    3) One Solution  
        simulated annealing  
        一开始的时候大学习率, 更好的跳出局部最优  
        接着就逐渐的降低学习率, 更好的收敛到最优解  
    4) learning schedule 控制学习率  
       learning rate is reduced too quickly, you may get stuck in a local minimum, or even end up frozen halfway to the minimum  
       learning rate is reduced too slowly, you may jump around the minimum for a long time and end up with a suboptimal solution if you halt training too early
- Mini-batch Gradient Descent  
    不同于批量梯度下降（每次处理全部实例的梯度），也不同于随机梯度下降（每次只处理一个实例的梯度），每次处理一小批实例的梯度（computes the gradients on small random sets of instances）
    1) 优势  
        对比SGD，更加稳定规则，结果更接近最优解
    2) 劣势  
        对比SGD，跳出局部最优稍微难一些

## 3.2 logistic regression
- 假设函数：
$$
h_\theta(x) = g(\theta^TX)=\frac{1}{1+e^{-\theta^TX}} \qquad \qquad  \qquad (Sigmoid函数)
$$

$$
\begin{cases}
 & \theta^TX小于0 =\Rightarrow h_\theta(x) < 0.5 =\Rightarrow y=0 \\
 & \theta^TX>0 =\Rightarrow h_\theta(x) > 0.5 =\Rightarrow y=1\\ 
 & \theta^TX=0 =\Rightarrow h_\theta(x) = 0.5 =\Rightarrow 决策边界
\end{cases}
$$
- 损失函数

线性回归的代价函数是平方损失函数，将逻辑回归的假设函数代入公式后的损失函数是一个非凸函数，有很多个局部最优解，没有办法快速的获得全局最优解，于是我们就用上了最大似然估计：
$$
J(\theta)=\begin{cases}
 & \text{if y=1    then } -y^{(i)}log(h_\theta(x^{(i)}) \\ 
 & \text{if y=0    then } -(1-y^{(i)})log(1-h_\theta(x^{(i)})) 
\end{cases}
$$

$$
J(\theta)=\frac{1}{m}\sum_{i=1}^{m}(-y^{(i)}log(h_\theta(x^{(i)})) - (1-y^{(i)})log(1-h_\theta(x^{(i)})))
$$

## 3.3 决策树
- 决策树损失函数
$$
J(k, t_k)=\frac{m_{left}}{m}G_{left} + \frac{m_{right}}{m}G_{right} \;\; 
(G \; measures \; the \; impurity \; of \; the \; subset)
$$

- 基尼指数
$$
Gini_i=1-\sum_{k=1}^{n}p_{i,k}^2
$$

- 香农指数
$$
H_i=-\sum_{k=1,p_{i,k}\neq 0}^{n}p_{i,k}log(p_{i,k})
$$

## 3.4 Kmeans
- 损失函数：
$$
J(c, \mu) = \sum_{i=1}^{M}||x_i-{\mu}_{c_i}||^2
$$
- 调优策略：  
数据归一化、去除离群点噪声、合理的选择K值（手肘法）、采用核函数  

- 缺点：  
需要手动设置K值、受初始值影响比较大、受噪声影响大、样本只能被分到一个类别

- 优化算法：kmeans++  
初始化第n+1个聚类中心的时候，尽可能的选择里0-n个点比较远的点

## 3.5 损失函数
- 0-1损失函数  
预测值和目标值不相等为1，否则为0
$$
L(Y, f(X)) = 
\left\{\begin{matrix}
1, & Y \neq f(x)\\ 
0, & Y = f(x)
\end{matrix}\right.
$$

- 绝对值损失函数
$$
L(Y, f(X)) = | Y - f(x) |
$$

- log对数损失函数
$$
L(Y, P(Y|X)) = -logP(Y|X)
$$

- 平方损失函数
$$
L(Y | f(x)) = \sum(Y-f(X))^2
$$

- 指数损失函数
$$
L(Y | f(x)) = exp[-yf(X)]
$$

- Hinge损失函数
$$
L(y) = max(0, 1-ty)
$$
## 3.6 正则项与Early Stopping
- Ridge Regression 岭回归 L2  
$$
a\frac{1}{2} \sum_{i=1}^{n}\theta_i^2
$$

- Lasso Regression 套索回归 L1  
    completely eliminate the weights of the least important features (i.e., set them to zero),  对重要性不高的特征打压比较重
$$
a\sum_{i=1}^{n}|\theta_i|
$$

- Elastic Net 弹性网络  
    a middle ground between Ridge Regression and Lasso Regression
$$
ra\sum_{i=1}^{n}|\theta_i| + (1-r)a\frac{1}{2} \sum_{i=1}^{n}\theta_i^2
$$

- 如何选择惩罚方式  
    1) Ridge is a good default
    2) 假设你怀疑有一些特征不重要的时候，you should prefer Lasso or Elastic Net，通常Elastic Net is preferred over Lasso

- Early Stopping  
    stop training as soon as the validation error reaches a minimum suspect，欠拟合和过拟合之间的拐点也就是验证集误差最小的点，毕竟训练误差会一直在下降  
    使用BGD时，遇到最低点就可以停止；使用SGD、MBGD就需要多观察一会儿看看会不会有更小值

# 四、集成学习
## 4.1 Voting Classifers
- aggregate the predictions of each classifier and predict the class that gets the most votes
- this voting classifier often achieves a higher accuracy than the best classifier in the ensemble
- Ensemble methods work best when the predictors are as independent from one another as possible，One way is to use very different training algorithms
```python
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    # 有两种投票方式
    # voting='hard', 选出被投票次数最多的分类
    # voting='soft', 选所有分类中预测概率最大的分类
    voting='hard'
)
voting_clf.fit(X_train, y_train)
```

## 4.2 Bagging and Pasting
- 用同一个算法不断的抽样训练  
    有放回的抽样 —— Bagging (bootstrap aggregating)  
    无放回的抽样 —— Pasting  
- Bagging VS Pasting  
    bagging slightly higher bias than pasting  
    pasting slightly higher variance than bagging  
- 训练可以并行
- Out-of-Bag Evaluation  
    从统计学上来说，有放回的抽样可以触达大概63%的样本，剩下的37%样本可以作为验证集来使用，out-of-bag 简称oob

## 4.3 Boosting
将弱模型组合起来成为更强大的模型
### AdaBoost  
1. 先训练一个模型，然后预测训练数据，找出预测错误的样本，加大权重，继续训练一个新的模型，再预测，再训练   
2. The algorithm stops when the desired number of predictors is reached, or when a perfect predictor is found  
3. Adaboost不支持并行，下一个模型的生成必须依赖于上一个模型预测完的结果  

第j个模型的错误率，instance weight w(i) is initially set to 1/m
$$
r_j= \frac{\sum_{i=1,\hat{y^{(i)}} \neq y^{(i)}}^{m}w^{(i)}}{\sum_{i=1}^{m}w^{(i)}}
$$
第j个模型的权重，准确率越高，权重值越大，随机模型权重为0
$$
\alpha_j=\eta log \frac{r_j}{1-r_j}
$$
更新每一个样本的权重值
$$
\begin{aligned}
&for \; i= 1, 2 \cdots m \\
&w^{(i)} \leftarrow \begin{cases}
w^{(i)} \quad &if \; \hat{y^{(i)}}=y^{(i)} \\
w^{(i)}exp(\alpha_j) \quad &if \; \hat{y^{(i)}} \neq y^{(i)} 
\end{cases}
\end{aligned}
$$
归一化，然后训练下一个模型...  
预测结果是由所有的N个模型投票，选出sum权重最高的分类
$$
\hat{y}(x) = \underset{k}{argmax}\sum_{\overset{j=1}{\hat{y_j}(x)=k}}^{N}\alpha_j
$$

### Gradient Boosting
1. 先训练一个模型，然后预测，将y-y_pre赋值给新y，再训练一个模型，and so on
2. 最后预测的结果等于SUM所有的模型预测结果

## 4.4. Stacking
- short for stacked generalization 组合泛化
- blender  
    多训练一个预测结果处理模型-blender，将多个模型的预测结果作为输入，输出一个最终的结果，替换之前的vote模式
- 实现过程  
    首先将训练分成两组，一组用来训练第一层的模型；然后将第一层模型的预测结果作为第二层blender模型的输入；或者将训练数据分成三组，训练两层组合模型+一个blender


> @ WHAT - HOW - WHY