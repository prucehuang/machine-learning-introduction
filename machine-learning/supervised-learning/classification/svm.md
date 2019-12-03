# SVM(Support Vector Machines)

## 0. Introduction
- Capable of performing linear or nonlinear classification, regression, and even outlier detection
- Well suited for classification of complex but small- or medium-sized datasets

## 1. Linear SVM Classifcation
- introduction  
    1) Fitting the widest possible street between the classes. This is called large margin classifcation 寻找一个离实例最宽的街道平面
    2) it is fully determined (or 'supported') by the instances located on the edge of the street. These instances are called the support vectors 街道完全是被边缘上的实例所影响的，这些实例就是支持向量
    3) SVMs are sensitive to the feature scales

- Hard margin classifcation  
    strictly impose that all instances be off the street  
    two main issues —— 1) the data is linearly separable， 2) it is quite sensitive to outliers

- Soft Margin Classifcation  
    允许部分异常值出现在street里面，The objective is to find a good balance between keeping the street as large as possible and limiting the margin violations   
    用C hyperparameter控制street的宽度 —— 1) C越大，宽度越小，对模型要求越严格， 2) C越小，宽度越大，对模型要求越不严格  
    三种实现API —— 1) LinearSVC(C=1, loss="hinge"), 2) SGDClassifier(loss="hinge", alpha=1/(m*C)) be useful to handle huge datasets that do not fit in memory (out-of-core training), or to handle online classification tasks, 3) SVC(kernel="linear", C=1) 核函数优化

## 2. Nonlinear SVM Classifcation
- Polynomial Kernel  
    面对非线性数据的时候，第一种办法是添加多项式使得数据线性可分，但是多项式degree需要考虑，如果过高，则特征数量巨大，训练很慢；如果太小，没办法处理复杂的数据集  
    the kernel trick 核函数，可以模拟出多项式的效果，without actually having to add them —— SVC(kernel="poly", degree=3, coef0=1, C=5)， 【degree，多项式的阶； coef0 controls how much the model is influenced by highdegree polynomials versus low-degree polynomials，C controls the width of street，C越大对street要求越严格】  

- Gaussian RBF Kernel  
    处理非线性数据，另一种方法是Adding Similarity Features，坐标系转换 —— 选择landmark，将每个点的坐标映射到与这个landmark的相似关系(a similarity function)中去，RBF就是一个这样的点X围绕点l转换的公式
$$
Gaussian \; RBF \; \phi\gamma(X, l) = e^{-\gamma||X-l||^2}
$$
可能将x_1巧妙的转化为x_2,x_3,,,的坐标系，然后线性可分  
The simplest approach is to create a landmark at the location of each and every instance，将数据从非线性的X(m, n) 转成 线性的X(m, m)  
SVC(kernel="rbf", gamma=5, C=0.001)  
超参数1 —— gamma (γ)，作为指数项里面的一个超参数，控制决策边界的regular程度，gamma越大，指数值变化越快，钟型曲线越陡峭，拟合程度越高，偏差越小，方差越大  
超参数2 —— C，同上述，控制street的宽度，C越大，street越窄，模型偏差越小，方差越大
C越小，street越宽，模型偏差越大，方差越小

- Computational Complexity

| Class         | 时间复杂度                  | 超大数据量 | 特征压缩处理 | 核函数 |
| ------------- | --------------------------- | ---------- | ------------ | ------ |
| LinearSVC     | O(m*n)                      | No         | Yes          | No     |
| SGDClassifier | O(m*n)                      | Yes        | Yes          | No     |
| SVC           | O(m\*m\*n) to O(m\*m\*m\*n) | No         | Yes          | Yes    |

## 3. SVM Regression
- Linear SVM Regression  
    LinearSVR(epsilon=1.5)  
    the training data should be scaled and centered first  
    超参ϵ epsilon越小，方差越大
- Nonlinear regression tasks  
    SVR(kernel="poly", degree=2, C=100, epsilon=0.1)  
    超参C越小，more regularization

## 4. Under the Hood
- Decision Function and Predictions  
    1）新约定，the bias term will be called b，the feature weights vector will be called w，No bias feature x_0  
    2）几个超平面  
        Decision function —— 决策函数是一个n+1维的超平面  
        Decision boundary —— 决策边界是当决策函数值为0时的一个n维的超平面，the set of points where the decision function is equal to 0  
        Margin boundary —— street的边界是 the decision function is equal to 1 or –1的超平面，永远和决策边界平行  
    3）Linear SVM classifer  
        ||w||决定了street的宽度，当||w||越大的时候，street的宽度越小
$$
\hat{y} = \begin{cases}
0 \quad if \; w^Tx+b<0 \\
1 \quad if \; w^Tx+b\geq 0 
\end{cases}
$$

- Training Objective  
    1）Hard margin    
    目标是最大化street宽度，也就是最小化||w||  
    define t(i) = –1 for negative instances (if y(i) = 0) and t(i) = 1 for positive instances (if y(i) = 1)  
    2）Soft margin  
    同时权衡最大化边界 和 允许部分实例落入边界  
    **ζ**表示可以出现在street内的概率 —— define  ζ(i) measures how much the i instance is allowed to violate the margin  
    超参**C**也就是上文的C ——  当C越大，ζ越小，模型的方差越大，street的width越小；当C越小，ζ越大，模型的偏差越大，street的width越大  
    不等式右边的1-ζ，表示margin到决策边界的距离，
    随着ζ增大，能出现在street里的实例越多，模型的偏差越大，C也就越小，margin越宽
> 如何从物理上解释ζ越大，1-ζ越小，模型反而越宽敞？
$$
\begin{aligned}
Hard \; margin \; &linear \; SVM \; classifier \; objective \\
\underset{w, b}{minimize} &\quad \frac{1}{2}w^Tw \\
subject \; to &\quad t^{(i)}(w^T \cdot x^{(i)} + b) \geq 1 \; for \; i=1,2, \cdots m \\
\\
Soft \; margin \; &linear \; SVM \; classifier \; objective \\
\underset{w, b}{minimize} &\quad \frac{1}{2}w^Tw + C \sum_{i=1}^{m}\zeta^{(i)}\\
subject \; to &\quad t^{(i)}(w^T \cdot x^{(i)} + b) \geq 1 - \zeta^{(i)} \; and \; \zeta^{(i)} \geq 0 \; for \; i=1,2, \cdots m
\end{aligned}
$$

- Quadratic Programming  
    The hard margin and soft margin problems are both convex quadratic optimization problems with linear constraints. Such problems are known as Quadratic Programming (QP) problems

- The Dual Problem  
    Given a constrained optimization problem, known as the primal problem, it is possible to express a different but closely related problem, called its dual problem  
    往往解决了dual问题，就相当于解决了原始问题，SVM就是这样

- Kernelized SVM  
    常用的核函数

$$
\begin{aligned}
Linear \; &: \quad K(a, b) = a^T \cdot b \\
Polynomial \; &: \quad K(a, b) = (\gamma a^T \cdot b + r)^d \\
Gaussion \; RBF \; &: \quad K(a, b) = exp(-\gamma||a-b||^2) \\
Sigmoid \; &: \quad K(a, b) = tanh(\gamma a^T \cdot b + r)
\end{aligned}
$$


- Online SVMs  
    Linear SVM classifier cost function  
$$
J(w, b) = \frac{1}{2}w^T \cdot w + C\sum_{i=1}^{m}max(0,\;  t^{(i)}(w^T \cdot x^{(i)} + b)) 
$$
前半部分表示截距的斜率 —— leading to a larger margin  
后半部分表示所有在street中点的误差 —— the margin violations as small and as few as possible  
Hinge Loss —— max(0, 1 – t) is called the hinge loss function，当t大于1的时候，函数值恒等于0，对照于off or up street margin的点集