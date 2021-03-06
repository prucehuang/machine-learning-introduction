# 逻辑回归
### 一、假设函数
$$
h_\theta(x) = g(\theta^TX)=\frac{1}{1+e^{-\theta^TX}} (Sigmoid函数)
$$
![Sigmoid函数](/pic/sigmoid函数.png)
Sigmoid函数的X取值范围是$$(-\infty, +\infty)$$，Y的取值范围是(0, 1)， 
$$
\begin{cases}
 & \theta^TX小于0 =\Rightarrow h_\theta(x) < 0.5 =\Rightarrow y=0 \\
 & \theta^TX>0 =\Rightarrow h_\theta(x) > 0.5 =\Rightarrow y=1\\ 
 & \theta^TX=0 =\Rightarrow h_\theta(x) = 0.5 =\Rightarrow 决策边界
\end{cases}
$$

### 二、代价函数 - 最大似然估计
线性回归的代价函数是平方损失函数，将逻辑回归的假设函数代入公式后的损失函数是一个非凸函数，有很多个局部最优解，没有办法快速的获得全局最优解，于是我们就用上了最大似然估计：
$$
J(\theta)=\begin{cases}
 & \text{ if y=1 then } -y^{(i)}log(h_\theta(x^{(i)}) \\ 
 & \text{ if y=0 then } -(1-y^{(i)})log(1-h_\theta(x^{(i)})) 
\end{cases}
$$
整合后
$$
J(\theta)=\frac{1}{m}(-y^{(i)}log(h_\theta(x^{(i)})) - (1-y^{(i)})log(1-h_\theta(x^{(i)})))
$$

### 三、目标函数
$$
    MinJ(\theta)
$$

### 四、求解目标函数
#### 1、梯度下降

#### 2、正规方程

### 五、为什么是Sigmoid函数

https://blog.csdn.net/bitcarmanlee/article/details/51154481
https://blog.csdn.net/bitcarmanlee/article/details/51292380
https://blog.csdn.net/baidu_15238925/article/details/81291247