# Linear Regression (线性回归)

### 一、函数模型
$$
    h_\theta(x) = \theta^TX = \sum_{i=0}^{n}\theta_ix_i = \theta_0 + \theta_1x + \cdots 
$$

### 二、代价函数 - 平方误差函数
$$
    J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h(x^{(i)}) - y^{(i)})^2
$$

### 三、目标函数
$$
    Min J(\theta)
$$

### 四、求解最小化目标函数

#### 1、梯度下降

**初始点不同可能导致的结果不一样**

伪代码  
repeat until convergence {
$$
    \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}J(\theta_0, \theta_1, ...)(for j=0,1,2,...)
$$
}
注意，所有的$$\thera_j$$需要再同时更新
$$\alpha$$表示学习率，学习率太小，则学习时间太长；学习率太大，可能会错过最低点，最终在最低点来回摆动而无法到达最优



#### 2、正规化函数



