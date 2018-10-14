# Linear Regression (线性回归)

### 一、假设函数
$$
    h_\theta(x) = \theta^TX = \sum_{i=0}^{n}\theta_ix_i = \theta_0 + \theta_1x_1 + \cdots + \theta_nx_n
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

伪代码  
repeat until convergence {
$$
    \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}J(\theta_0, \theta_1, ...)
    (simulate - for j=0,1,2,...)
$$
$$   
    \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}\frac{1}{2m}\sum_{i=1}^{m}(h(x^{(i)}) - y^{(i)})^2   
    (for j=0,1,2,...)
$$
$$   
    \theta_j := \theta_j - \alpha \frac{1}{m}\sum_{i=1}^{m}(h(x^{(i)}) - y^{(i)})x^{(i)}   
    (for j=0,1,2,...)
$$ 
}

开始梯度下降之前：
- 梯度下降对特征的值敏感，所以需要对特征进行归一化，如$$x_1=\frac{x_1-\mu_1}{s_1}$$， 处理后数据集的均值为0，方差为1
- 初始点不同可能导致的求解的结果不一样
- $$\alpha$$表示学习率，学习率太小，则学习时间太长；学习率太大，可能会错过最低点，最终在最低点来回摆动而无法到达最优；

梯度下降ing：
- 所有的$$\theta_j$$需要同一批次更新，即更新$$\theta_3$$的时候用的不是最新一批的$$\theta_1, \theta_2$$,而是上一批次的值，只有等到n-1个变量**全都**更新完后，才使用最新的值去计算下一批；
- 求解的最低点倒数为0，越接近最低点倒数绝对值越小，所以$$\theta_j$$的变化也就越小
- 每次迭代一个$$\theta_j$$都需要用全量的数据，消耗资源多

梯度下降后：
- 绘制**梯度下降曲线**，横坐标是迭代次数，纵坐标是损失函数的值。正常情况下曲线应该是单调递减，最后趋近稳定

#### 2、正规化函数



