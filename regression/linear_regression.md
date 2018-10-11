# Linear Regression (线性回归)

### 一、函数模型
$$
    h_\theta(x) = \theta^TX = \sum_{i=0}^{n}\theta_ix_i = \theta_0 + \theta_1x + \cdots + \theta_nx
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
&nbsp;&nbsp;&nbsp;&nbsp;$$
    \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}J(\theta_0, \theta_1, ...)
    (for j=0,1,2,...)
$$
&nbsp;&nbsp;&nbsp;&nbsp;$$   
    \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}\frac{1}{2m}\sum_{i=1}^{m}(h(x^{(i)}) - y^{(i)})^2   
    (for j=0,1,2,...)
$$
&nbsp;&nbsp;&nbsp;&nbsp;$$   
    \theta_j := \theta_j - \alpha \frac{1}{m}\sum_{i=1}^{m}(h(x^{(i)}) - y^{(i)})x^{(i)}   
    (for j=0,1,2,...)
$$ 

}

注意:
- 初始点不同可能导致的求解的结果不一样
- 所有的$$\theta_j$$需要同一批次更新，即更新$$\theta_3$$的时候用的不是最新一批的$$\theta_1, \theta_2$$,而是上一批次的值，只有等到n-1个变量全都更新完后，才使用最新的值去计算下一批；
- $$\alpha$$表示学习率，学习率太小，则学习时间太长；学习率太大，可能会错过最低点，最终在最低点来回摆动而无法到达最优；
- 求解的最低点倒数为0，越接近最低点倒数绝对值越小，所以$$\theta_j$$的变化也就越小
- 每次迭代一个$$\theta_j$$都需要用全量的数据

#### 2、正规化函数



