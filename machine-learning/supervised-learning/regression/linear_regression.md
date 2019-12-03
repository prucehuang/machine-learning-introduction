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

### 四、求解目标函数

#### 1、梯度下降

伪代码  
repeat until convergence {
$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}J(\theta_0, \theta_1, ...) \qquad 
    (simulate - for \quad j=0,1,2,...)
$$
$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}\frac{1}{2m}\sum_{i=1}^{m}(h(x^{(i)}) - y^{(i)})^2 \qquad 
(for \quad j=0,1,2,...)
$$

$$
\theta_j := \theta_j - \alpha \frac{1}{m}\sum_{i=1}^{m}(h(x^{(i)}) - y^{(i)})x^{(i)}   
 \qquad (for \quad j=0,1,2,...)
$$
}

开始梯度下降之前：
- 梯度下降对特征的值敏感，不同的特征取值差别太大会影响收敛效率，所以需要对特征进行归一化，如  $$  x_1=\frac{x_1-\mu_1}{s_1}  $$ 处理后数据集的均值为0，方差为1

- 初始点不同可能导致的求解的结果不一样

- $$\alpha$$表示学习率，学习率太小，则学习时间太长；学习率太大，可能会错过最低点，最终在最低点来回摆动而无法到达最优；

梯度下降ing：
- 所有的$$\theta_j$$需要同一批次更新，即更新$$\theta_3$$的时候用的不是最新一批的$$\theta_1, \theta_2$$,而是上一批次的值，只有等到n-1个变量**全都**更新完后，才使用最新的值去计算下一批；
- 求解的最低点倒数为0，越接近最低点倒数绝对值越小，所以$$\theta_j$$的变化也就越小
- 每次迭代一个$$\theta_j$$都需要用全量的数据，消耗资源多

梯度下降后：
- 绘制**梯度下降曲线**，横坐标是迭代次数，纵坐标是损失函数的值。正常情况下曲线应该是单调递减，最后趋近稳定

几种梯度下降方法：

- Batch Gradient Descent  
    1) 每一次迭代都需要使用全量的训练数据X
    2) 超参数  
        学习率eta, 可以用grid search去找一个    
        总迭代次数n_iterations, 用两次迭代之间的cost变化值，来衡量最大迭代次数选择的好坏
- Stochastic Gradient Descent  
    picks a random instance in the training set at every step and computes the gradients based only on that single instance随机的选择一个实例进行梯度下降
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
    5) 两种随机遍历的方法  
        每次从m个实例中随机选取一个实例用以迭代，选择m次作为一次iteration  
        先将m个实例shuffle，然后逐一遍历全部实例，也是执行m次作为一个iteration  
- Mini-batch Gradient Descent  
    不同于批量梯度下降（每次处理全部实例的梯度），也不同于随机梯度下降（每次只处理一个实例的梯度），每次处理一小批实例的梯度（computes the gradients on small random sets of instances）
    1) 优势  
        对比SGD，get a performance boost from hardware optimization of matrix operations, especially when using GPUs  
        对比SGD，更加稳定规则，结果更接近最优解
    2) 劣势  
        对比SGD，跳出局部最优稍微难一些

#### 2、正规方程
居于，最优点的斜率应该为0，直接求解最小损失函数对应的$$\theta$$向量值

$$
X = \begin{pmatrix}
            x_0^{(1)} & x_1^{(1)} & \cdots & x_n^{(1)} \\ 
            x_0^{(2)} & x_1^{(2)} & \cdots & x_n^{(2)} \\ 
            \cdots & \cdots & \cdots & \cdots \\ 
            x_0^{(m)} & x_1^{(m)} & \cdots & x_n^{(m)} \\
        \end{pmatrix}
$$
将所有的样本数据转化为一个(m, n+1)维的$$X$$  
记住结论

$$
    \theta = (X^TX)^{-1}X^TY
$$

注意：
- $$(X^TX)^{-1}$$，(n+1, m)乘(m, n+1)等于(n+1, n+1)维，求这个矩阵的逆不容易，当n很
大的时候，求解非常耗时，时间复杂度为$$\Theta(n^3)$$；
- 当求解不可逆的时候，
可能是因为$$x_i和x_j$$存在线性关系(特征正规化处理)，
或者因为$$m \leq n$$,即样本数量太小(减小n，删除一些特征来处理)

#### 3、梯度下降 VS 正规方程

梯度下降 | 正规方程
-- | --
需要选择学习率 | 不需要学习率
需要很多次的迭代 | 不需要迭代
特征维度n很大的时候可以工作 | n很大的时候，(n,n)矩阵的逆完全算不出来的

### 五、多项式回归
将原假设函数$$h_\theta(x) = \theta_0 + \theta_1x_1 + \cdots + \theta_nx_n$$中的$$x_i$$改成多项式的形式，比如$$\theta_1x_1 + \theta_2x_2 + \theta_3x_1x_2 + \theta_4x_1^2 + \theta_5x_2^2$$

注意，多项式回归方程中每一项都需要进行**归一化**

> @ 学必求其心得，业必贵其专精
> @ WHAT - HOW - WHY
> @ 不积跬步 - 无以至千里