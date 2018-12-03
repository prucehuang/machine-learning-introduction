# 逻辑回归
### 一、假设函数
$$
h_\theta(x) = g(\theta^TX)=\frac{1}{1+e^{-\theta^TX}} (Sigmoid函数)
$$
Sigmoid函数的值范围是(0, 1)， 所以当$$h_\theta(x)$$的值大于0.5的时候y=1，且$$\theta^TX>0$$
我们把$$\theta^TX=0$$的面称为**决策边界**

### 二、代价函数 - 平方误差函数
线性回归的代价函数是平方损失函数，将逻辑回归的假设函数代入公式后的损失函数是一个非凸函数，有很多个局部最优解，没有办法快速的获得全局最优解
$$
\begin{cases}
 & \text{ if y=1 then } -y^{(i)}log(h_\theta(x^{(i)}) \\ 
 & \text{ if y=0 then } (1-y^{(i)})log(1-h_\theta(x^{(i)})) 
\end{cases}
$$

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

```chart
{
    "title": {
        "text": "Fruits number"
    },
    "tooltip": {},
    "legend": {
        "data":["Number"]
    },
    "xAxis": {
        "data": ["Apple","Banana","Peach","Pear","Grape","Kiwi"]
    },
    "yAxis": {},
    "series": [{
        "name": "Number",
        "type": "bar",
        "data": [5, 20, 36, 10, 10, 20]
    }]
}
```