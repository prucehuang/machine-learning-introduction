# Typothetical Test 假设检验

## U检验：$$\sigma^2$$已知，关于$$\mu$$的检验
### 假设检验
$$
    H_0: \mu = \mu_0, H_1: \mu \neq \mu_0
$$
### 统计量
$$
    U = \frac{\bar{x}-\mu_0}{\frac{\sigma}{\sqrt{n}}} \sim N(0,1)
$$
### 拒绝域
根据定义，对于一个给定的置信区间$$\alpha$$，我们可以在正态分布两端取到分位点$$\pm u_\frac{\alpha}{2}$$,既
$$
    P\left\{ \left| U \right| \geq u_\frac{\alpha}{2} \right\}= \alpha 
$$
如果统计量的值u，$$\left| u \right| \geq u_\frac{\alpha}{2}$$，则意味着发生了小概率事件，因此原假设$$H_0$$为小概率事件，拒绝原假设
故拒绝域为
$$
    W_1 = \left \{  \left| u \right| \geq u_\frac{\alpha}{2} \right \}
$$


## T检验：$$\sigma^2$$未知，关于$$\mu$$的检验
### 假设检验
$$
    H_0: \mu = \mu_0, H_1: \mu \neq \mu_0
$$
### 统计量
$$
    T = \frac{\bar{x}-\mu_0}{\frac{S}{\sqrt{n}}} \sim t(n-1)
$$
$$S^2$$为$$\alpha^2$$的无偏估计
### 拒绝域
$$
    W_1 = \left \{  \left| t \right| \geq t_\frac{\alpha}{2}(n-1) \right \}
$$
t分布和正态分布的曲线类似，所以拒绝域的计算方式也类似，不同的是方差未知我们只能用$$S^2$$来代替$$\alpha^2$$


## 卡方$$\chi^2$$检验：$$\mu$$未知，关于$$\sigma^2$$的检验
### 假设检验
$$
    H_0: \sigma^2 = \sigma_0^2, H_1: \sigma^2 \neq \sigma_0^2
$$
### 统计量
$$
    \chi^2 = \frac{(n-1)S^2}{\sigma^2} \sim \chi^2(n-1)
$$
### 拒绝域
$$
    W_1 = \left \{ 
        \chi^2 \leq \chi^2_{1-\frac{\alpha}{2}}(n-1) 
        或者 
        \chi^2 \geq \chi^2_\frac{\alpha}{2}(n-1) 
    \right \}
$$
标准卡方分布$$\chi^2$$分布的左右两边不对称，所以将两边分开来










## Z-检验：双样本均值差检验

### 假设条件
1. 两个样本是独立的样本
2. 总体正态分布 或 非正态分布大样本（样本量不小于30）
3. 两样本方差已知

### 统计量
$$
    Z=\frac{\bar{x}-\bar{y} }{\sqrt{\frac{\sigma_1^2}{n_1}+\frac{\sigma_2^2}{n_2}}} \sim N(0,1)
$$

* 原假设 $$H_0 : \mu_1 = \mu_2$$
* 备选假设 $$H_1$$

## T-检验：平均值的成对二样本检验

* 假设条件
* 两个总体配对差值构成的总体服从正态分布
* 配对差是由总体差随机抽样得来的
* 数据配对或匹配（重复测量（前/后））

* 统计计算

## T-检验：双样本等方差假设

* 假设条件
* 两个独立的小样本
* 两总体都是正态总体
* 两总体方差未知，但值相等

* 统计计算

## T-检验：双样本异方差假设

* 假设条件
* 两总体都是正态总体
* 两总体方差未知，且值不等

* 统计计算

## F检验：双样本方差检验

介绍

* 假设条件
* 
* 统计计算

## 参考文章

[【Excel系列】Excel数据分析：假设检验](https://www.jianshu.com/p/1c60c9c3fe33)  
[正态总体均值与方差的假设检验](https://wenku.baidu.com/view/6080ea93970590c69ec3d5bbfd0a79563c1ed48a.html)

> @ WHAT - HOW - WHY  
> @ 不积跬步 - 无以至千里



