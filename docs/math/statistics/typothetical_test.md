# Typothetical Test 假设检验

# 一、双边检验
## 1.1 U检验：$$\sigma^2$$已知，关于$$\mu$$的检验
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


## 1.2 T检验：$$\sigma^2$$未知，关于$$\mu$$的检验
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
    W_1 = \left \{ \left| t \right| \geq t_\frac{\alpha}{2}(n-1) \right \}
$$
t分布和正态分布的曲线类似，所以拒绝域的计算方式也类似，不同的是方差未知我们只能用$$S^2$$来代替$$\alpha^2$$


## 1.3 卡方$$\chi^2$$检验：$$\mu$$未知，关于$$\sigma^2$$的检验
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
标准卡方分布$$\chi^2$$分布的左右两边不对称，所以将两边分开


# 二、单边检验

## 2.1 U单边检验：$$\sigma^2$$已知，关于$$\mu$$的检验
### 假设检验
$$
    H_0: \mu = \mu_0, H_1: \mu > \mu_0 (或 \mu < \mu_0)
$$
### 统计量
$$
    U = \frac{\bar{x}-\mu_0}{\frac{\sigma}{\sqrt{n}}} \sim N(0,1)
$$
### 拒绝域
根据定义，对于一个给定的置信区间$$\alpha$$，我们可以在正态分布取到单个分位点$$ u_\alpha$$,既
$$
    P\left\{ U > u_\alpha \right\}= \alpha (或P\left\{ U < -u_\alpha \right\}= \alpha )
$$
如果统计量的值u，$$u > u_\alpha(或u < -u_\alpha)$$，则意味着发生了小概率事件，因此原假设$$H_0$$为小概率事件，拒绝原假设
故拒绝域为
$$
    W_1 = \left \{ u > u_\alpha \right \}(或W_1 = \left \{ u < -u_\alpha \right \})
$$


## 2.2 T单边检验：$$\sigma^2$$未知，关于$$\mu$$的检验
### 假设检验
$$
    H_0: \mu = \mu_0, H_1: \mu > \mu_0(或\mu < \mu_0)
$$
### 统计量
$$
    T = \frac{\bar{x}-\mu_0}{\frac{S}{\sqrt{n}}} \sim t(n-1)
$$
$$S^2$$为$$\alpha^2$$的无偏估计
### 拒绝域
$$
    W_1 = \left \{ t > t_\alpha(n-1) \right \}(或W_1 = \left \{ t < -t_\alpha(n-1) \right \})
$$
t分布和正态分布的曲线类似，所以拒绝域的计算方式也类似，不同的是方差未知我们只能用$$S^2$$来代替$$\alpha^2$$


## 2.3 卡方$$\chi^2$$单边检验：$$\mu$$未知，关于$$\sigma^2$$的检验
### 假设检验
$$
    H_0: \sigma^2 = \sigma_0^2, H_1: \sigma^2 > \sigma_0^2(或\sigma^2 < \sigma_0^2)
$$
### 统计量
$$
    \chi^2 = \frac{(n-1)S^2}{\sigma^2} \sim \chi^2(n-1)
$$
### 拒绝域
$$
    W_1 = \left \{ \chi^2 > \chi^2_\alpha(n-1) \right \}
    (或W_1 = \left \{ \chi^2 < \chi^2_{1-\alpha}(n-1) \right \})
$$


# 三、两个独立正态分布总体均值与方差的检验
设$$X_1,X_2,X_3,...X_{n_1}$$为总体$$N(\mu_1, \sigma_1^2)$$的样本,
设$$Y_1,Y_2,Y_3,...Y_{n_2}$$为总体$$N(\mu_2, \sigma_2^2)$$的样本

## 3.1 U检验：$$\sigma_1^2, \sigma_2^2$$已知，关于$$\mu_1, \mu_2$$的检验
### 假设检验
$$
    H_0: \mu_1 = \mu_2, H_1: \mu_1 \neq \mu_2
$$
### 统计量
$$
    U = \frac{\bar{x}-\bar{y}}{\sqrt{\frac{\sigma_1^2}{n_1}+\frac{\sigma_2^2}{n_2}}} \sim N(0,1)
$$
### 拒绝域
虽然统计量的计算方程变了，但拒绝域形式不变
$$
    W_1 = \left \{  \left| u \right| \geq u_\frac{\alpha}{2} \right \}
$$


## 3.2 T检验：$$\sigma_1^2, \sigma_2^2$$未知，但已知$$\sigma_1^2 = \sigma_2^2$$，关于$$\mu_1, \mu_2$$的检验
### 假设检验
$$
    H_0: \mu_1 = \mu_2, H_1: \mu_1 \neq \mu_2
$$
### 统计量
$$
    T = \frac{\bar{x}-\bar{y}}{S_w\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}} \sim t(n_1+n_2-2) 其中 
    S_w = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1+n_2-2}
$$
$$S^2$$为$$\alpha^2$$的无偏估计
### 拒绝域
$$
    W_1 = \left \{ \left| t \right| \geq t_\frac{\alpha}{2}(n_1+n_2-2) \right \}
$$


## 3.3 T检验：$$\sigma_1^2, \sigma_2^2$$未知，但已知$$\sigma_1^2 \neq \sigma_2^2$$，关于$$\mu_1, \mu_2$$的检验
### 假设检验
$$
    H_0: \mu_1 = \mu_2, H_1: \mu_1 \neq \mu_2
$$
### 统计量
$$
    T = \frac{\bar{x}-\bar{y}}{\sqrt{\frac{S_1}{n_1} + \frac{S_2}{n_2}}} \sim t(f) 其中 
    f = \frac{(\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2})^2}{\frac{(S_1^2/n_1)^2}{n_1-1} + \frac{(S_2^2/n_2)^2}{n_2-1}}
$$
$$S^2$$为$$\alpha^2$$的无偏估计
### 拒绝域
$$
    W_1 = \left \{ \left| t \right| \geq t_\frac{\alpha}{2}(n_1+n_2-2) \right \}
$$


## 3.4 F检验(方差齐性检验)：$$\sigma_1^2, \sigma_2^2, \mu_1, \mu_2$$未知，关于$$\sigma_1^2, \sigma_2^2$$的检验
从两研究总体中随机抽取样本，要对这两个样本进行比较的时候，首先要判断两总体方差是否相同，即方差齐性。若两总体方差相等，则直接用t检验，若不等，可采用秩和检验等方法
### 假设检验
$$
    H_0: \sigma_1^2 = \sigma_2^2, H_1: \sigma_1^2 \neq \sigma_2^2
$$
### 统计量
$$
    F = \frac{S_1^2}{S_2^2} \sim F(n_1-1, n_2-1)
$$
### 拒绝域
$$
    W_1 = \left \{ 
        f \geq F_\frac{\alpha}{2}(n_1-1, n_2-1)
        或者
        f \leq F_{1-\frac{\alpha}{2}}(n_1-1, n_2-1) 
    \right \}
$$


# 四、非参检验

## 秩和检验






## T-检验：平均值的成对二样本检验

* 假设条件
* 两个总体配对差值构成的总体服从正态分布
* 配对差是由总体差随机抽样得来的
* 数据配对或匹配（重复测量（前/后））

* 统计计算


## 参考文章
[正态总体均值与方差的假设检验](https://wenku.baidu.com/view/6080ea93970590c69ec3d5bbfd0a79563c1ed48a.html)
[【Excel系列】Excel数据分析：假设检验](https://www.jianshu.com/p/1c60c9c3fe33)  
[matplotlib库的常用知识](https://www.cnblogs.com/yinheyi/p/6056314.html)

> @ WHAT - HOW - WHY  
> @ 不积跬步 - 无以至千里



