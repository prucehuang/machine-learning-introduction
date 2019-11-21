# 损失函数

### 1.0-1损失函数 （0-1 loss function） 
$$L(Y,f(X))=\left\{\begin{matrix}
 1,& Y \neq f(X) \\ 
 0,& Y = f(X)
\end{matrix}\right.$$
### 2.平方损失函数（quadratic loss function) 
$$L(Y,f(X))=(Y−f(X))^2$$
### 3.绝对值损失函数(absolute loss function) 
$$L(Y,f(x))=|Y−f(X)|$$
### 4.对数损失函数（logarithmic loss function) 或对数似然损失函数(log-likehood loss function) 
$$L(Y,P(Y|X))=−logP(Y|X)$$

逻辑回归中，采用的则是对数损失函数。如果损失函数越小，表示模型越好
