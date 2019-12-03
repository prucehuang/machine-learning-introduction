# 决策树

# Decision Trees
## 1. Training and Visualizing a Decision Tree
can perform both classification and regression tasks, and even multioutput tasks
```
tree_clf = DecisionTreeClassifier(max_depth=2)
export_graphviz(
	tree_clf,
	out_file=image_path("iris_tree.dot"),
	feature_names=iris.feature_names[2:],
	class_names=iris.target_names,
	rounded=True,
	filled=True
)
$ dot -Tpng iris_tree.dot -o iris_tree.png
```
## 2. Making Predictions
- require very little data preparation.
- don't require feature scaling or centering at all
- algorithm  
    CART, binary trees  
    ID3, mul-children trees  
- etimating class probabilities  
    根据叶子节点的value，就可以输出每个分类的概率p_k  
- gini 节点的纯洁程度，0最纯洁
$$
Gini_i=1-\sum_{k=1}^{n}p_{i,k}^2
$$
$$ p_{i,k} $$ 表示第i个节点上，第k类出现的概率

## 3. The CART Training Algorithm
递归的为每个节点寻找最好的划分特征k和划分特征的阈值t，CART Cost Function For classification
$$
J(k, t_k)=\frac{m_{left}}{m}G_{left} + \frac{m_{right}}{m}G_{right} \;\; 
(G \; measures \; the \; impurity \; of \; the \; subset)
$$
处了GINI指数可以作为G，香农信息熵也是一种方法
$$
H_i=-\sum_{k=1,p_{i,k}\neq 0}^{n}p_{i,k}log(p_{i,k})
$$
默认选择GINI指数，计算复杂度低一些，二者训练出来的树差不多，Gini impurity tends to isolate the most frequent class in its own branch of the tree, while entropy tends to produce slightly more balanced trees  
- CART 全称是 Classifcation And Regression Tree
- CART is a greedy algorithm 贪心算法  
    1. A greedy algorithm often produces a reasonably good solution, 
    2. but it is not guaranteed to be the optimal solution.  
    3. finding the optimal tree is known to be an NP-Complete problem  
    4. it requires O(exp(m)) time
- mathematical question
    1. P is the set of problems that can be solved in polynomial time
    2. NP is the set of problems whose solutions can be verified in polynomial     time
    3. NP-Hard problem is a problem to which any NP problem can be reduced in polynomial time. 
    4. An NP-Complete problem is both NP and NP-Hard

## 4. Regularization Hyperparameters
- a nonparametric model
    the number of parameters is not determined prior to training
- a few parameters restrict the shape of the Decision Tree
    1. min_samples_split
    2. min_samples_leaf
    3. min_weight_fraction_leaf, same as min_samples_leaf but expressed as a fraction of the total number of eighted instances
    4. max_leaf_nodes
    5. max_features, maximum number of features that are evaluated for splitting at each node
- increasing min_* hyperparameters or reducing max_* hyperparameters will regularize the model
- 另可以先不加任何约束训练一棵树，完成后再对树进行裁剪的方式正则化
- The computational complexity of training a Decision Tree is O(n × m log(m))

## 5. Regression
将混乱程度修改为均值平方差
```
from sklearn.tree import DecisionTreeRegressor
# setting min_samples_leaf=10 to obviously overfitting
tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)
```
返回的value值，是这一个区间内的所有samples的平均值

## 6. Instability 不确定性
- 优点 a lot going
    1. simple to understand and interpret
    2. easy to use
    3. versatile, and powerful
- 缺点 a few limitations
    1. orthogonal decision boundaries 对非线性的样本不好处理
    2. very sensitive to small variations in the training data