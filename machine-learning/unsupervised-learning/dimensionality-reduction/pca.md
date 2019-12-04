# PCA

# Dimensionality Reduction

## 1. The Curse of Dimensionality
- The main motivations for dimensionality reduction
    1. To speed up a subsequent training algorithm (in some cases it may even  remove noise and redundant features, making the training algorithm perform better)
    2. To visualize the data and gain insights on the most important features
    3. Simply to save space (compression)
- The main drawbacks
    1. Some information is lost, possibly degrading the performance of subse‐
    2. quent training algorithms
    3. It can be computationally intensive
    4. It adds some complexity to your Machine Learning pipelines Transformed features are often hard to interpret

## 2. Main Approaches for Dimensionality Reduction
- Projection 投影    
    Many features are almost constant，highly correlated，将高纬数据投影到低纬度数据
- Manifold Learning  
    d-dimensional的数据在n-dimensional的空间卷起来，然后可以压缩回d-dimensional，假设高纬的数据是由低纬的数据变换来的  
    if you reduce the dimensionality of your training set before training a model, it will definitely speed up training, but it may not always lead to a better or simpler solution; it all depends on the dataset

## 3. PCA
- 主要思想  
    First it identifies the hyperplane that lies closest to the data 找到最优的超平面, preserves the maximum amount of Variance，then it projects the data onto it 将数据投影上去
- Principal Components  
    The unit vector that defines the ith axis is called the ith principal component (PC) 主成分是投影平面的单位坐标轴向量，n维平面有n个主成分向量，主成分的方向不重要，重要的是定义的平面
```
# Singular Value Decomposition (SVD)求解矩阵的主成分向量
X_centered = X - X.mean(axis=0) #主成分要求数据以原点为中心
U, s, V = np.linalg.svd(X_centered)
c1 = V.T[:, 0]
c2 = V.T[:, 1]
W2 = V.T[:, :2]
X2D = X_centered.dot(W2)
```
$$
V^T=\begin{pmatrix}
| & | &  & | \\ 
c_1 & c_2 & \cdots & c_n\\ 
| & | &  & | 
\end{pmatrix}
$$
$$
X_{d-proj}=X \cdot W_d
$$
- Projecting Down to d Dimensions (m, d) = (m, n) · (n, d)
```
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X2D = pca.fit_transform(X)
# automatically takes care of centering the data
pca.components_.T[:, 0])
```
- Choosing the Right Number of Dimensions 通过PC的方差占比选择压缩的维度d
```
pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
print(pca.explained_variance_ratio_)
array([ 0.84248607, 0.14631839])
```
或者直接设置我们要保存的数据方差SUM
```
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)
```
或者绘制维度-方差和曲线来选择拐点
- PCA inverse transform 从(m, d)=(m, n)·(n, d) 到 (m, n)=(m, d)·(d, n)
```
# 将压缩的数据反转回去 
pca = PCA(n_components = 154)
X_mnist_reduced = pca.fit_transform(X_mnist)
X_mnist_recovered = pca.inverse_transform(X_mnist_reduced)
```
the reconstruction error  
The mean squared distance between the original data and the reconstructed data
$$
X_{recovered} = X_{d-proj} \cdot W_d^T
$$
- Incremental PCA 在线PCA  
    增强PCA，IPCA，支持在线增量fit，专门用于处理超大数据和在线学习情况
```
from sklearn.decomposition import IncrementalPCA
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_mnist, n_batches):
	inc_pca.partial_fit(X_batch)
X_mnist_reduced = inc_pca.transform(X_mnist)
```
或者用memmap, 支持将数据以二进制的形式存储在磁盘上, 按需加载进内存使用
```
X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))
batch_size = m // n_batches
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mm)
```
- Randomized PCA 随机PCA，当你确定要快速大量缩减维度的时候    
    a stochastic algorithm that quickly finds an approximation of the first d principal components，O(m × d x d) + O(d x d x d), instead of O(m × n x n) + O(n x n x n)
```
rnd_pca = PCA(n_components=154, svd_solver="randomized")
X_reduced = rnd_pca.fit_transform(X_mnist)
```
- Kernel PCA 核PCA，主要用于非线性数据  
    将核函数运用到PCA中，得到 KPCA
```
from sklearn.decomposition import KernelPCA
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)
```
Selecting a Kernel and Tuning Hyperparameters  
有label，训练一个分类器，对比准确率来选择KPCA参数
```
clf = Pipeline([
	("kpca", KernelPCA(n_components=2)),
	("log_reg", LogisticRegression())
])
param_grid = [{
	"kpca__gamma": np.linspace(0.03, 0.05, 10),
	"kpca__kernel": ["rbf", "sigmoid"]
}]
grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)
print(grid_search.best_params_)
```
无label，将原始数据作为label，训练一个回归模型  
```
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433,
	fit_inverse_transform=True)
X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced)
mean_squared_error(X, X_preimage)
```

## 4. LLE
- a Manifold Learning technique  

first measuring how each training instance linearly relates to its closest neighbors（c.n）
$$
\begin{aligned}
&{\hat{W} = \underset{w}{argmin} \sum_{i=1}^{m}
\left \|  
x^{(i)}-\sum_{j=1}^{m}\hat{w}_{i,j}x^{(j)}
\right \|^2} \\
& subject \; to \; \left\{\begin{matrix}
w_{i,j}=0 & if \; x^{(j)} \; is \; not \; one \; of \; the \; k \; c.n \; of \; x^{(i)}\\ 
\\
\sum_{j=1}^{m}\hat{w}_{i,j}=1 & for \; i=1,2,\cdots,m 
\end{matrix}\right.
\end{aligned}
$$
then looking for a low-dimensional representation of the training set  
尤其擅长展开卷状数据
$$
\hat{Z} = \underset{z}{argmin} \sum_{i=1}^{m}
\left \|  
z^{(i)}-\sum_{j=1}^{m}\hat{w}_{i,j}z^{(j)}
\right \|^2
$$
```
from sklearn.manifold import LocallyLinearEmbedding
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_reduced = lle.fit_transform(X)
```
- computational complexity, scale poorly to very large datasets
    1. O(m log(m)n log(k)) for finding the k nearest neighbors
    2. O(mnk^3) for optimizing the weights
    3. O(dm^2) for constructing the low-dimensional representations

## 5. Other Dimensionality Reduction Techniques
- Multidimensional Scaling (MDS)   
    reduces dimensionality while trying to preserve the distances between the instances 
- Isomap  
    trying to preserve the geodesic distances9 between the instances
- t-Distributed Stochastic Neighbor Embedding (t-SNE)  
    trying to keep similar instances close and dissimilar instances apart mostly used for visualization
- Linear Discriminant Analysis (LDA)  
    is actually a classification algorithm  
    the projection will keep classes as far apart as possible