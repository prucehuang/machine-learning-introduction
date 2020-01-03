# PU Learning

## 简介  
正例和无标记样本学习（Learning from Positive and Unlabled Example）简称PU或LPU学习，是一种半监督的二元分类模型，通过标注过的正样本和大量未标注的样本训练出一个二元分类器。
与普通分类问题不同，PU问题中P的规模通常相当小，扩大正样本集合也比较困难；而U的规模通常很大，比如在网页分类中，未标识的网页资源可以非常廉价、方便的从网络中获取。引入U的目的就是降低人工分类的预备工作量，同时提高精度，尽可能达到自动分类的效果。

## PU 学习方法
PU学习最终被转换成一个有限制的最优化问题（Constrained Optimization Problem），即算法试图在使得正例数据中的错误率低于1-r的情况下最小化无标注数据中正例数据的数目  
基于限制最优化问题，可采用多种方法来建立PU分类器：  
- 直接方法（Direct Approach）[Learning classifiers from only positive and unlabeled data](https://dl.acm.org/citation.cfm?id=1401920)  
- 两步方法（Two-step Approach）[An Evaluation of Two-Step Techniques for Positive-Unlabeled Learning in Text Classification](https://pdfs.semanticscholar.org/bd10/ba5f30744e4755cbe7757e8c657ce5d6ec45.pdf)  
-  PU bagging [A bagging SVM to learn from positive and unlabeled examples ，PRL 2014](https://www.sciencedirect.com/science/article/abs/pii/S0167865513002432)  
-  Positive unlabeled random forest [Towards Positive Unlabeled Learning for Parallel Data Mining: A Random Forest Framework](https://www.researchgate.net/publication/269040485_Towards_Positive_Unlabeled_Learning_for_Parallel_Data_Mining_A_Random_Forest_Framework)

### Direct Approach
直接利用标准分类方法是这样的：将正样本和未标记样本分别看作是positive samples和negative samples, 然后利用这些数据训练一个标准分类器。分类器将为每个物品打一个分数（概率值）。通常正样本分数高于负样本的分数。因此对于那些未标记的物品，分数较高的最有可能为positive  
#### 有偏的 SVM 算法
由于正样本集合P中难免会有一些噪声，以及正负样本的数据倾斜，可使用一种有偏的SVM算法，使用C+和C-分别控制正负样本的误差(直观上看，对C+取一个较大的值，而C-取一个较小的值)：
为了选择合适的C+和C-，通常用验证集来评估在不同取值下的分类器性能。性能指标可采用F值（F=2pr/(p+r)，其中，p为准确率，r为召回率）。但又个问题：验证集中没有负样本，如何评估F值呢？
“Learning with positive and unlabeled examples using weighted logistic regression”中给出了一种直接使用不含负样本的验证集来评估分类器性能的方法。判断准则是采用：pr / Pr[Y=1]（其中，Pr[Y=1]是正样本的实际概率），其等价于r2 / Pr[f(X)=1] （其中，Pr[f(X)=1]是一个样本被分为正样本的概率）。其中，r可以用验证集中的正样本来计算得到，Pr[f(X)=1]可以由整个验证集计算得到。这个判断指标随p和r的增加而增加，随p或r中任一个减小而减小，所以可以像F值一样来判断分类器的性能。

### Two-step Approach
- 根据已标注过的正样本P在未标注样本集U中找出可靠的负样本集合(Reliable Negative Examples，简称RN)，将PU问题转化为二分类的问题；
- 利用正负样本通过迭代训练得到一个二元分类器。  
理论上已经证明：如果最大化未标注样本集U中负样本的个数，同时保证正样本被正确分类，则会得到一个性能不错的分类器。
上述两个步骤中，找出RN以及训练二元分类器都有很多方法可以选择，下面对这些方法进行简单的介绍。
#### 计算 RN
##### Spy 算法
> RN集合置空
> 从P中随机选取子集S（比例一般为15%），得到新的正样本集PS=P-S和负样本集US=U+S
> 使用贝叶斯算法计算每个样本的概率值
> 使用间谍子集S确定出一个概率阈值th，如果样本的概率小于阈值概率th，则把其加入RN集合。

##### Cosine-Rocchio 技术
> 从U中使用Cosine距离计算出一些潜在的负样本(potential negatives，PN)
> 使用P、PN训练一个Rocchio的分类器
> 使用分类器从U中选出RN

##### 1-DNF 算法
> 寻找一些正样本中出现频率很高的词，作为正样本词，正特征(Positive Feature, PF)
> 在U中选出完全不包含正样本词的样本作为RN

##### 朴素贝叶斯（Naive Bayesian，NB）
> 把P作为正样本，U作为负样本；
> 使用P和U训练得到贝叶斯分类器；
> 对U中的每个样本使用上述分类器进行分类，如果分类结果为-1，则把该样本加入RN。

##### Rocchio
> 把P作为正样本，U作为负样本；
> 使用P和U训练得到Rocchio分类器；
> 对U中的每个样本使用上述分类器进行分类，如果分类结果为-1，则把该样本加入RN。

#### 训练分类器
##### SVM
如果RN样本足够，直接使用SVM对P和RN进行训练得到分类器

##### EM-NB
EM算法主要由Expectation和Maximization两个步骤组成。前者对缺失标签的数据打上标签；后者则用全部数据一起计算模型参数。算法步骤描述如下：
>  P为正样本， RN为负样本，Q=U-RN中为无标签样本
>  使用P、RN训练一个NB分类器，给Q的样本预测一个概率 — — Expectation
>  从Q中选取负样本加RN，循环第一步 — — Maximization

##### I-SVM
>  P为正样本， RN为负样本，Q=U-RN中为无标签样本
>  使用P、RN训练一个SVM分类器，给Q的样本预测一个概率 
>  从Q中选取负样本加RN，循环第一步

##### PEBL 算法
PEBL算法主要思想是使用SVM迭代地从U-RN中抽取出更多的负样本，并把它们放到RN集合中，直至U-RN中没有可以被分为负样本的数据为止。算法步骤如下：
对 P 中的每个样本标记为类别 1；
对 RN 中的每个样本标记为类别-1；
令 i=1，Q=U-RN，开始以下的循环： 
使用 P 和 RN 训练一个 SVM 分类器 Si；
使用 Si 对 Q 中的样本进行分类，把其中所以分类为-1 的样本记为 W；
如果 W 为空，则结束循环；否则：Q = Q-W, RN = RN ∪ W, i = i + 1

### 4. Roc-SVM 算法
PEBL算法中得到的最后一个分类器不一定是最优分类器，为此，对该算法进行一些改进，得到了Roc-SVM算法。算法步骤如下：
使用 PEBL 算法直至收敛，记最后一个分类器为 S_last；
使用 S_last 对 P 进行分类；
如果 P 中超过 8%的样本被分为负样本，则选择 S1 作为最终的分类器；否则，选择 S_last 作为最终分类器。
由于SVM算法对噪声很敏感，如果在迭代过程中，把Q中的一些正样本错分为-1而划分到RN中，那么会导致最终的分类器S_last性能很差，这也是PEBL算法的一个缺陷。为此，需要对S_last的分类性能进行评估，看是否选择其作为最终分类器。选择8%作为判断阈值也是一个保守的做法，以防选择一个性能不好的分类器。
上述的选择S1或S_last的做法其实还是欠妥，因为这两个分类器可能性能都很差。S1性能差是因为初始的RN集合中可能包含很少的负样本，不能体现出负样本的整体分布情况；S_last性能差则是由于PEBL算法在某个迭代过程中把一些正样本错分到RN中。为此，我们可以选择使用Spy或Rocchio算法得到初始的RN，这样可以使S1、更加稳健。有一点要注意的是：多数情况下，最佳分类器可能并不是S1或S_last，而是在迭代过程中产生的某一个分类器，然而，这个最佳分类器却是很难被“catch”的。

### PU bagging
一个更加复杂的方法来解决PU问题是采用bagging的变种：
a)、通过将所有正样本和未标记样本进行随机组合来创建训练集。
b)、利用这个“bootstrap”样本来构建分类器，分别将正样本和未标记样本视为positive和negative。
c)、将分类器应用于不在训练集中的未标记样本 - OOB（“out of bag”）- 并记录其分数。
d)、重复上述三个步骤，最后为每个样本的分数为OOB分数的平均值

> 参考文章  
[PU learning简介（附python代码）](https://www.wandouip.com/t5i225512/)  
[Positive-unlabeled learning -- 知乎](https://zhuanlan.zhihu.com/p/42435966)  
[基于 PU-Learning 的分类方法](http://blog.xiaoduoai.com/?p=344)  
[PU learning算法简介](https://blog.csdn.net/wangyiqi806643897/article/details/46341189)  
[PU learning techniques applied to artificial data ipynb](https://github.com/roywright/pu_learning/blob/master/circles.ipynb)  

> 参考文献  
Bing Liu, Yang Dai, Xiaoli Li, Wee Sun Lee and and Philip Yu. “Building Text Classifiers Using Positive and Unlabeled Examples”.
Proceedings of the Third IEEE International Conference on Data Mining (ICDM-03), Melbourne, Florida, November 19-22, 2003  
[PU Learning - Learning from Positive and Unlabeled Examples ](https://www.cs.uic.edu/~liub/NSF/PSC-IIS-0307239.html)