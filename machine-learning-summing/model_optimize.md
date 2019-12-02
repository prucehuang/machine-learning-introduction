# 优化算法

## 一、主要解决的挑战
### 1. bad data  
- Insufficient Quantity of Training Data 数据不够  
- Nonrepresentative Training Data 数据不具有代表性
- Poor-Quality Data 很多脏数据
- Irrelevant Features 一些毫无相关性的属性
### 2. bad algorithm
#### 1）Overftting the Training Data 过拟合  
Overfitting happens when the model is too complex relative to the amount and noisiness of the training data. 
> The possible solutions  

- To simplify the model, by reducing the number of attributes in the training data or by constraining the model(regularization)
- To gather more training data
- To reduce the noise in the training data (e.g., fix data errors and remove outliers)

#### 2）Underftting the Training Data 欠拟合  
> The main options to fix this problem

- Selecting a more powerful model, with more parameters
- Feeding better features to the learning algorithm (feature engineering)
- Reducing the constraints on the model (e.g., reducing the regularization hyperparameter)

## 二、Classification Performance Measures评估
### 准确率和召回率的权衡
- 交叉验证矩阵，从左往右，从上往下分别是  

    |         | 预测-假         | 预测-真         |
    | ------- | --------------- | --------------- |
    | 真实-假 | true negatives  | false positives |
    | 真实-真 | false negatives | true positives  |

- 准确率 precision = TP/(TP + FP)  
    预测是1的数据里面有多少是真1  
    from sklearn.metrics import precision_score
    
- 召回率 recall = TP /(TP + FN )  
    真1里面有多少被发现找到了  
    from sklearn.metrics import recall_score
    
- increasing precision reduces recall, and vice versa  
    1) 用阈值来判断分类结果，阈值越大结果越准（准确率越高），选出来的正例越少（召回率越低）  
    2) 画出阈值 -（准确率、召回率、F1）曲线  
    3) 画出recall-precision曲线  
    4) 阈值取多少，取决于你的系统要求  
        寻找犯罪，要求高recall，宁可错抓一把，不能发过一个；推荐系统，更在乎准确率
    
- F1 = 2 / [(1/precision) + (1/recall)] = 2PR/(P+R)  
from sklearn.metrics import f1_score

### ROC(The receiver operating characteristic)的权衡
- ROC曲线  
    1) 横坐标是 FPR（False Positive Rate）  
        True Negative Rate 又叫特异度Specificity，1-TNR = FPR   
    2) 纵坐标是 TPR（True Positive Rate）召回率  
    3) the higher the recall (TPR), the more false positives (FPR) the classifier produces   
4) 随机分类算法的ROC曲线是一条直线，a good classifier stays as far away from that     line as possible (toward the top-left corner)
- ROC曲线的面积，即AUC（the area under the curve）
    1) AUC is one way to compare classifiers 
    2) A perfect classifier will have a ROC AUC equal to 1
    3) a purely random classifier will have a ROC AUC equal to 0.5

### PR曲线 VS ROC曲线
- PR曲线选阈值
- ROC曲线选模型

> @ 学必求其心得，业必贵其专精
> @ WHAT - HOW - WHY
> @ 不积跬步 - 无以至千里