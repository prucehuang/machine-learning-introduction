# Go Through an ML project
## 1. Look at the big picture.
- Frame the Problem  
    what exactly is the business objective
- Select a Performance Measure  
    Root Mean Square Error (RMSE) - L2范式  
    Mean Absolute Error (MAE) - L1范式  
- Check the Assumptions

## 2. Get the data.
### Create the Workspace
Anaconda ( Jupyter | Spyder )

### Download the Data
Popular open data repositories
- UC Irvine Machine Learning Repository
- Kaggle datasets
- Amazon's AWS datasets  

Meta portals (they list open data repositories)
- http://dataportals.org/
- http://opendatamonitor.eu/
- http://quandl.com/

Other pages listing many popular open data repositories
- Wikipedia's list of Machine Learning datasets
- Quora.com question
- Datasets subreddit

### Take a Quick Look at the Data Structure
head()、info()、['key'].value_counts() 统计值出现的次数、describe()-shows a summary of the numerical attributes、hist()  
Jupyter's magic command "%matplotlib inline"

### Create a Test Set
pick 20% of the dataset randomly, and set them aside  
```
train_set, test_set = sklearn.model_selection.train_test_split(housing, test_size=0.2, random_state=42)
层次取样 
split = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
```

## 3. Discover and visualize the data to gain insights.
- Visualizing Geographical Data
- Looking for Correlations  
    corr() 查看各个特征与当前特征的关系  
    pandas.tools.plotting.scatter_matrix() 查看n个特征两两之间的关系，并plot绘图
- Experimenting with Attribute Combinations  
    such as bedrooms_per_room better than the total number of rooms

## 4. Prepare the data for Machine Learning algorithms.
### Data Cleaning
- Get rid of the corresponding districts.  
    housing.dropna(subset=["total_bedrooms"]) # option 1
- Get rid of the whole attribute.  
    housing.drop("total_bedrooms", axis=1) # option 2
- Set the values to some value (zero, the mean, the median, etc.)  
    比如 #直接设置空值为平均数  
    median = housing["total_bedrooms"].median()  
    housing["total_bedrooms"].fillna(median) # option 3  
    或者 #使用Imputer管理转换空值  
    imputer = sklearn.preprocessing.Imputer(strategy="median")  
    X = imputer.fit_transform(housing_num)  
    housing_tr = pd.DataFrame(X, columns=housing_num.columns)  

### Handling Text and Categorical Attributes
convert these text labels to numbers

- sklearn.preprocessing.LabelEncoder将类别转换成[0, 1, 2,,,]  
    【问题】类别值只代表某一个类，大小无意义，但是用数值可能会有误导
- sklearn.preprocessing.OneHotEncoder 将类别转换成OneHot形式  
    sklearn.preprocessing.OneHotEncoder #1  
    sklearn.preprocessing.LabelBinarizer #2  
    #1 和 #2 都可以将数值型、文本型数据转为OneHot形式，区别是#1输入二维数组，#2输入一维数组（0.22.x版本之前categories维度判断，#1用的是Max(value)只能处理数值型，升级之后用的是Unique(value)就OK了）

### Custom Transformers 订制转换
sklearn.base.BaseEstimator、sklearn.base.TransformerMixin

### Feature Scaling
- min-max scaling ( subtracting the min value and dividing by the max minus the min )  
sklearn.preprocessing.MinMaxScaler
- standardization ( subtracts the mean value (so standardized values always have a zero mean), and then it divides by the variance so that the resulting distribution has unit variance )  
    sklearn.preprocessing.StandardScaler

标准化对比MAX-MIN来说，异常值的容错性更大；但值域却不在[0, 1]

### Transformation Pipelines
- sklearn.pipeline.Pipeline 串行Pipeline
- sklearn.pipeline.FeatureUnion 并行Pipeline  
Pipeline要求流程中出最后一个步骤以外，之前的步骤都必须是transformers，每一步都会调用fit_transform()，并将输出作为下一次输入

## 5. Select a model and train it.
### Training and Evaluating on the Training Set
- underfitting  
    select a more powerful model  
    better features  
    reduce the constraints(regularized) on the model  

- overfitting  
    simplify the model  
    gather more training data  
    reduce the noise in the training data  
    add the constraints(regularized) on the model  

### Better Evaluation Using Cross-Validation
sklearn.model_selection.cross_val_score 交叉训练  
an estimate of the performance of model  
a measure of how precise this estimate is

### model save ( sklearn.externals.joblib )

## 6. Fine-tune your model.
- Grid Search（超参不多）  
    fiddle with the hyperparameters manually 手动调整超参  
    sklearn.model_selection.GridSearchCV 设定多组超参数，自动遍历，对比
- Randomized Search（超参很多）  
    sklearn.model_selection.RandomizedSearchCV 随机的选择组合超参对比  
    1,000 iterations，可以迭代一千次，尝试一千个随机值，而不是指定的特定值；在有足够的资源尝新时，可能寻到更好的参数组合
- Ensemble Methods
- Analyze the Best Models and Their Errors
- Evaluate Your System on the Test Set

## 7. Present your solution.
- high‐lighting what you have learned
- what worked and what did not
- what assumptions were made
- what your system's limitations are
- document everything
- create nice presentations with clear visualizations and easy-to-remember statements

## 8. Launch, monitor, and maintain your system.
- check your system's live performance at regular intervals and trigger alerts when it drops
- evaluate the system's input data quality.
- setting up human evaluation pipelines
- automating regular model training

> @ 学必求其心得，业必贵其专精
> @ WHAT - HOW - WHY  
> @ 不积跬步 - 无以至千里