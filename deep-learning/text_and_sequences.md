[TOC]

# 第一部分 文本处理
## 一、处理文本数据
文本向量化的几种思路
- 文本分割成单词，单词向量化
- 文本分割成字符，字符向量化
- 文本分割成单词或字符的n-gram，将每个n-gram向量化（n-gram是从句子中提取N个或更少的连续单词的集合，如对于句子a b c d， 2-gram提取的集合是{a, ab, b, bc, c, cd, d}，该集合叫二元语法袋）  

### 1.1 one-hot 编码
- one-hot

- 散列one-hot


### 1.2 词嵌入，也叫词向量
one-hot编码得到的向量是二进制的、稀疏的、维度很高的，而词嵌入是低维的密集浮点数向量  
![word_embeddings](http://km.oa.com/files/photos/pictures/202007/1594688053_28_w5018_h5030.png)
词嵌入（word embedding或word vector）是   ，



从数据中学习得到的，常见的维度256、512、1024  

- 直接在网络中增加Embedding层，在完成主任务的时候同时学习词嵌入

- 直接将其他预训练好的词嵌入，直接用于模型，预训练词嵌入（ pretrained word embedding）  


### 1.3 整合在一起：从原始文本到词嵌入

# 第二部分 循环神经网络
## 一、理解循环神经网络
循环神经网络（RNN、recurrent neural network）是具有内部环的神经网络，上一层的输出作为下一层的状态输入，状态输入+本层输入得到本层输出
![rnn](http://km.oa.com/files/photos/pictures/202007/1594773392_52_w728_h292.png)
### 1.1 SimpleRNN


### 1.2 LSTM
随着层数的增加容易出现梯度消失，增加网络层数将变得无法训练，继而就有了长短期记忆（LSTM，long short-term memory)。LSTM增加了一种携带信息跨越多个时间步的方法。
![LSTM](http://km.oa.com/files/photos/pictures/202007/1594774451_92_w778_h319.png)

### 1.3 GRU

## 循环神经网络的高级用法
### 使用循环dropout 来降低过拟合
### 循环层堆叠
### 使用双向RNN

# 第三部分 使用一维卷积神经网络
## 用卷积神经网络处理序列
时间可以理解为一个空间维度
### 理解序列数据的一维卷积
### 序列数据的一维池化
### 实现一维卷积神经网络
### 结合CNN 和RNN 来处理长序列















> @ WHAT - HOW - WHY  
> @ 不积跬步 - 无以至千里  
> @ 学必求其心得 - 业必贵其专精  