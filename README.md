# spam_classifer_naive_bayes

主要利用了朴素贝叶斯方法对文本进行分类。

主要步骤如下：
1 从训练文件中剥离标签和文本

2 对文本进行正则化处理，只留下字母，并全部小写

3 建立词典，并建立word：id对应关系

4 将全部文本进行数值化处理，文本的shape为[邮件数，词典长度+1].最后一维用于存储所有未出现的词的个数
  例如词典为{'this':0, 'and':1,'that':2,'where':3}
  邮件为‘this and that', 转化成数值类型为[1,1,1,0，0]
  
5 求出每个词的条件概率，也就是P[wi|Y=1]和P[wi|Y=0],为了防止之后连乘后数值太小溢出，将所有非0概率取log

6 根据贝叶斯公式，最后分别求P[Y=1|w1,w2,w3....]和P[Y=0|w1,w2,w3...],取大的值作为结果。
