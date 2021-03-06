# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:50:26 2017

@author: caoxun
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:22:07 2017

@author: caoxun
"""

import codecs
import jieba
import re  


#读取数据   
shoes = codecs.open('C:\\Users\\caoxun\\Desktop\\淘宝评论project\\鞋子评论3.csv', mode='r', errors='strict', buffering=1)

#初始化    
shoes_new =""

#合并行
for line in shoes.readlines():
    shoes_new += "".join(line.split('\n'))
    
#去除标点
#shoes_new = re.sub(',','',shoes_new)
#print(re.sub('～','',shoes_new))
#用[]一次性替换多个标点
#去掉特殊符号
shoes_new = re.sub('[_-╯﹏╰()xgjjjnbgggf!,~&middot&middot]','',shoes_new)
#按照表示分句的标点将string分割成list，这样list上的每一位就为一个短句了
#import re
#shoes_new =  re.split(r'[～，;！。?？、...]', shoes_new)
#shoes_new_str = ''.join(shoes_new)

#导入人工定义的字典,提高分词精准率
jieba.load_userdict('C://Users//caoxun//Desktop//淘宝评论project//newDict.txt')
#分词
segList = jieba.cut(shoes_new,cut_all=False) #=False > 精准模式

#分词结果储存
segResult = " ".join(segList)
#打印分词结果
print (segResult)

#将停用词txt读取成为string变量
with open('C://Users//caoxun//Desktop//淘宝评论project//Chinese Stop Words.txt', 'r') as myfile:
stopwords=myfile.read().replace('\n', '')
      
      
#删除停用词
#segList = segResult.split()
#stopwordsSet = set(stopwords)
#noStop_list=[seg for seg in segList if seg not in stopwordsSet]
#noStop= " ".join(noStop_list)
noStop=segResult
#去除多余空格，避免词库报错
noStop = re.sub('     ',' ',noStop)

#输出分词结果为txt
import codecs

f = codecs.open("C://Users//caoxun//Desktop//淘宝评论project//segResult4.txt", "w",'utf-8')
f.write(segResult)
f.close()


f =codecs.open("C://Users//caoxun//Desktop//淘宝评论project//noStop4.txt", "w",'utf-8') 
f.write(noStop)
f.close()



#使用word2vector
from gensim.models import word2vec
import logging
#from gensim import corpora

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus(r"C:\Users\caoxun\Desktop\淘宝评论project\noStop3.txt",)#需要读取成utf-8，现在前面文件写入时设置好
#sentences = LoadCorpora('r"C:\Users\caoxun\Desktop\淘宝评论project\segResult3")
# 加载语料
model = word2vec.Word2Vec(sentences,min_count=2, size=200) #200维的词向量
model.most_similar(u"鞋子", topn=20)




#将model中的词和对应向量保存
model.save('mymodel')
#读取model
model = word2vec.Word2Vec.load('mymodel') #can continue training with the loaded model!


#以短句为单位，计算这些短句的均值向量
#提取出词向量保存 
word_vectors = model.wv
#提出词语
words = model.wv.vocab
#提出vector
wordsVect = model.wv.syn0

import re
#noStop2 =re.sub("[\.\!\/_$%^*(+\"\'+——！、@#￥%& ? ？（）~]+", " 。",noStop)
#noStop2 = re.split(r"[～，;！。?？?、...？]+", noStop)
#noStop2 =  re.split(r'[～，;！。?？?、...？?。;?！ ]*', noStop2)
#shoes_new_str = ''.join(shoes_new)
#用正则替换清理标点
noStop2 = re.sub(r"[～，;！。?？?、？]+", " 。",noStop)
noStop2 =  re.split(r'[。]+', noStop2)


#按照短句计算短句的平均值特征向量

#初始化feature vector

import numpy as np
featureVec = np.zeros((200,),dtype="float32")
commentFeatureVecs = []


#遍历noStop2小短句中的每一句
for i in range(len(noStop2)):
    #这个小短句中的词组数 = 空格数-1
    wordNum = (noStop2[i].count(' ')) -1
    #遍历当前小短句里的每一个词
    for word in noStop2[i].split():
        #如果当前的词在w2v里的词典里
        if word in words:
            featureVec = np.add(featureVec,model[word])
    if wordNum != 0:
    #只有在上一个for loop遍历完noStop2[当前的i]后才执行下一步
        featureVec = np.divide(featureVec,wordNum)      
    commentFeatureVecs.append(featureVec.tolist())
                
#按照comment Feature Vecs 来聚类
#K-means聚类
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#按照k-means k的取值进行迭代并画图
#Elbow方法确定k值
#计算sum of squares between groups 除以 the sum of squares total所得的商
from scipy.spatial.distance import cdist, pdist

K = range(1,30)
KM = [KMeans(n_clusters=k).fit(commentFeatureVecs) for k in K]
centroids = [k.cluster_centers_ for k in KM]

D_k = [cdist(commentFeatureVecs, cent, 'euclidean') for cent in centroids] #对于不同的k,计算每个分句和cluster中心的欧式距离
cIdx = [np.argmin(D,axis=1) for D in D_k] #返回最小距离的index, for different k
dist = [np.min(D,axis=1) for D in D_k] #返回最小距离, for different k
avgWithinSS = [sum(d)/len(commentFeatureVecs) for d in dist] #计算SSE的均值

# Total with-in sum of square
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(commentFeatureVecs)**2)/len(commentFeatureVecs) #compute pairwise distance
bss = tss-wcss

#
# elbow curve,根据图像估测曲线在k=15后趋于平稳

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, avgWithinSS, 'b*-')
ax.plot(K[20], avgWithinSS[20], marker='o', markersize=12, 
markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for KMeans clustering')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, bss/tss*100, 'b*-')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')
plt.title('Elbow for KMeans clustering')



#选定k值，再一次聚类建模
#so, we choose k=20
num_clusters = 20
# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans(n_clusters = num_clusters)
idx = kmeans_clustering.fit_predict(commentFeatureVecs) 

wordDict = dict(zip(idx,commentFeatureVecs))


#按照类目数目，打印出每一类中的词组
#初始化一个list of lists,然后按照cluster number依次传入
test = [[] for i in range(0,num_clusters-1)]
for cluster in range(0,num_clusters-1):
    for i in range(len(idx)):
        if idx[i]==cluster:
            #contentTemp.append()
            test[cluster].append(noStop2[i])

#打印出每个cluster的内容
for i in range(0,num_clusters-1):
    print("Cluster" ,i, test[i]) #固定字符和变量的联合打印
            

from collections import Counter 
freqcount = Counter(test[1]).most_common(20)
print( freqcount)  
#[('  ', 19),
# (' 穿着 很 舒服  ', 11),
# (' 不错  ', 8),
# (' 鞋子 收到 了  ', 7),
# (' 大小 合适  ', 5),
# (' 轻便  ', 5),
#(' 赞 一个  ', 5),
#(' 但是 其他 地方 刚好 合适  ', 4),
#(' 是 正品  ', 4),
#(' 物流 很快  ', 4),
#(' 鞋口 有点 小  ', 3),
#(' 穿起来 很 舒服  ', 3),
#(' 一个 字  ', 3),
#(' 穿 的 非常 舒服  ', 3),
#(' 这个 价格 不值得 买  ', 3),
#(' 鞋子 穿 的 很 舒服  ', 3),
#(' 但是 真的 帅  ', 3),
#(' 我 第一次 穿 这样 鞋口 的 鞋  ', 3),
#(' 透气  ', 3),
#(' 超级 喜欢  ', 3)]


#提高频的词，去掉次数
freqwords = []
for j in range(0,19):
       freqwords.append(freqcount[j][0])
    
    
    
    
    
    
    
    
    
    
    
    
    
                    
