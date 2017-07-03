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
shoes_new = re.sub(',','',shoes_new)
print(re.sub('～','',shoes_new))
#用[]一次性替换多个标点
shoes_new = re.sub('[～，;！。?？、..._-╯﹏╰()xgjjjnbgggf!,~&middot&middot]','',shoes_new)


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
segList = segResult.split()
stopwordsSet = set(stopwords)
noStop_list=[seg for seg in segList if seg not in stopwordsSet]
noStop= " ".join(noStop_list)


#输出分词结果为txt
import codecs

f = codecs.open("C://Users//caoxun//Desktop//淘宝评论project//segResult3.txt", "w",'utf-8')
f.write(segResult)
f.close()

f =codecs.open("C://Users//caoxun//Desktop//淘宝评论project//noStop3.txt", "w",'utf-8') 
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
#保存模型 save the model
#model.save("file://C:/Users/caoxun/Desktop/淘宝评论project/model")
#load the model
#model = word2vec.Word2Vec.load("file://C:/Users/caoxun/Desktop/淘宝评论project/model")

#[('买的', 0.9999780058860779),
#('正品', 0.9999773502349854),
#('穿', 0.9999763369560242),
#('质量', 0.9999760389328003),
#('不错', 0.9999759197235107),
#('顺丰', 0.9999749660491943),
#('快递', 0.9999749660491943),
#('喜欢', 0.999973714351654),
#('感觉', 0.9999736547470093),
#('一个', 0.999973475933075),
#('舒服', 0.9999730587005615),
#('舒适', 0.9999728202819824),
#('鞋底', 0.9999727606773376),
#('买', 0.9999727606773376),
#('可以', 0.9999725818634033),
#('穿起来', 0.9999715685844421),         
#('物流', 0.9999715089797974),
#('穿着', 0.9999712705612183),
#('不知道', 0.9999710917472839),
#('客服', 0.9999710321426392)]


#去除多余空格，避免词库报错
noStop = re.sub('     ',' ',noStop)

#提取出词向量保存 
word_vectors = model.wv
vector = []
for word in noStop :  
    if word in model.wv.vocab:
       vector.append(word_vectors[word]) 
       
#将model中的词和对应向量保存
model.save('mymodel')
#读取model
model = word2vec.Word2Vec.load('mymodel') #can continue training with the loaded model!

#提出词语
#model.wv.vocab
#提出vector
#model.wv.syn0


#K-means聚类
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#取出由word2vec生成的词向量
word_vectors = model.wv.syn0



#按照k-means k的取值进行迭代并画图
from scipy.spatial.distance import cdist, pdist

K = range(1,50)
KM = [KMeans(n_clusters=k).fit(word_vectors) for k in K]
centroids = [k.cluster_centers_ for k in KM]

D_k = [cdist(word_vectors, cent, 'euclidean') for cent in centroids] #Computes distance between each word and cluster centroid， for different k
cIdx = [np.argmin(D,axis=1) for D in D_k] #返回最小距离的index, for different k
dist = [np.min(D,axis=1) for D in D_k] #返回最小距离, for different k
avgWithinSS = [sum(d)/word_vectors.shape[0] for d in dist] #计算SSE的均值

# Total with-in sum of square
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(word_vectors)**2)/word_vectors.shape[0] #compute pairwise distance
bss = tss-wcss

#
# elbow curve,根据图像估测曲线在k=20后趋于平稳

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
#so, we choose k=13
num_clusters = 20
# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans(n_clusters = num_clusters)
idx = kmeans_clustering.fit_predict(word_vectors)

wordDict = dict(zip(model.wv.index2word, idx))

#return dictionary key by its value
#将dict变为list
wordDict_list = list(wordDict)
