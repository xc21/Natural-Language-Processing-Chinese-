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
shoes_new = re.sub('[～，;！。?？、╯﹏╰xgjjjnbgggf!,~&middot&middot]','',shoes_new)


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
from sklearn.cluster import KMeans
# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.wv.syn0
# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans(n_clusters = num_clusters)
idx = kmeans_clustering.fit_predict(word_vectors)



#按照k-means k的取值进行迭代并画图
from scipy.spatial.distance import cdist, pdist

K = range(1,50)
KM = [KMeans(n_clusters=k).fit(word_vectors) for k in K]
centroids = [k.cluster_centers_ for k in KM]

D_k = [cdist(word_vectors, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
avgWithinSS = [sum(d)/word_vectors.shape[0] for d in dist]

# Total with-in sum of square
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(word_vectors)**2)/word_vectors.shape[0]
bss = tss-wcss

kIdx = 10-1

# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, avgWithinSS, 'b*-')
ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, 
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




#so, we choose k=10


num_clusters = 5
for clsNum in range(2,20):
    



wordDict = dict(zip(model.wv.index2word, idx))

#return dictionary key by its value
#将dict变为list
wordDict_list = list(wordDict)
words=[]
for cluster in range(0,5):
    print ("\nCluster %d" % cluster)
    for word, clst in wordDict.items():
      if clst == cluster:
        words.append(word)
    print(words)


lenth = len(wordDict_list)



#return words in each cluster
words = [[] for i in range(5)]
for cluster in range(0,5):
    #print ("\nCluster %d" % cluster)
    key = next(key for key, value in wordDict.items() if value == cluster)
    words[cluster].append(key)#在最内部的list中没有循环起来
    print(words[cluster])


#每个custer的结果一样
words = [[] for i in range(5)]

for cluster in range(0,5):
    # Print the cluster number  
    # print ("\nCluster %d" % cluster)
    #
    # Find all of the words for that cluster number, and print them out
 
    for i in range(0,lenth):
       word = list(wordDict .keys())[list(wordDict .values()).index(cluster)]
       words[cluster].append(word)
print (words)
    
#进行PCA降维
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# load the word2vec model
rawWordVec=vector

# reduce the dimension of word vector
X_reduced = PCA(n_components=2).fit_transform(rawWordVec) #从200维降至二维，方便可视化


#可视化
# show some word(center word) and it's similar words
from gensim import similarities
index = similarities.MatrixSimilarity('mymodel')

from gensim import models
lsi = models.LsiModel(noStop, id2word=dictionary, num_topics=2)
index = similarities.MatrixSimilarity(lsi[noStop]) 







#层次聚类
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

l = linkage(model.wv.syn0, method='complete', metric='seuclidean')

# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.ylabel('word')
plt.xlabel('distance')

dendrogram(
    l,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=16.,  # font size for the x axis labels
    orientation='left',
    leaf_label_func=lambda v: str(model.wv.index2word[v])
)
plt.show()






index1,metrics1 = model.cosine(u'物流')
index2,metrics2 = model.cosine(u'舒服')
index3,metrics3 = model.cosine(u'好看')
index4,metrics4 = model.cosine(u'价格')
index5,metrics5 = model.cosine(u'尺码')

# add the index of center word 
index01=np.where(model.wv.vocab==u'物流')
index02=np.where(model.wv.vocab==u'舒服')
index03=np.where(model.wv.vocab==u'好看')
index04=np.where(model.wv.vocab==u'价格')
index05=np.where(model.wv.vocab==u'尺码')

index1=np.append(index1,index01)
index2=np.append(index2,index03)
index3=np.append(index3,index03)
index4=np.append(index4,index04)
index5=np.append(index5,index05)

# plot the result
zhfont = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
fig = plt.figure()
ax = fig.add_subplot(111)

for i in index1:
    ax.text(X_reduced[i][0],X_reduced[i][1], model.wv.vocab[i], fontproperties=zhfont,color='r')

for i in index2:
    ax.text(X_reduced[i][0],X_reduced[i][1],model.wv.vocab[i], fontproperties=zhfont,color='b')

for i in index3:
    ax.text(X_reduced[i][0],X_reduced[i][1], model.wv.vocab[i], fontproperties=zhfont,color='g')

for i in index4:
    ax.text(X_reduced[i][0],X_reduced[i][1],model.wv.vocab[i], fontproperties=zhfont,color='k')

for i in index5:
    ax.text(X_reduced[i][0],X_reduced[i][1],model.wv.vocab[i], fontproperties=zhfont,color='c')

ax.axis([0,0.8,-0.5,0.5])
plt.show()
