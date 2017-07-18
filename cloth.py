# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 19:54:57 2017

@author: caoxun
"""

import codecs
import jieba
import re  


#初始化    
product_new =""

#读取评论，合并行
#for line in shoes.readlines():
 #   shoes_new += "".join(line.split('\n'))

with codecs.open('C:\\Users\\caoxun\\Desktop\\淘宝评论project\\cloth.csv', "r") as f:
    proruct_new = " ".join(line.strip() for line in f)  
    
#遍历，输入评论中重复的标点 （空格可以之后再.strip）
def get_solo(text):
    duels=[x+x for x in list('。，,！?')]
    for d in duels:
        while d in text:
            text=text.replace(d,d[0])
    return text
 
if __name__=='__main__':
    text=proruct_new
    proruct_new=get_solo(text)
   
    
#去除标点
#shoes_new = re.sub(',','',shoes_new)
#print(re.sub('～','',shoes_new))
#用[]一次性替换多个标点
#去掉特殊符号
product_new = proruct_new.replace(u'\ufeff', '')



#导入人工定义的字典,提高分词精准率
jieba.load_userdict('C://Users//caoxun//Desktop//淘宝评论project//newDict.txt')
#分词
segList = jieba.cut(product_new,cut_all=False) #=False > 精准模式

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
noStop=segResult
#去除标点前的多余空格，避免词库报错
noStop = re.sub(r'\s{2,}',' ',noStop)
noStop = re.sub(r' 。','。',noStop)
noStop = re.sub(r' ，','，',noStop)
noStop = re.sub(r' ！','！',noStop)
noStop = re.sub(r' !','!',noStop)



#输出分词结果为txt
import codecs

f = codecs.open("C://Users//caoxun//Desktop//淘宝评论project//segResult6.txt", "w",'utf-8')
f.write(segResult)
f.close()


f =codecs.open("C://Users//caoxun//Desktop//淘宝评论project//noStop6.txt", "w",'utf-8') 
f.write(noStop)
f.close()



#将去完停词的评论按照标点划分小短句
import re 
noStop2 =  re.split(r'[。！，]+', noStop)


#用list of lists读取function tags功能词表
functTags=[]
with open('C://Users//caoxun//Desktop//淘宝评论project//功能点库//funcTags.txt', 'r', encoding='utf-8') as file:
     lines = [line for line in file.read().split('\n')] #分行处理，一行一位
for line in lines:
    functTags.append(line.split())


#逐句遍历，小短句中是否有function tag，若有，则保留这个小短句
cmtKeep = [[] for i in range(10)]       
for i in range(len(functTags)):
    for j in range(len(functTags[i])):
        for k in range(len(noStop2)): 
           if functTags[i][j] in noStop2[k].split():
                cmtKeep[i].append(noStop2[k])
cmtKeep                

#下一步： 将cmtKeep上每一个功能词中对应的评论提出，分别聚类训练
cmt_quality =  ','.join(cmtKeep[0])
f =codecs.open("C://Users//caoxun//Desktop//淘宝评论project//cmt_quality.txt", "w",'utf-8') 
f.write(cmt_quality)
f.close()
cmt_shipping = ''.join(cmtKeep[1])
cmt_size = ''.join(cmtKeep[2])
cmt_style =''.join( cmtKeep[3])
cmt_price = ''.join( cmtKeep[4])
cmt_service = ''.join(cmtKeep[5])   
cmt_description = ''.join(cmtKeep[6])
cmt_cltGeneral = ''.join(cmtKeep[7])
cmt_buy = ''.join(cmtKeep[8])
cmt_pos = ''.join(cmtKeep[9])
   
####################################################测试“质量”标签下对应的评论##############################################             

#使用word2vector
from gensim.models import word2vec
import logging
#from gensim import corpora

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus(r"C:\Users\caoxun\Desktop\淘宝评论project\cmt_quality.txt",)#需要读取成utf-8，先在前面文件写入时设置好
#sentences = LoadCorpora('r"C:\Users\caoxun\Desktop\淘宝评论project\segResult3")
# 训练w2v
model = word2vec.Word2Vec(sentences,min_count=2, size=200) #200维的词向量

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

#标点处理 （可删除？）
import re
#noStop2 =re.sub("[\.\!\/_$%^*(+\"\'+——！、@#￥%& ? ？（）~]+", " 。",noStop)
#noStop2 = re.split(r"[～，;！。?？?、...？]+", noStop)
#noStop2 =  re.split(r'[～，;！。?？?、...？?。;?！ ]*', noStop2)
#shoes_new_str = ''.join(shoes_new)
#用正则替换清理标点
#noStop2 = re.sub(r"[～~，;！。。?？?、？！]+\1", " 。",noStop)

#处理文字，把多余的空格替换为逗号
cmt_quality=re.sub(r'\s{3,}', ', ', cmt_quality)
#去除一个位置上多余的标点
if __name__=='__main__':
    text=cmt_quality
    cmt_quality=get_solo(text)
   
cmt_quality=get_solo(cmt_quality)
print(cmt_quality)

#re.split(r'[～,?… …、]+', cmt_quality)
cmt_quality2 = cmt_quality.split(',')
print(cmt_quality2)


#按照短句计算短句的平均值特征向量

#初始化feature vector （小短句为单位）
import numpy as np
featureVec = np.zeros((200,),dtype="float32")
commentFeatureVecs = []


#遍历noStop2小短句中的每一句
for i in range(len(cmt_quality2 )):
    #这个小短句中的词组数 = 空格数-1
    wordNum = (cmt_quality2 [i].count(' ')) -1
    #遍历当前小短句里的每一个词
    for word in cmt_quality2 [i].split():
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
# elbow curve,根据图像估测曲线在k= 后趋于平稳

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
num_clusters = 3
# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans(n_clusters = num_clusters)
idx = kmeans_clustering.fit_predict(commentFeatureVecs) 

wordDict = dict(zip(idx,commentFeatureVecs))


#按照类目数目，打印出每一类中的词组
#初始化一个list of lists,然后按照cluster number依次传入
#将cmt_quality_list转变为list
cmt_quality_list=cmt_quality.split()
test = [[] for i in range(0,3)]
for cluster in range(0,3):
    for i in range(len(idx)):
        if idx[i]==cluster:
            #contentTemp.append()
            test[cluster].append(cmt_quality2[i])

#打印出每个cluster的内容
for i in range(0,3):
    print("Cluster" ,i, test[i]) #固定字符和变量的联合打印
            

from collections import Counter 
for i in range(0, 3):
    freqcount = Counter(test[i]).most_common(20)
    print( "Cluster" , i, freqcount)  


#提高频的词，去掉次数
freqwords = []
for j in range(0,19):              
       freqwords.append(freqcount[j][0])
    
    
    

    
    
                    
