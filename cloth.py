# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 17:49:36 2017

@author: caoxun
"""

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
product = codecs.open('C:\\Users\\caoxun\\Desktop\\淘宝评论project\\Comment_Test.txt', mode='r', errors='strict', buffering=1)

#初始化    
product_new =""

#合并行
#for line in shoes.readlines():
 #   shoes_new += "".join(line.split('\n'))

import codecs    
with codecs.open('C:\\Users\\caoxun\\Desktop\\淘宝评论project\\Comment_Test.txt', "r",'utf-8') as f:
    proruct_new = " ".join(line.strip() for line in f)  
    
#遍历，输入评论中重复的标点 （空格可以之后再.strip）
def get_solo(text):
    duels=[x+x for x in list('。，！?')]
    #如需增加标点符号,比如问号,直接将list('。，!')换成list('。，!？')即可.
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


#按照表示分句的标点将string分割成list，这样list上的每一位就为一个短句了
#import re
#shoes_new =  re.split(r'[～，;！。?？、...]', shoes_new)
#shoes_new_str = ''.join(shoes_new)

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
#去除多余空格，避免词库报错
noStop = re.sub('     ',' ',noStop)

#输出分词结果为txt
import codecs

f = codecs.open("C://Users//caoxun//Desktop//淘宝评论project//segResult6.txt", "w",'utf-8')
f.write(segResult)
f.close()


f =codecs.open("C://Users//caoxun//Desktop//淘宝评论project//noStop6.txt", "w",'utf-8') 
f.write(noStop)
f.close()





#使用word2vector
from gensim.models import word2vec
import logging
#from gensim import corpora

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus(r"C:\Users\caoxun\Desktop\淘宝评论project\noStop6.txt",)#需要读取成utf-8，先在前面文件写入时设置好
#sentences = LoadCorpora('r"C:\Users\caoxun\Desktop\淘宝评论project\segResult3")
# 加载语料
model = word2vec.Word2Vec(sentences,min_count=2, size=200) #200维的词向量
model.most_similar(u"偏大", topn=5)
#[('太大', 0.9695711135864258),
# ('小', 0.969524085521698),
# ('这么', 0.9694556593894958),
# ('一样', 0.9694411754608154),
# ('到', 0.9694200754165649)]

#结论： “偏大一组”的找相似效果比较好，但其他词语效果欠佳

model.most_similar(u"物流", topn=5)
#[('，', 0.9996504187583923),
# ('好', 0.9996373653411865),
# ('质量', 0.9996233582496643),
# ('了', 0.999622106552124),
# ('的', 0.9996165633201599)]

model.most_similar(u"快递", topn=5)
#[('，', 0.9994956254959106),
# ('很', 0.999434769153595),
# ('衣服', 0.9994258880615234),
# ('的', 0.9994170665740967),
# ('好', 0.9994134902954102)]

model.most_similar(u"质量", topn=5)
#[('，', 0.9998703598976135),
# ('的', 0.9998605251312256),
# ('好', 0.999846339225769),
# ('衣服', 0.9998286962509155),
# ('很', 0.9998286962509155)]


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
noStop2 =  re.split(r'[。！，]+', noStop)


#按照短句计算短句的平均值特征向量

#初始化feature vector （小短句为单位）
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
num_clusters = 5
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
for i in range(0, 3):
    freqcount = Counter(test[i]).most_common(20)
    print( "Cluster" , i, freqcount)  


#提高频的词，去掉次数
freqwords = []
for j in range(0,19):              
       freqwords.append(freqcount[j][0])
    
    
    
 #   但是爬虫结果有重复， 
#Cluster 0 [(' 大小 合适 ', 5), (' 衣服 也 不 咋 地 ', 2), (' 就是 买大 了 点 ', 2), ('   挺 好 的 ', 2), ('   质量 不错 ', 2), (' 一切 都 满意 ', 2), (' 看着 也 不错 ', 2), (' 卖家 给我发 的 是 桔 蓝色 的 ', 2), (' 质量 好   衣服 收到 了 ', 2), (' 给 弟弟 买的 ', 2), (' 还 以为 很 薄 ', 2), (' 他 非常 喜欢 ', 2), ('   衣服 挺不错 的 ', 2), (' 质量 蛮 好   挺 好 的 ', 2), (' 特别 暖和 ', 2), (' 以后 买 衣服 找 这家 了 昂 ', 2), (' 质量 跟 六七十 的 一样 ', 2), (' 我弟 穿着 有点 大 ', 2), (' 是 帮 老弟 买的 ', 2), (' 可以 买 ', 2)]
#Cluster 1 [(' 失望 失望 ', 2), (' 服务态度 也好 ', 2), (' 跟 图片 一样 ', 2), (' 值这个价 钱 ', 2), (' 原本 想放 着 ', 2), (' 谁 知道 下雨 ', 2), (' 正好 用 ', 2), ('   对得起 这个 价钱 ', 2), (' 尺码 准 颜色 正 棒棒哒 ', 2), (' 换货 麻烦 ', 2), (' 有需要 还会 光顾 哦   好好 很 暖和   衣服 还有 看   感觉 还 可以   这 也 太大 啦   还 行 ', 2), (' 但是 感觉 布料 摸 着 怪怪的   不错   衣服 开线 了 ', 2), (' 款式 简单 大气 ', 2), ('   还好 吧       ', 2), (' 不知道 以后 会 不会 起球   这个 不错 ', 2), (' 但是 要个 L 为什么 那么 大 ？   有些 大 了 … … 摸 起来 感觉 怪怪的   还好 ', 2), (' 气味 重了 ', 2), (' 差不多 就 这价 吧   先 收货   漂亮 暖和 ', 2), (' 好评   看起来 不错 五星 好评 ', 2), (' 样子 如图 ', 2)]
#Cluster 2 [(' 质量 很 好 ', 6), (' 好评 ', 6), (' 质量 也 很 好 ', 4), (' 非常 好 ', 3), (' 质量 好 ', 2), (' 质量 还 可以 ', 2), (' 不错 ', 2), (' 喜欢 ', 2), (' 很 喜欢 ', 2), (' 质量 不错 ', 2), (' 值 了 ', 2), (' 挺厚 的 ', 2), (' 很 合适 ', 2), (' 衣服 也 很 合身 ', 2), (' 非常 非常 的 好 ', 2), (' 好 哦 ', 2), (' 质量 很 不错 ', 2), (' 质量 也 不错 ', 2), (' 物有所值 ', 1)]    
    
    
    
    
    
    
    
    
                    
