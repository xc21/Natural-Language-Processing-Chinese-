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
f = open("C://Users//caoxun//Desktop//淘宝评论project//segResult3.txt", "w")
f.write(segResult)
f.close()

f = open("C://Users//caoxun//Desktop//淘宝评论project//noStop3.txt", "w")
f.write(noStop)
f.close()



#使用word2vector
from gensim.models import word2vec
import logging
from gensim import corpora

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus(r"C:\Users\caoxun\Desktop\淘宝评论project\noStop3.txt")
#sentences = LoadCorpora('r"C:\Users\caoxun\Desktop\淘宝评论project\segResult3")
# 加载语料
model = word2vec.Word2Vec(sentences,min_count=2, size=200) #200维的词向量
model.most_similar(u"鞋子", topn=20)
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



#进行PCA降维
#import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# load the word2vec model
rawWordVec=vector

# reduce the dimension of word vector
X_reduced = PCA(n_components=3).fit_transform(rawWordVec)


from gensim.models.keyedvectors import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format(word_vectors,binary=False)
