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
segList = segResult.split()
stopwordsSet = set(stopwords)
noStop_list=[seg for seg in segList if seg not in stopwordsSet]
noStop= " ".join(noStop_list)

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
noStop2 =  re.split(r'[～，;！。?？、...]', noStop)
#shoes_new_str = ''.join(shoes_new)


#按照短句计算短句的平均值特征向量

#初始化feature vector
featureVec = np.zeros((200,),dtype="float32")
commentFeatureVecs = np.zeros((len(noStop2),200),dtype="float32")


for i in range(len(noStop2)):
    #这个小短句中的词组数 = 空格数-1
    wordNum = (noStop2[i].count(' ')) -1
    for word in noStop2[i].split():
        if word in words:
            featureVec = np.add(featureVec,model[word])
            if wordNum != 0:
                featureVec = np.divide(featureVec,wordNum)
