# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:42:44 2017

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
#打印分词结果
print ('/'.join(segList))
#分词结果储存
segResult = " ".join(segList)


#将停用词txt读取成为string变量
with open('C://Users//caoxun//Desktop//淘宝评论project//Chinese Stop Words.txt', 'r') as myfile:
    stopwords=myfile.read().replace('\n', '')
      
      
#删除停用词
segList = segResult.split()
stopwordsSet = set(stopwords)
noStop=[seg for seg in segList if seg not in stopwordsSet]
noStop= " ".join(noStop)


#输出分词结果为txt
f = open("C://Users//caoxun//Desktop//淘宝评论project//segResult3.txt", "w")
f.write(segResult)
f.close()

f = open("C://Users//caoxun//Desktop//淘宝评论project//noStop3.txt", "w")
f.write(noStop)
f.close()

#同义词替换
file = codecs.open('C://Users//caoxun//Desktop//淘宝评论project//Synonyms.txt', 'r', encoding="utf8")
#讲同义词替换为list of lists
syn = []
with open('C://Users//caoxun//Desktop//淘宝评论project//Synonyms.txt', 'r', encoding="utf8") as file:
     lines = [line for line in file.read().split('\n')]
for line in lines:
    syn.append(line.split())

#试着用循环去寻找和替换
#将noStop变为list
noStop_list = noStop.split()
for i in range(len(noStop_list)):
    for j in range(len(syn)):
        if noStop_list[i] in syn[j][1:]:
            noStop_list[i] = syn[j][0]            
print(noStop_list)

#将noStop转回string,写成txt,测试
noStop_string = " ".join(noStop_list)
f = open("C://Users//caoxun//Desktop//淘宝评论project//synSub.txt", "w")
f.write( noStop_string )
f.close()

#找到noStop_string中出现频率最高的数个词
words = re.findall(r'\w+', noStop_string) 
from collections import Counter 
print( Counter(words).most_common(10))  


