# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 10:17:37 2017

@author: caoxun
"""

#用规则去写
import codecs
import jieba
import re  


#初始化    
product_new =""

#读取评论，合并行
#for line in shoes.readlines():
 #   shoes_new += "".join(line.split('\n'))

with codecs.open('C:\\Users\\caoxun\\Desktop\\淘宝评论project\\衣服鞋子的商品库\\43783779796_test.csv', "r") as f:
    proruct_new = " ".join(line.strip() for line in f)  
proruct_new

#把默认为标点的空格explict替换成标点，这里先替换成逗号
#把所有的标点都替换成逗号，然后在下一步去重
proruct_new = re.sub('[*｀Ω?*v ？！_~ …→....，]',',', proruct_new)
    
#遍历，输入评论中重复的标点 （空格可以之后再.strip）
def get_solo(text):
    duels=[x+x for x in list('。，,！！?…～：')]
    for d in duels:
        while d in text:
            text=text.replace(d,d[0])
    return text
 
if __name__=='__main__':
    text=proruct_new
    proruct_new=get_solo(text)
   
    

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

#去除标点前的多余空格，避免词库报错
noStop = re.sub(r'\s{2,}','',noStop)
noStop = re.sub(r' 。','。',noStop)
noStop = re.sub(r' ，','，',noStop)
noStop = re.sub(r' ！','！',noStop)
noStop = re.sub(r' !','!',noStop)
noStop = re.sub(r' ？','？',noStop)




#输出分词结果为txt
import codecs

f = codecs.open("C://Users//caoxun//Desktop//淘宝评论project//segResult7.txt", "w",'utf-8')
f.write(segResult)
f.close()


f =codecs.open("C://Users//caoxun//Desktop//淘宝评论project//noStop7.txt", "w",'utf-8') 
f.write(noStop)
f.close()


#将去完停词的评论按照标点划分小短句
import re 
noStop2 =  re.split(r'[。！,，]+', noStop)
print(noStop2)

#分别读取正向词，负向词和否定词
pos=[]
negative=[]
with open('C://Users//caoxun//Desktop//淘宝评论project//衣服鞋子的商品库//正向情感词.txt', 'r', encoding='utf-8') as file:
    pos=file.read().replace('\n', '')
pos=pos.replace(u'\ufeff', '').split()

with open('C://Users//caoxun//Desktop//淘宝评论project//衣服鞋子的商品库//负向情感词.txt', 'r', encoding='utf-8') as file:
    negative=file.read().replace('\n', '')
negative=negative.replace(u'\ufeff', '').split()

neg=[]
with open('C://Users//caoxun//Desktop//淘宝评论project//衣服鞋子的商品库//否定词.txt', 'r', encoding='utf-8') as file:
    neg=file.read().replace('\n', '').split()



#找出正评价中最常出现的
cmtKeepPos = []       
for i in range(len(pos)):
    for k in range(len(noStop2)): 
        if pos[i] in noStop2[k].split():
                cmtKeepPos.append(noStop2[k])
cmtKeepPos   

from collections import Counter 
Counter(cmtKeepPos).most_common(50)

#找出负评价中最常出现的

cmtKeepNeg = []       
for i in range(len(negative)):
    for k in range(len(noStop2)): 
        if negative[i] in noStop2[k].split():
                cmtKeepNeg.append(noStop2[k])
cmtKeepNeg   
Counter(cmtKeepNeg).most_common(20)

#找出评论中带否定词的
cmtKeepN = []       
for i in range(len(neg)):
    for k in range(len(noStop2)): 
        if neg[i] in noStop2[k]:
                cmtKeepN.append(noStop2[k])
cmtKeepN   

#检查否定词是否逆转负向词变成正向的
test=[]
for i in range(len(negative)):
    for k in range(len(cmtKeepN)): 
        if negative[i] in cmtKeepN [k]:
           test.append(cmtKeepN [k])
#list内去重
test=list(set(test))
#合并两个list
cmtKeepPos.extend(test)
# 合并了转义词后的most common
Counter(cmtKeepPos).most_common(50)



#检查否定词是否逆转负向词变成正向的
test2=[]
for i in range(len(pos)):
    for k in range(len(cmtKeepN)): 
        if pos[i] in cmtKeepN [k]:
           test2.append(cmtKeepN [k])
test2=list(set(test2))
cmtKeepNeg.extend(test2)
# 合并了转义词后的most common
Counter(cmtKeepNeg).most_common(20)
