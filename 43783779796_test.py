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
proruct_new = re.sub(r'\s{1,}','，',proruct_new)
proruct_new 

#把所有的标点都替换成逗号，然后在下一步去重
proruct_new = re.sub('[*｀Ω?*v 。、？!！_~～ …→....，3_ヽュ]',',', proruct_new)
    
#遍历，输入评论中重复的标点 （空格可以之后再.strip）
def get_solo(text):
    duels=[x+x for x in list('。，,,！！?…～：')]
    for d in duels:
        while d in text:
            text=text.replace(d,d[0])
    return text
 
if __name__=='__main__':
    text=proruct_new
    proruct_new=get_solo(text)
   
proruct_new    

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
stopList=stopwords.split()
noStop=[]
for i in range(len(segList)):
    if segList[i] not in stopList:
        noStop.append(segList[i])
noStop= " ".join(noStop)

#删除程度副词
degree=[]
with open('C://Users//caoxun//Desktop//淘宝评论project//衣服鞋子的商品库//程度副词.txt', 'r', encoding='utf-8') as file:
    degree=file.read().replace('\n', '').split()
noStopKeep = []       
for i in range(len(degree)):
    for k in range(len( noStop)): 
        if degree[i] not in noStop[k]:
                noStopKeep.append(noStop[k])
#list to string
noStopKeep="".join(noStopKeep)
            
#重新去除一遍标点


#去除标点前的多余空格，避免词库报错
noStop = re.sub(r'\s{2,}','',noStopKeep)
noStop = re.sub(r' , , ',',',noStop)
noStop


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
#去掉element里前后的空格
cmtKeepPos=[x.strip() for x in cmtKeepPos]
# 合并了转义词后的most common
Counter(cmtKeepPos).most_common(25)
#[('不错', 8270),
# ('好评', 3200),
# ('行', 1984),
# ('满意', 1954),
# ('质量 不错', 1731),
# ('穿着 舒服', 1642),
# ('舒服', 1476),
# ('裤子 不错', 1379),
# ('喜欢', 1186),
# ('凉快', 963),
# ('赞', 962),
# ('合身', 897),
# ('透气', 832),
# ('物美价廉', 832),
# ('适合 夏天 穿', 768),
# ('东西 不错', 708),
# ('推荐 购买', 576),
# ('值得 信赖', 384),
# ('一如既往', 384),
# ('价格 实惠', 352),
# ('总体 不错', 321),
# ('夏天 穿 凉快', 320),
# ('值得 购买', 320),
# ('不错 不错', 290),
# ('轻', 288)]


#检查否定词是否逆转负向词变成正向的
test2=[]
for i in range(len(pos)):
    for k in range(len(cmtKeepN)): 
        if pos[i] in cmtKeepN [k]:
           test2.append(cmtKeepN [k])
test2=list(set(test2))
cmtKeepNeg.extend(test2)
cmtKeepNeg=[x.strip() for x in cmtKeepNeg]
# 合并了转义词后的most common
Counter(cmtKeepNeg).most_common(20)
#[('长', 837),
# ('线头 多', 803),
# ('裤腿 长', 422),
# ('线头', 416),
# ('裤子 长', 326),
# ('懒得 退', 288),
# ('裤腿 肥', 227),
# ('发货 慢', 226),
# ('裤脚 长', 224),
# ('无语', 192),
# ('做工 粗糙', 192),
# ('肥', 168),
# ('价格 贵', 160),
# ('裤裆 短', 160),
# ('裤子 长 点', 160),
# ('退货', 160),
# ('裤子 太 长', 155),
# ('物流 慢', 130),
# ('贵', 128),
# ('稍微 有点儿 长', 128)]
