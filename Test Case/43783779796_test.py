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


#去除标点前的多余空格，避免词库报错
noStop = re.sub(r'\s{2,}','',noStop)
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



#删除程度副词
degree=[]
with open('C://Users//caoxun//Desktop//淘宝评论project//衣服鞋子的商品库//程度副词.txt', 'r', encoding='utf-8') as file:
    degree=file.read().replace('\n', '').split()
    
noStopKeep = []
a = 0 #指示 degree词中是否有element出现过
   
for k in range(len(noStop)):
    for i in range(len(degree)):
        if degree[i] in noStop[k]: 
            a =1 
    if a==0: #如果在noStop的k位上遍历完都没有找到degree的element,则a依旧为0
      noStopKeep.append(noStop[k])
    a=0
#list to string
noStopKeep="".join(noStopKeep)
noStopKeep  

#将去完停词和程度词的评论按照标点划分小短句
import re 
noStopKeep =  re.split(r'[。！,，]+', noStopKeep)
print(noStopKeep)


          

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


#找出评论中的正面评论
cmtKeepPos = []       
b=0
for k in range(len(noStopKeep)):
    for i in range(len(pos)):
        if pos[i] in noStopKeep[k]:
            b=1
    if b==1:
        cmtKeepPos.append(noStopKeep[k])
    b=0    
cmtKeepPos   


#找出样本中的负面评论
cmtKeepNeg = []       
c=0
for k in range(len(noStopKeep)):
    for i in range(len(negative)):
        if negative[i] in noStopKeep[k] and '不错' not in noStopKeep[k] and '合适' not in noStopKeep[k]:
            c=1
    if c==1:
        cmtKeepNeg.append(noStopKeep[k])
    c=0    
cmtKeepNeg

#找出评论中带否定词的
cmtKeepN = []       
for k in range(len(noStopKeep)): 
    for i in range(len(neg)):
        if neg[i] in noStopKeep[k]:
                cmtKeepN.append(noStopKeep[k])
cmtKeepN   

#检查否定词是否逆转负向词变成正向的
test=[]
d=0
for i in range(len(negative)):
    for k in range(len(cmtKeepN)): 
        if negative[i] in cmtKeepN [k]:
              d=1
    if d==1:
           test.append(cmtKeepN [k])
    d=0
cmtKeepN
#list内去重
test=list(set(test))
#合并两个list
cmtKeepPos.extend(test)
#去掉element里前后的空格
cmtKeepPos=[x.strip() for x in cmtKeepPos]
# 合并了转义词后的most common
from collections import Counter 
Counter(cmtKeepPos).most_common(25)
f = codecs.open("C://Users//caoxun//Desktop//淘宝评论project//cmtPos.txt", "w",'utf-8')
f.write("".join(cmtKeepPos))
f.close()


#[('不错', 269),
# ('好评', 100),
# ('满意', 63),
# ('穿着 舒服', 61),
# ('质量 不错', 54),
# ('舒服', 50),
# ('轻薄', 44),
# ('裤子 不错', 43),
# ('喜欢', 39),
# ('凉快', 32),
# ('行', 31),
# ('合身', 29),
# ('还好', 29),
# ('透气', 26),
# ('物美价廉', 26),
# ('适合 夏天 穿', 24),
# ('东西 不错', 23),
# ('赞', 16),
# ('还行', 14),
# ('一如既往', 12),
# ('价格 实惠', 11),
# ('夏天 穿 凉快', 10),
# ('穿起来 舒服', 10),
# ('值得 购买', 10),
# ('总体 不错', 10)]


#检查否定词是否逆转负向词变成正向的

test2=[]
for k in range(len(cmtKeepN)):
    for i in range(len(pos)):
        #发现很多有正向词的短句在有转义的cmtKeepN list里是因为短句中包含‘不错’，而
        #本身正向词并未被转义，故筛选时剔除‘不错’,'大小 合适' 中的‘大小同理’
        if pos[i] in cmtKeepN [k] and '不错' not in cmtKeepN [k] and '合适' not in cmtKeepN [k] :
           test2.append(cmtKeepN [k])
#list内去重
test2=list(set(test2))
#合并两个list
cmtKeepNeg.extend(test)
#去掉element里前后的空格
cmtKeepNeg=[x.strip() for x in cmtKeepNeg]
# 合并了转义词后的most common
from collections import Counter 
Counter(cmtKeepNeg).most_common(25)
#[('长', 31),
# ('线头 多', 28),
# ('不知 耐 耐穿', 21),
# ('裤腿 长', 19),
# ('色差', 18),
# ('大小 合适', 18),
# ('裤子 长', 16),
# ('肥', 13),
# ('线头', 13),
# ('裤腿 肥', 10),
# ('发货 慢', 9),
# ('懒得 退', 9),
# ('大', 8),
# ('小', 7),
# ('买大', 7),
# ('裤脚 长', 7),
# ('做工 粗糙', 6),
#('物流 慢', 6),
# ('裤子  长', 6),
# ('裤子 长 点', 5),
# ('价格 贵', 5),
# ('质量 差', 5),
# ('退货', 5),
# ('裤裆 短', 5),
# ('长短 合适', 4)]





#实际淘宝的标签：
#质量不错， 穿着效果不错 尺码合适 便宜 舒适度挺好的 裤子很合适 薄厚度一般
