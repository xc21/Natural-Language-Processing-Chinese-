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

with codecs.open('C:\\Users\\caoxun\\Desktop\\淘宝评论project\\衣服鞋子的商品库\\545285029491_test.csv', "r") as f:
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
noStop = re.sub(r', ',',',noStop)
noStop


#输出分词结果为txt
import codecs

f = codecs.open("C://Users//caoxun//Desktop//淘宝评论project//segResult9.txt", "w",'utf-8')
f.write(segResult)
f.close()


f =codecs.open("C://Users//caoxun//Desktop//淘宝评论project//noStop9.txt", "w",'utf-8') 
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
      noStopKeep.append(noStop[k].strip())
    a=0
#list to string
noStopKeep="".join(noStopKeep)
noStopKeep=noStopKeep.strip()
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

#找出评论中带否定词的
cmtKeepN = []       
cmtKeepNN = []   
e=0
for k in range(len(noStopKeep)): 
    for i in range(len(neg)):
        if neg[i] in noStopKeep[k].strip():
                cmtKeepN.append(noStopKeep[k].strip())#去掉首位空格
        else: 
            e=1
    if e==1:       
            cmtKeepNN.append(noStopKeep[k].strip())
    e=0       
cmtKeepN 
cmtKeepNN  # NN= no negation

#检查否定词是否逆转负向词变成正向的
test=[]
d=0
for k in range(len(cmtKeepN)): 
    for i in range(len(negative)):
        if negative[i] in cmtKeepN [k]:
              d=1
    if d==1:
           test.append(cmtKeepN[k])
    d=0
test
#list内去重
test=list(set(test))
#合并两个list
cmtKeepPos=[]
cmtKeepPos.extend(test)
#去掉element里前后的空格
cmtKeepPos=[x.strip() for x in cmtKeepPos]
cmtKeepPos

#检查否定词是否逆转正向词变成负向的
test2=[]
for k in range(len(cmtKeepN)):
    for i in range(len(pos)):
        #发现很多有正向词的短句在有转义的cmtKeepN list里是因为短句中包含‘不错’，而
        #本身正向词并未被转义，故筛选时剔除‘不错’,'大小 合适' 中的‘大小同理’
        if pos[i] in cmtKeepN [k] and '不错' not in cmtKeepN [k] :
           test2.append(cmtKeepN [k])
#list内去重
test2=list(set(test2))
#合并两个list
cmtKeepNeg=[]
cmtKeepNeg.extend(test2)
#去掉element里前后的空格
cmtKeepNeg=[x.strip() for x in cmtKeepNeg]
cmtKeepNeg









#在没有否定词转义的评论里找出评论中的正面评论
b=0
for k in range(len(cmtKeepNN )):
    for i in range(len(pos)):
        if pos[i] in cmtKeepNN [k]:
            b=1
    if b==1:
        cmtKeepPos.append(cmtKeepNN [k])
    b=0    
cmtKeepPos   


#在没有否定词转义的评论里找出评论中的负面评论
c=0
for k in range(len(cmtKeepNN)):
    for i in range(len(negative)):
        if negative[i] in cmtKeepNN[k] and '不错' not in cmtKeepNN[k] and '合适' not in cmtKeepNN[k] and '正好' not in cmtKeepNN[k]:
            c=1
    if c==1:
        cmtKeepNeg.append(cmtKeepNN[k])
    c=0    
cmtKeepNeg


#频次统计
from collections import Counter 
Counter(cmtKeepPos).most_common(20)
#[('穿着 舒服', 308),
# ('喜欢', 268),
# ('不错', 247),
# ('舒服', 188),
# ('好评', 178),
# ('满意', 162),
# ('面料 舒服', 147),
# ('质量 不错', 133),
# ('穿上 舒服', 94),
# ('料子 舒服', 76),
# ('好看', 62),
# ('值得 购买', 56),
# ('大小 合适', 56),
# ('物美价廉', 51),
# ('物流 快', 47),
# ('赞', 47),
# ('衣服 不错', 47),
# ('穿起来 舒服', 46),
# ('漂亮', 45),
# ('合身', 45)]

Counter(cmtKeepNeg).most_common(25)





#实际淘宝的标签：
#很舒适 质量很好 布料好 衣服合身 便宜 快递不错 穿着不错 跟描述一致 尺寸有偏差 不舒适
