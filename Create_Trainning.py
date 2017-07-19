# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 13:48:44 2017

@author: caoxun
"""

#merge multiple csv to become the training set

a = open(r"C:\Users\caoxun\Desktop\淘宝评论project\衣服鞋子的商品库\38864927013.csv",encoding="gbk").readlines()
b = open(r"C:\Users\caoxun\Desktop\淘宝评论project\衣服鞋子的商品库\545315841608.csv",encoding="gbk").readlines()
c = open(r"C:\Users\caoxun\Desktop\淘宝评论project\衣服鞋子的商品库\547599346803.csv",encoding="gbk").readlines()
d = open(r"C:\Users\caoxun\Desktop\淘宝评论project\衣服鞋子的商品库\548111754857.csv",encoding="gbk").readlines()
e = open(r"C:\Users\caoxun\Desktop\淘宝评论project\衣服鞋子的商品库\cloth.csv",encoding="gbk").readlines()
f  = open(r"C:\Users\caoxun\Desktop\淘宝评论project\衣服鞋子的商品库\531965497915.csv",encoding="gb18030").readlines()


result = a+b+c+d+e+f
with open(r'C:\Users\caoxun\Desktop\淘宝评论project\衣服鞋子的商品库\output.csv', 'w',encoding="gb18030") as fp:
    fp.write(''.join(result))
    
result.to_csv(r'C:\Users\caoxun\Desktop\淘宝评论project\衣服鞋子的商品库\merged.csv')


