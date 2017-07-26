import requests
import re

urls=[]
for i in list(range(1,99)):
    urls.append('https://rate.tmall.com/list_detail_rate.htm?itemId=547528421887&spuId=837657021&sellerId=2935656061&order=3&currentPage='+ str(i)+'&append=0&content=1&tagId=&posi=&picture=&ua=044UW5TcyMNYQwiAiwQRHhBfEF8QXtHcklnMWc%3D%7CUm5Ockt%2FRXtEeUd4R31FfCo%3D%7CU2xMHDJ7G2AHYg8hAS8XIgwsAl4%2FWTVSLFZ4Lng%3D%7CVGhXd1llXGhSbFNuUG9QalJrXGFDe0B6RXFMdkJ6RH9HfkdpPw%3D%3D%7CVWldfS0TMw82DCwQKgokFDAUKVhmCz5Ldkp3T3FUfUVrPWs%3D%7CVmhIGCQfPwQ7AiIeIhYrCzELPgEhHSkWKws3Cj8CIh4qFSgINAkwDVsN%7CV25OHjBKK0cqC3ERdg8uTylHIA4uEi4aOgY%2FBj0dJxohFUMV%7CWGFBET8RMQo0DCwQLRItDTIPMA47bTs%3D%7CWWBAED4QMAg3AiIeJxgjAzoHOAE%2FaT8%3D%7CWmNDEz19KXYfdT1ULwEhGCIePgI8ATsbIh8qFS17LQ%3D%3D%7CW2NDEz19KXYfdT1ULwEhcUhySWlVa1ZvORkmBigGJh8iFyMeSB4%3D%7CXGVFFTsVNQwwCysXLxYtDTYLNA0xZzE%3D%7CXWREFDoUNAA7GyAfJho6ADQINwJUAg%3D%3D%7CXmdHFzkXNw8wDS0VIR4lBTkFMQ0xC10L%7CX2dHFzkXN2dYZll5QHhDdyEBPBwyHDwAPwY%2BAj5oPg%3D%3D%7CQHlZCSdMK09uA3IPdB0zEy8SKx4%2BBz4KPx8jHCcYJhNFEw%3D%3D%7CQXhYCCYIKB0mGzsCPQQ4GCQbIBsjHEoc%7CQntbCyVOKU1sAXANdh8xES0QLBExCDYNMhIuESsULBJEEg%3D%3D%7CQ3tbCyULK3tPekBgWWFaYzUVKAgmCCgUKx4rFix6LA%3D%3D%7CRHxcDCIMLHxEek5uUGVdYTcXKgokCioWLxsmHyVzJQ%3D%3D%7CRX1dDSMNLX1HfURkXWVQaUlxSXRUblNnRXlMdk5uUGpKdEAWNgsrBSsLNwwzDDYOWA4%3D%7CRn9Cf19iQn1dYVhkRHpCeFhhQX1AYFR0QWFbe0d%2FX2dHfl5gQHxAYFx8QWFAfFxmRn1dZTM%3D&isg=AkREM6gEwQiRuHVFTV_3FBm_FcI8rSZOvvFJC17kg4_rieZThm04V3qzvxeq&needFold=0&_ksTS=1497320485156_1607&callback=jsonp1608')

# 构建字段容器
nickname = []

ratedate = []

color = []

size = []

ratecontent = []


# 循环抓取数据

for url in urls:

    content = requests.get(url).text

    nickname.extend(re.findall('"displayUserNick":"(.*?)"',content))

    color.extend(re.findall(re.compile('颜色分类:(.*?);'),content))

    size.extend(re.findall(re.compile('尺码:(.*?);'),content))

    ratecontent.extend(re.findall(re.compile('"rateContent":"(.*?)","rateDate"'),content))

    ratedate.extend(re.findall(re.compile('"rateDate":"(.*?)","reply"'),content))

    print(nickname,color)

# 写入数据

file =open('2017夏季新款男士帆布鞋运动休闲男鞋子韩版潮流平板鞋男百搭学生.csv','w')

for j in list(range(0,len(nickname))):

    file.write(','.join((nickname[j],ratedate[j],color[j],size[j],ratecontent[j]))+'\n')

file.close()
    
