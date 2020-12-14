import re
str = 'dog runs to cat'
print('dog' in str)
print('bird' in str)
part1='dog'
part2='bird'
print(re.search(part1,str))
print(re.search(part2,str))
part=r'r[au]n'
print(re.search(part,str))
print(re.search(r'r[A-Z]n',str))
print(re.search(r'r[a-z]n',str))
print(re.search(r'r[0-9]n',str))
print(re.search(r'r\Dn',str))
string='''
dog runs to cat.
I run to dog.'''
print(re.search(r'^I',string,flags=re.M))
# ----------------
from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
html=urlopen('http://www.jueshitangmen.info/tian-meng-bing-can-11.html').read().decode('utf-8')
print(html)
soup=bs(html,features='lxml')
all_p=soup.find_all('p')
print(all_p)
for i in all_p:
    print('\n',i.get_text())
# ----------------
from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
html=urlopen('http://www.weather.com.cn/weather/101270101.shtml').read().decode('utf-8')
#print(html)
soup=bs(html,features='lxml')
all_ul=soup.find_all('ul',attrs={'class':'t clearfix'})
all_li=all_ul[0].find_all('li')
#print(all_li)
for i in all_li:
    #print(i)
    h1=i.find('h1').get_text()
    p1=i.find('p',attrs={'class':'wea'}).get_text()
    p2 = i.find('p', attrs={'class': 'tem'})
    tem = p2.find('span').get_text()+'~'+p2.find('i').get_text()
    win=i.find('p',attrs={'class':'win'}).find('i').get_text()
    print(h1)
    print(p1)
    print(tem)
    print(win)
# ----------------
import urllib.request
import urllib.parse
import re
import os
#添加header，其中Referer是必须的,否则会返回403错误，User-Agent是必须的，这样才可以伪装成浏览器进行访问
header=\
{
     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
     "referer":"https://image.baidu.com"
    }
url = "https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord={word}&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=0&word={word}&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&cg=girl&pn={pageNum}&rn=30&gsm=1e00000000001e&1490169411926="
keyword = input("请输入搜索关键字：")
#转码
keyword = urllib.parse.quote(keyword,"utf-8")

n = 0
j = 0

while(n<3000):
    error = 0
    n+=30
    url1 = url.format(word = keyword,pageNum=str(n))
    #获取请求
    rep = urllib.request.Request(url1,headers=header)
    #打开网页
    rep = urllib.request.urlopen(rep)
    #获取网页内容
    try:
        html = rep.read().decode("utf-8")
        # print(html)
    except:
        print("出错了！")
        error = 1
        print("出错页数："+str(n))
    if error == 1:
        continue
    #正则匹配
    p = re.compile(r"thumbURL.*?\.jpg")
    #获取正则匹配到的结果，返回list
    s = p.findall(html)

    if os.path.isdir(r"E:\py\training\pic") != True:
        os.makedirs(r"E:\py\training\pic")
    with open("testpic.txt","a") as f:
        #获取图片
        for i in s:
            i = i.replace('thumbURL":"','')
            print(i)
            f.write(i)
            f.write("\n")
            #保存图片
            urllib.request.urlretrieve(i,r"E:\py\training\pic\{num}.jpg".format(num=j))
            j+=1
        f.close()
print("总共爬取图片数为："+str(j))
# ----------------
from flask import  Flask,request,render_template,redirect,url_for,send_from_directory
from werkzeug.utils import secure_filename
import os
app=Flask(__name__)
@app.route("/upload",methods=["POST","GET"])
def upload():
    if request.method=='POST':
        f=request.files['file']
        basepath=os.path.dirname(__file__)
        upload_path=os.path.join(basepath,'static/upload/',secure_filename(f.filename))
        f.save(upload_path)
        return redirect(url_for('upload'))
    return render_template('upload.html')
if __name__=='__main__':
    app.run(port=6699,debug=True)

