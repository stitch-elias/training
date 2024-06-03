import requests
from lxml import etree

n = 0  # name initialize of picture
for x in range(1,5000): #爬取页数

    url = f"https://wallhaven.cc/toplist?page={x}"

    headers1 = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.0.0 Safari/537.36'
        }
    response = requests.get(url=url,headers=headers1)
    #print(response.text)
    tree = etree.HTML(response.text)
    list = tree.xpath('//*[@id="thumbs"]/section[1]/ul/li/figure/a/@href')
    response.close()
    def repage(url1,n):
        response1 = requests.get(url = url1,headers=headers1)
        tree1 = etree.HTML(response1.text)
        src_url = tree1.xpath('//*[@id="wallpaper"]/@src')
        for j in src_url:
            response_img = requests.get(j)
        print(response_img)
        with open('img//'+'wallhaven//'+str(n)+'.jpg', mode="wb") as f:
            f.write(response_img.content) #二进制存入图片
        print("Done!")
        response_img.close()
        response1.close()

    for i in list:
        n=n+1
        repage(i,n) #本页图片获取