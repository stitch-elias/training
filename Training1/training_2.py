#1
i=1
n=1
while i<10:
    n=(n+1)*2
    i+=1
print(n)
#2
i=2
j=1
s=0
for k in range(20):
    s=s+(i/j)
    l=i
    i=(i+j)
    j=l
print(s)
#3
s=0
for i in range(1,21):
    n=1
    for j in range(1,i+1):
        n*=j
    s+=n
print(s)
#4
list1=["a","b","a","d","c","b"]
list2=list1
for i in list1:
    k=0
    for j in list2:
        if i==j:
            k+=1
            if k>1:
                list2.remove(j)
list1=list2
print(list1)
#------------------------------
list1=list(set(list1))
print(list1)
#5
str="string"
s=str[::-1]
print(s)
#-----------------------------
s=list(str)
s.reverse()
print("".join(s))
#6
list1=[1,2,3,4,5,6,7,8,9]
print(sum(map(lambda x:x+3,list1[2::])))
#7
List=[-2,1,3,-6]
List.sort(key=abs)
print(List)
#8
list1=[1,2,3]
list2=[4,5,6]
list3=list1+list2
print(list3)
#9
'lambda是python的匿名函数，可以用来定义一个简单的表达式；可以使代码更简洁；提供了map（）、sort（）等函数，提供了装饰器，闭包等语法'
#10
'list:[元素]可以存放多个元素的有序可变的容器，可以通过索引访问元素'
'tuple:(元素)可以存放多个元素的有序不可变的容器，访问效率高'
'dict：{键：值}以键值对的方式存储数据的可变的容器，可以通过键访问对应的值'
#11
alist=[{'name':'a','age':20},{'name':'b','age':30},{'name':'c','age':25}]
alist.sort(key=lambda x:x['age'],reverse=True)
print(alist)
#12
a1="k:1|k1:2|k2:3|k3:4"
a2=a1.split(sep='|')
d={}
for i in a2:
    a3=i.split(sep=':')
    d[a3[0]]=a3[1]
print(d)
#13
a=[0,1]
n=int(input("项数："))
for i in range(0,n-2):
    a.append(a[i]+a[i+1])
print(a)
#14
class student:
    def __init__(self,name,age,score):
        self.name = name
        self.age = age
        self.score = score
    def get_name(self):
        return self.name
    def get_age(self):
        return  self.age
    def get_courese(self):
        return max(self.score)
zm=student('zhangming',20,[69,88,100])
print(zm.get_name())
print(zm.get_age())
print(zm.get_courese())
#15
class dictclass:
    def __init__(self,d):
        self.d=d
    def del_dict(self,key):
        self.d.pop(key)
    def get_dict(self,key):
        a=self.d.keys()
        for i in a:
            if i==key:
                return key
        else:
            return "not found"
    def get_key(self):
        return self.d.keys()
    def update_dict(self,new_d):
        self.d.update(new_d)
        return self.d.values()
di=dictclass({1:1,2:2,3:3})
di.del_dict(3)
print(di.get_dict(2))
print(di.get_dict(3))
print(di.get_key())
print(di.update_dict({4:4,5:5}))
#16
class Listinfo:
    def __init__(self,l):
        self.l=l
    def add_key(self,keyname):
        self.l.append(keyname)
    def get_key(self,num):
        return num in self.l
    def update_list(self,list):
        self.l=self.l+list
        return self.l
    def del_key(self):
        a=self.l[len(self.l)-1]
        del self.l[len(self.l)-1]
        return a
list_info = Listinfo([44,222,111,333,454,'sss','333'])
print(list_info.l)
list_info.add_key(999)
print(list_info.get_key(999))
print(list_info.update_list([123,456]))
print(list_info.del_key())
print(list_info.l)
#17
class Setinfo:
    def __init__(self,s):
        self.s=s
    def add_setinfo(self,keyname):
        self.s.add(keyname)
    def get_intersection(self,unioninfo):
        return self.s.intersection(unioninfo)
    def get_union(self,unioninfo):
        return self.s.union(unioninfo)
    def del_difference(self,unioninfo):
        return self.s.difference(unioninfo)
set_info=Setinfo({1,3,2,4})
print(set_info.s)
set_info.add_setinfo(9)
print(set_info)
print(set_info.get_intersection({3}))
print(set_info.get_union({5}))
print(set_info.del_difference({2,4,6}))
#18
class student:
    name='MR.Lu'
    sexual='man'
    age=22
    address='Chengdu'
    def display(self):
        print('name:{0},sexual:{1},age:{2},address:{3}'.format(self.name,self.sexual,self.age,self.address))
s=student()
s.display()
#19
class peple:
    def eat(self):
        print('吃')
    def run(self):
        print('跑步')
class student(peple):
    def study(self):
        print('学习')
s=student()
s.eat()
s.run()
s.study()

