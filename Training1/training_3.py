#1
def display_message():
    print("函数")
display_message()
#2
def favorite_book(title):
    print('One of my favourite book is {}'.format(title))
favorite_book('Alice in Wonderland')
#3
def make_shirt(a,b):
    print('尺码：{0}，字样：{1}'.format(a,b))
make_shirt('S','ace')
make_shirt(b='cat',a='L')
#4
def make_shirt(a,b='I love Python'):
    print('尺码：{0}，字样：{1}'.format(a,b))
make_shirt('L')
make_shirt(a='M')
make_shirt('S','ace')
#5
def describe_city(name,country='China'):
    print('{0} is in {1}'.format(name,country))
describe_city('Chengdu')
describe_city('Dalian')
describe_city('Reykjavik','Iceland')
#6
def describe_city(name,country):
    return name+','+country
describe_city('Chengdu','China')
describe_city('Dalian','China')
describe_city('Reykjavik','Iceland')
#7
def make_album(singer,name,*count):
     if count!=():
         return {singer:{name:count}}
     else:
        return {singer:name}
print(make_album('a','b'))
print(make_album('c','d'))
print(make_album('e','f',10))
#8
#普通参数就是像下面这段函数传入的参数一样，传入函数，没有默认值
#指定参数按照顺序传入的话，如果没有指定参数的值那么就会按照顺序赋初始值
#默认参数在函数创建时，为没有填写的函数参数指定默认值
#动态参数 *args  是指当我们需要传入多个参数时，可以用*args代表多个参数，不用分别在括号里指定多个参数
#9
def t1(a):
    return len(a)>5
print(t1([1,2,3,4,5,6]))
#10
def t2(b):
    for i in b:
        if i.isspace():
            return True
    return False
print(t2([' ',2,4,6]))
#11
def t3(c):
    if len(c)>2:
        return c[:1]
print([1,2,3,4])
#12
def t4(d):
    o=[]
    for i in range(1,len(d),2):
        o.append(i)
    return o
print(t4([1,2,3,4,5,6,7]))
#13
def t5(e):
    for key in e:
        if len(e[key])>2:
            e[key]=e[key][:2]
        return e
print(t5({1:'asdsad',2:[1,2,3,4,5]}))
#14
def t6(f):
    s=0
    a=b=1
    for i in range(1,(f+1)):
        if s<f and s>2:
            s+=1
            g=a+b
            a=b
            b=g
        else:
            s+=1
            g=1
    return g
print(t6(10))