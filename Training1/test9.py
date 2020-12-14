# def ti():
#     a=3
#     print(a)
# ti()
# print(a)
# ----------------
name='张三'
name=str('张三')#内建
def outer():
    a=3#闭包
    def inner():
        b=2#局部
        print(a)
        print(name)
    inner()

outer()
# ----------------
if True:
    a=3
def t1():
    b=4
print(a)
#print(b)
# ----------------
total=0
def sum(arg1,arg2):
    global total
    total=arg1+arg2
    print('局部变量',total)
    return  total
sum(1,2)
print('全局变量',total)
# ----------------
# a=6
# def change(a):
#     global a
#     a=4
#     print(a)
# change(a)
# print(a)
# ----------------
def outer():
    a=1
    def inner():
        nonlocal a
        a=2
        print(a)
    inner()
    print(a)
outer()
# ----------------
list=['a','b','c','d']
for i,j in enumerate(list):
    print(i,'=>>',j)
# ----------------
def add(a,b):
    return a+b
# ----------------
def pr():
    for i in range(5):
        for j in range(5):
            print('*',end='')
        print()
# ----------------
print(__name__)
# ----------------
import 机器学习.test10
def t1():
    print('1')
def t2():
    print('2')




