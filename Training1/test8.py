a=2
b=6
print(a+b)
a=7
b=8
print(a+b)
# ----------------
print(max(4,18))
print(min(5,7))
# ----------------
def pr_tr():
    for i in range(5):
        for j in range(5):
            print('*',end='')
        print()
    print()
for i in range(3):
    pr_tr()
# ----------------
def add(a,b):
    print(a+b)
add(7,8)
add(12,13)
# ----------------
def my_max():
    if a>b:
        print(a)
    else:
        print(b)
my_max()
# ----------------
list=[4,1,8,9,3,7]
def my_sort(my_list,flag):
    for i in range(len(my_list)-1):
        for j in range(i+1,len(my_list)):
            if flag==True:
                if my_list[i]>my_list[j]:
                    temp=my_list[i]
                    my_list[i]=my_list[j]
                    my_list[j]=temp
            else:
                if my_list[i]<my_list[j]:
                    temp=my_list[i]
                    my_list[i]=my_list[j]
                    my_list[j]=temp
    print(my_list)
my_sort(list,flag=True)
my_sort(list,flag=False)
# ----------------
def add(a,b):
    return a+b
def t1():
    return
def t2():
    print('hello')
    return
print(add(3,4))
print(t1)
print(t2)
# ----------------
def t1():
    for i in range(5):
        print(i)
        return
t1()
# ----------------
a=3
print(type(a))
a='123'
print(type(a))
a=[1,2,3]
print(type(a))
# ----------------
def add(a,b):
    pass
print(add(1,2))
# ----------------
a=1
def chage(a):
    a=3
chage(a)
print(a)
a=[1,2]
def chage(c):
    c.append(3)
    return c
chage(a)
print(a)
print(chage(a))
# ----------------
def pr(name,age=0):
    print('姓名：',name)
    print('年龄：',age)
pr(age=23,name='张三')
pr('张三')
# ----------------
def add(a,*b):
    sum=0
    for i in b:
        sum+=i
    return a+sum
add(1,2,3,4)
add(1)
# ----------------
sum=lambda a,b:a+b
print(sum(2,3))
