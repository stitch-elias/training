a=3
a+4
a,b,*c="你好吗?"
print(a,b,c)
#----------------
sex='男'
if sex=='男':
    print('进去')
elif sex=='女':
    print('进女厕')
else:
    print('输入错误')
print('程序结束')
#----------------
num=9
if num>0:
    print('a')
elif num>1:
    print('b')
elif num>3:
    print('c')
elif num>4:
    print('d')
elif num > 5:
    print('e')
else:
    print('f')
#----------------
if 1:
    print("123")
print("程序结束")
#----------------
sex='男'
num=80
if sex=='男' and num>60:
    print('yes')
else:
    print('no')
#----------------
sex='男'
if sex=='男':
    if num>60:
        print('yes')
else:
    print('no')
#----------------
# a=0
# b=1
# print(b/a)
#----------------
a=0
b=1
if (a>0) and (b/a>2):
    print('yes')
else:
    print('no')
print(3*2+5)
a=3
if a>0:print('123')
#----------------
a=3
while a>0:
    print('hello')
    a-=1
#----------------
a=1
while a<=100:
    if a%2==0:
        print(a)
    a+=13
#----------------
num=123
while num>0:print(num)
#----------------
while True:
    print('hello')
#----------------
a=[1,2,3,4,5,6,7,8,9]
for num in range(len(a)):
    print(num)
else:
    print('循环结束')
#----------------
a=range(3)
print(a)
for i in range(1,9,2):
    print(a)
print(range(1,9,2))
#----------------
sum=0
for num in range(1,101):
    sum+=sum
print(sum)
#----------------
for num in range(100,1000):
    a=num//100
    b=num//10%10
    c=num%10
    if a**3+b**3+c**3==num:
        print(num)
#----------------
for i in range(3):
    for j in range(3):
        print('Hello')
#----------------
for j in  range(5):
    for i in range(5):
        print('*',end='')
    print()
#----------------
for j in  range(1,10):
    for i in range(1,j+1):
        print('{0}*{1}={2}'.format(j,i,j*i),end='')
    print()
#----------------
for i in  range(3):
    for j in range(3):
        for x in range(3):
            print('Hello')
            break
# ----------------
a=0
while True:
    if a==3:
        break
    print(a)
    a+=1
# ----------------
a=0
while True:
    if a==8:
        a+=1
        continue
    print(a)
    a += 1
print('程序结束')
# ----------------
print(char(97))