class animal():
    def eat(self):
        print('吃东西')
class cat(animal):
    def catch_mouse(self):
        print('抓老鼠')

class persian_cat(cat):
    def cute(self):
        print('卖萌')
c1=persian_cat()
c1.eat()
c1.catch_mouse()
c1.cute()
# ----------------
class A:
    a=1
    def pr(self):
        print('A')
class B(A):
    a=2
    def pr(self):
        print('B')
class C(B):
    a=3
    def pr(self):
        print('C')
c=C()
c.pr()
# ----------------
class animal():
    def eat(self):
        print('吃东西')
class cat(animal):
    def eat(self):
        print('吃鱼')
c1=cat()
c1.eat()
# ----------------
class person:
    def __init__(self,age):
        self._age=age
    def pr(self):
        print(self._age)
    def change_age(self,age):
        if 0<age<120:
            self._age=age
        else:
            print('年龄不合法')
p1=person()
p1.change_age(23)
p1.pr()
# ----------------
class student:
    def __init__(self,name,age,sex):
        self.name=name
        self.age=age
        self.sex=sex
    def __str__(self):
        return '姓名：{0}，性别：{1}，年龄：{2}'.format(self.name,self.sex,self.age)

s1=student('张三',23,'男')
print(s1)
# ----------------
def mul(a,b):
    try:
        c=a/b
        print(c)
    except ZeroDivisionError:
        print('除数不能为0！')
    except TypeError:
        print('类型不统一!')
    except:
        print('出现未知错误!')
mul(2,3)

try:
    mul(2, 0)
except:
    print('除数不能为0！')
# ----------------
def abc():
    raise TypeError
abc()
