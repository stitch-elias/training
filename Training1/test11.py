class person:
    def open_door(self,door):
        door.opened()
class door:
    def opened(self):
        print('门开了')
p1=person()
d1=door()
p1.open_door(d1)
# ----------------
class car:
    color='白色'
    name='特斯拉'
    speed='360km/h'
    def run(self):
        print('车跑起来了：颜色：{0}，型号{1}，时速：{2}'.format(self.color,self.name,self.speed))
c1=car()
c1.run()
c1.color='红色'
c1.run()
car.color='蓝色'
print(car.color)
# ----------------
class car:
    def __init__(self):
        self.color='白色'
        self.name = '特斯拉'
        self.speed = '360km/h'

    def run(self):
        print('车跑起来了：颜色：{0}，型号{1}，时速：{2}'.format(self.color, self.name, self.speed))
c1=car()
c2=car()
c1.color='红色'
c1.run()
# ----------------
class person:
    def __init__(self,name):
        self.name=name
        self.cry()
    def cry(self):
        print(self.name,'哭了')
p1=person('李四')
p2=person('张三')
# ----------------
class person:
    def eat(self):
        print('吃饭')

    def speak(self):
        print('说话')
class student(person):
    def study(self):
        print('读书学习写作业')
class worker(person):
    def work(self):
        print('板砖')
s1=student()
w1=worker()
s1.speak()
s1.study()
s1.eat()
w1.eat()
w1.speak()
w1.work()