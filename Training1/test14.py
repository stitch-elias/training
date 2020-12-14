import json
data = {
    'no':1,
    'name':'猎维',
    'url':'http://www.liev.top',
    None:True
}
json_str=json.dumps(data)
print('原数据：',data)
print('JSON数据：',json_str)
print(type(data))
print(type(json_str))
str=json.loads(json_str)
print(str)
print(type(str))
# ----------------
import  json
data=(1,2,3,4,5)
# with open('data.json','w') as f:
#     json.dump(data,f)
with open('data.json','r') as f:
    data1=json.load(f)
print(data1)
print(type(data1))
data1=tuple(data1)
print(data1)
# ----------------
try:
    import xml.etree.cElementTree as et
except ImportError:
    import xml.etree.ElementTree as et
tree=et.parse('movies.xml')
root=tree.getroot()
print(root,root.get('shelf'))
for movie in root.findall('movie'):
    title=movie.get('title')
    print(title)
    type=movie.find('type').text
    print(type)
    format=movie.find('format').text
    print(format)
# ----------------
import threading
import time
def fun1(a):
    for i in range(10):
        print(threading.current_thread().getName(),i)
        time.sleep(a)
def fun2(a):
    for i in range(65,91):
        print(threading.current_thread().getName(),chr(i))
thread1=threading.Thread(target=fun1,name='线程1',args=(1,))
thread2=threading.Thread(target=fun2,name='线程2',args=(1,))
thread1.start()
thread2.start()
# ----------------
class MyThread1(threading.Thread):
    def __init__(self,a,name):
        super().__init__(args=(a,),name=name)
        self.a=a
    def run(self):
        fun1(self.a)
class MyThread2(threading.Thread):
    def __init__(self,a,name):
        super().__init__(name=name)
        self.a=a
    def run(self):
        fun2(self.a)
thread1=MyThread1(1,'线程1')
thread2=MyThread2(1,'线程2')
thread1.start()
thread2.start()
# ----------------
class A:
    a=1
    def fun(self):
        print('a')
a1=A()
a1.fun()
# ----------------
import threading
def fun1():
    for i in range(10):
        print(threading.current_thread().getName(),i)
        if i == 5:
            thread2=threading.Thread(target=fun2,name='线程2')
            thread2.start()
            thread2.join()
def fun2():
    for i in range(65,91):
        print(threading.current_thread().getName(),chr(i))
thread1=threading.Thread(target=fun1,name='线程1')
thread1.start()
# ----------------
import threading
lock=threading.Lock()
ticket = 100
def seal():
    global ticket
    global lock
    lock.acquire()
    while ticket>=0:
        print(threading.current_thread().getName(),ticket)
        ticket-=1
    lock.release()
for i in range(3):
    thread=threading.Thread(target=seal,name='线程{}'.format(i+1))
    thread.start()
# ----------------
import time
print(time.asctime(time.localtime()))
print(time.strftime('%a %b %d %H:%M:%S %Y',time.localtime()))
print(time.time())
# ----------------
import calendar
cal=calendar.month(2016,1)
print(cal)