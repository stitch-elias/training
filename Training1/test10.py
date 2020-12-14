import 机器学习.test9
print(机器学习.test9.add(3,4))
# ----------------
import sys
print('命令行参数如下')
for i in sys.argv:
    print(i)
print('\n\npython 路径为:',sys.path,'\n')
# ----------------
from 机器学习.test9 import add
add(3,4)
# ----------------
from  机器学习.test9 import*
print(add(3,4))
pr()
# ----------------
print(__name__)
# ----------------
a=3
b=4
print(dir())
# ----------------
import 机器学习.test9
if __name__=='__name__':
    机器学习.test9.t1()
    机器学习.test9.t2()
# ----------------
