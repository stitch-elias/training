import numpy as np
arr1=np.asarray([[0,0,0],[1,1,1],[2,2,2],[3,3,3]])
arr2=np.asarray([1,2,3])
arr_sum=arr1+arr2
print(arr1.shape)
print(arr2.shape)
print(arr_sum)
# ----------------
num1=input('请输入第一个数字')
num2=input('请输入第二个数字')
print(num1+num2)
# ----------------
path='test.txt'
file=open(path,'r')
#file.write('python是一门很好的语言')
#file.close()
str=file.read()
print(str)
# ----------------
import os
path=r'E:\py\training\test.txt'
file_path,file_name=os.path.split(path)
print(file_path,file_name)
if not os.path.exists(file_path):
    os.makedirs(file_path)
file=open(path,'w')
file.write('python是一门很好的语言')
file.close()
# ----------------
path=r'list_bbow_celeba.txt'
file=open(path,'r')
#str=file.read()
str=file.readlines()
for i in str:
    print(i)
# ----------------
path=r'list_bbow_celeba.txt'
file=open(path,'r')
for line in file:
    print(line)
print(type(file))
# ----------------
path=r'test.txt'
file=open(path,'wb')
file.write(bytes(97))
file.close()
# ----------------
