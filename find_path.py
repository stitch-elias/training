import numpy as np

a = np.zeros((10,10))
a[4,1:-1]=1
a[5,1]=1
a[6,1]=1
a[7,1]=1
a[5,-2]=1
a[6,-2]=1
a[7,-2]=1

s = [2,5]
c = [8,5,0,0,0]

a[8,5]=2
a[2,8]=3

opened = []
closed = []

closed.append(c)
kk = 1
while True:
    kk+=1
    if len(opened)==0 and len(closed)!=1:
        break
    for i in closed:
        if a[i[0]-1,i[1]] not in [1,2] and [i[0]-1,i[1]] not in [[j[0],j[1]]for j in opened ]:
            print(i)
            opened.append([i[0]-1,i[1],abs(i[0]-1-2)+abs(i[1]-8),i[3]+1,i[0],i[1]])
        if i[0]+1<10:
            if a[i[0]+1,i[1]] not in [1,2] and [i[0]+1,i[1]] not in [[j[0],j[1]]for j in opened ]:
                opened.append([i[0]+1,i[1],abs(i[0]+1-2)+abs(i[1]-8),i[3]+1,i[0],i[1]])
        if a[i[0],i[1]-1] not in [1,2] and [i[0],i[1]-1] not in [[j[0],j[1]]for j in opened ]:
            opened.append([i[0],i[1]-1,abs(i[0]-2)+abs(i[1]-1-8),i[3]+1,i[0],i[1]])
        if i[1]+1<10:
            if a[i[0],i[1]+1] not in [1,2] and [i[0],i[1]+1] not in [[j[0],j[1]]for j in opened ]:
                opened.append([i[0],i[1]+1,abs(i[0]-2)+abs(i[1]+1-8),i[3]+1,i[0],i[1]])
    m=-1
    index = 0
    for i in range(len(opened)):
        if m ==-1:
            m=opened[i][2]+opened[i][3]
            index=i
        elif opened[i][2]+opened[i][3]<m:
            m=opened[i][2]+opened[i][3]
            index=i
    closed.append([opened[index][0],opened[index][1],opened[index][2],opened[index][3],opened[index][4],opened[index][5]])
    if a[opened[index][0],opened[index][1]]==3:
        break
    a[opened[index][0], opened[index][1]]=2
    opened.pop(index)
    # print(a)
print(a)
print(closed)
print(kk)