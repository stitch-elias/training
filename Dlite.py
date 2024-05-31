import numpy as np

map = np.zeros((6,4))
map[1:5,0]=1
map[1:5,2]=1
print(map)

s = [0,3,0,6]
g = [5,0,6,0]

rhs = np.ones((6,4))*500
key = np.ones((6,4))*500
parent = np.zeros((6,4,2))

state=np.zeros((6,4))
state[s[0],s[1]]=2
rhs[s[0],s[1]]=0
key[s[0],s[1]]=0
while True:
    for i in np.argwhere(state==2):
        minx = max(0,i[1]-1)
        miny = max(0,i[0]-1)
        maxx = min(3,i[1]+1)
        maxy = min(5,i[0]+1)
        for j in range(miny,maxy+1):
            for k in range(minx,maxx+1):
                if j==i[0] and k==i[1]:
                    continue
                if map[j,k]!=1 and state[j,k]==0:
                    state[j,k]=1
                    rhs[j,k]=rhs[i[0],i[1]]+1
                    key[j,k]=key[i[0],i[1]]+1
                    parent[j,k,0]=i[0]
                    parent[j,k,1]=i[1]
                if map[j,k]!=1 and state[j,k]==1:
                    if rhs[i[0],i[1]]+1<rhs[j,k]:
                        rhs[j,k]=rhs[i[0],i[1]]+1
                        key[j,k]=key[i[0],i[1]]+1
                        parent[j,k,0]=i[0]
                        parent[j,k,1]=i[1]

    if state[g[0],g[1]]==1:
        state[g[0], g[1]]=2
        break
    mask = (state==1)*rhs
    for i in np.argwhere(mask==0):
        mask[i[0],i[1]]=500
    alph = np.argwhere(mask==np.min(mask))
    state[alph[0][0],alph[0][1]]=2

map0=map.copy()
map0[g[0],g[1]]=2
gx = g[1]
gy = g[0]
while True:
    index = parent[int(gy),int(gx)]
    map0[int(index[0]),int(index[1])] = 2
    gx = index[1]
    gy = index[0]
    if gx==s[1] and gy==s[0]:
        map0[s[0], s[1]] = 2
        break
print(map0)

map[3,1]=1

map1 = map.copy()
state[3,1]=0
rhs[3,1]=500
parent[3,1,:]=0
gx = 1
gy = 3
while True:
    index = np.argwhere((parent[:,:,0]==gy)*(parent[:,:,1]==gx))
    for i in index:
        gx = i[1]
        gy = i[0]
        state[int(gy), int(gx)] = 0
        rhs[int(gy), int(gx)] = 500
        parent[int(gy), int(gx), :] = 0

    if len(index)==0:
        break


while True:
    for i in np.argwhere(state==2):
        minx = max(0,i[1]-1)
        miny = max(0,i[0]-1)
        maxx = min(3,i[1]+1)
        maxy = min(5,i[0]+1)
        for j in range(miny,maxy+1):
            for k in range(minx,maxx+1):
                if j==i[0] and k==i[1]:
                    continue
                if map[j,k]!=1 and state[j,k]==0:
                    state[j,k]=1
                    rhs[j,k]=rhs[i[0],i[1]]+1
                    key[j,k]=key[i[0],i[1]]+1
                    parent[j,k,0]=i[0]
                    parent[j,k,1]=i[1]
                if map[j,k]!=1 and state[j,k]==1:
                    if rhs[i[0],i[1]]+1<rhs[j,k]:
                        rhs[j,k]=rhs[i[0],i[1]]+1
                        key[j,k]=key[i[0],i[1]]+1
                        parent[j,k,0]=i[0]
                        parent[j,k,1]=i[1]

    if state[g[0],g[1]]==1:
        state[g[0], g[1]]=2
        break
    mask = (state==1)*rhs
    for i in np.argwhere(mask==0):
        mask[i[0],i[1]]=500
    alph = np.argwhere(mask==np.min(mask))
    state[alph[0][0],alph[0][1]]=2

map2=map.copy()
map2[g[0],g[1]]=2
gx = g[1]
gy = g[0]
while True:
    index = parent[int(gy),int(gx)]
    map2[int(index[0]),int(index[1])] = 2
    gx = index[1]
    gy = index[0]
    if gx==s[1] and gy==s[0]:
        map0[s[0], s[1]] = 2
        break
print(map2)


map[3,1]=0

map3 = map.copy()
state[3,1]=0
rhs[3,1]=0
parent[3,1,:]=0

minx = max(0, g[1] - 1)
miny = max(0, g[0] - 1)
maxx = min(3, g[1] + 1)
maxy = min(5, g[0] + 1)
for j in range(miny, maxy + 1):
    for k in range(minx, maxx + 1):
        state[j,k]=0
while True:
    for i in np.argwhere(state==2):
        minx = max(0,i[1]-1)
        miny = max(0,i[0]-1)
        maxx = min(3,i[1]+1)
        maxy = min(5,i[0]+1)
        for j in range(miny,maxy+1):
            for k in range(minx,maxx+1):
                if j==i[0] and k==i[1]:
                    continue
                if map[j,k]!=1 and state[j,k]==0:
                    state[j,k]=1
                    rhs[j,k]=rhs[i[0],i[1]]+1
                    key[j,k]=key[i[0],i[1]]+1
                    parent[j,k,0]=i[0]
                    parent[j,k,1]=i[1]
                if map[j,k]!=1 and state[j,k]==1:
                    if rhs[i[0],i[1]]+1<rhs[j,k]:
                        rhs[j,k]=rhs[i[0],i[1]]+1
                        key[j,k]=key[i[0],i[1]]+1
                        parent[j,k,0]=i[0]
                        parent[j,k,1]=i[1]

    if state[g[0],g[1]]==1:
        state[g[0], g[1]]=2
        break
    mask = (state==1)*rhs
    for i in np.argwhere(mask==0):
        mask[i[0],i[1]]=500
    alph = np.argwhere(mask==np.min(mask))
    state[alph[0][0],alph[0][1]]=2

map4=map.copy()
map4[g[0],g[1]]=2
gx = g[1]
gy = g[0]
while True:
    index = parent[int(gy),int(gx)]
    map4[int(index[0]),int(index[1])] = 2
    gx = index[1]
    gy = index[0]
    if gx==s[1] and gy==s[0]:
        map0[s[0], s[1]] = 2
        break
print(map4)