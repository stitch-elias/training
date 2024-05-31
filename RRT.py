import numpy as np
import cv2
import random

def line(startx, starty, endx, endy):
    line=[]
    if startx >= endx:
        for x in range(endx, startx):
            y = int((x - startx) * (endy - starty) / (endx - startx)) + starty
            line.append([x,y])
    else:
        for x in range(startx, endx):
            y = int((x - startx) * (endy - starty) / (endx - startx)) + starty
            line.append([x,y])
    return line

img0 = cv2.imread("6665.jpeg")
img = cv2.cvtColor(img0,cv2.COLOR_RGB2GRAY)
_,img = cv2.threshold(img,175,255,cv2.THRESH_BINARY )
# img = cv2.resize(img,(260,205))
# cv2.imshow("",img)
# cv2.waitKey()

s = [87,44,0,0]
g = [90,162,0,0]
step = 5
l = list(range(0,9))
nodelist = [s]
while(True):
    end=False
    for i in nodelist:
        if(np.sqrt((g[1] - i[1]) ** 2 + (g[0] - i[0]) ** 2) < 5):
            end=True
            break
    if end:
        break
    movenode = []
    if random.sample(l,1)[0]%10==0:
        movenode=g
    else:
        while len(movenode)==0:
            x = random.randint(1,259)
            y = random.randint(1,203)
            if img[y,x]!=0:
                movenode=[x,y]
    choosenode = []
    distance=-1
    for i in nodelist:
        if distance==-1:
            distance=abs(i[0]-movenode[0])+abs(i[1]-movenode[1])
            choosenode=i
        elif distance>abs(i[0]-movenode[0])+abs(i[1]-movenode[1]):
            distance=abs(i[0]-movenode[0])+abs(i[1]-movenode[1])
            choosenode=i

    if(movenode[0]-choosenode[0]==0 and movenode[1]-choosenode[1]==0):
        continue
    if(movenode[0]-choosenode[0]==0):
        dx=0
        dy=(movenode[1]-choosenode[1])//abs((movenode[1]-choosenode[1]))*5
    elif(movenode[1]-choosenode[1]==0):
        dx=(movenode[0]-choosenode[0])//abs((movenode[0]-choosenode[0]))*5
        dy=0
    else:
        delt = np.arctan((movenode[1]-choosenode[1])/(movenode[0]-choosenode[0]))
        dx = int(5*np.sin(delt))
        dy = int(5*np.cos(delt))

    if dx+choosenode[0]<=0 or dx+choosenode[0]>=260 or dy+choosenode[1]<=0 or dy+choosenode[1] >=204:
        continue

    if [dx+choosenode[0],dy+choosenode[1],choosenode[0],choosenode[1]] in nodelist:
        continue

    if img[dy+choosenode[1],dx+choosenode[0]]==0:
        continue
    flag=0
    for i in line(choosenode[0], choosenode[1], dx+choosenode[0], dy+choosenode[1]):
        if img[i[1],i[0]]==0:
            flag=1
            break
    if flag==1:
        continue

    result = [dx+choosenode[0],dy+choosenode[1],choosenode[0],choosenode[1]]
    for i in nodelist:
        if np.sqrt((i[0]-result[0])**2+(i[1]-result[1])**2)<np.sqrt((result[2]-result[0])**2+(result[3]-result[1])**2):
            result[2]=i[0]
            result[3]=i[1]
    nodelist.append(result)
print(nodelist)

# img1 = cv2.resize(img0,(120,115))
img1 = img0
cv2.circle(img1, (int(s[0]), int(s[1])), 1, (255, 0, 0), 10)
cv2.circle(img1, (int(g[0]), int(g[1])), 1, (255, 0, 0), 10)

ex=nodelist[-1][2]
ey=nodelist[-1][3]
cv2.circle(img1, (int(nodelist[-1][0]), int(nodelist[-1][1])), 1, (0,0,255), 1)
for i in range(len(nodelist)-1,-1,-1):
    cv2.circle(img1, (int(ex), int(ey)), 1, (0, 0, 255), 1)
    if(nodelist[i][0]==ex and nodelist[i][1]==ey):
        ex=nodelist[i][2]
        ey=nodelist[i][3]
cv2.imshow("",img1)
cv2.waitKey()