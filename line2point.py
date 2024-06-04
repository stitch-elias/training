import math
import cv2
import numpy as np

def circleLeastFit(points):
    N = len(points)
    sum_x = 0
    sum_y = 0
    sum_x2 = 0
    sum_y2 = 0
    sum_x3 = 0
    sum_y3 = 0
    sum_xy = 0
    sum_x1y2 = 0
    sum_x2y1 = 0
    for i in range(N):
        x = float(points[i][0][0])
        y = float(points[i][0][1])
        x2 = x*x
        y2 = y*y
        sum_x += x
        sum_y += y
        sum_x2 += x2
        sum_y2 += y2
        sum_x3 += x2 * x
        sum_y3 += y2 * y
        sum_xy += x*y
        sum_x1y2 += x*y2
        sum_x2y1 += x2*y
    C = N*sum_x2-sum_x*sum_x
    D = N*sum_xy-sum_x*sum_y
    E = N*sum_x3+N*sum_x1y2-(sum_x2+sum_y2)*sum_x
    G = N*sum_y2-sum_y*sum_y
    H = N*sum_x2y1 + N*sum_y3 - (sum_x2+sum_y2)*sum_y
    a = (H*D-E*G)/(C*G-D*D+1e-100)
    b = (H*C-E*D)/(D*D-G*C+1e-100)
    c = -(a*sum_x+b*sum_y+sum_x2+sum_y2)/N
    centerx = a/(-2)
    centery = b/(-2)
    rad = math.sqrt(a*a+b*b-4*c)/2
    return centerx,centery,rad


image = cv2.imread("/home/qiu/下载/2024_05_31_0h8_Kleki.png")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
_,thre = cv2.threshold(gray,175,255,cv2.THRESH_BINARY_INV)
c,_ = cv2.findContours(thre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

apsilon = cv2.approxPolyDP(c[0],0.1*cv2.arcLength(c[0],True),True)

direction = apsilon[1]-apsilon[0]
direction = direction/np.linalg.norm(direction)
normal = np.array([-direction[0,1],direction[0,0]])
C = -np.dot(normal,apsilon[1,0])
m_distance = 0
m_point = 0
for i in c[0]:
    distance = np.abs(np.dot(normal,i[0])+C)/np.linalg.norm(normal)
    if distance>m_distance:
        m_distance=distance
        m_point=i

center_point = (apsilon[1]+apsilon[0])/2
distance_line = apsilon[1]-apsilon[0]
k = distance_line[0,1]/distance_line[0,0]
b_line = center_point[0,1]-center_point[0,0]*k

k_point = -1/k
b_point = center_point[0,1]-center_point[0,0]*k_point

# m_distance*sqrt(k*k+1) = abs(kx-k_point*x-b_point+b_line)
# (k-k_point)**2*x**2+2*(k-k_point)*(b_line-b_point)*x+((b_line-b_point)**2=k*k+1)
b_ = 2*(k-k_point)*(b_line-b_point)
a_ = (k-k_point)*(k-k_point)
c_ = ((b_line-b_point)*(b_line-b_point)-(k*k+1)*m_distance*m_distance)
D_ = b_*b_-4*a_*c_
x1 = (-b_+math.sqrt(D_))/(2*a_)
x2 = (-b_-math.sqrt(D_))/(2*a_)
y1 = k_point*x1+b_point

m_point[0,0]=x1
m_point[0,1]=y1
backgrand = np.zeros(image.shape,dtype=np.uint8)
points = np.array([apsilon[0],m_point,apsilon[1]])

nnx,nny,rr = circleLeastFit(points)

cv2.circle(image, m_point[0], 5, (255, 0, 255))
cv2.circle(backgrand, [int(nnx),int(nny)], 5, (255, 0, 255))
cv2.circle(image, [int(nnx),int(nny)], int(rr), (255, 0, 255))
for i in c[0]:
    cv2.circle(image,i[0],1,(255,255,100))


# circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,100,param1=50,param2=30,minRadius=10,maxRadius=500)
# x = circles[0][0][0]-circles[0][0][2]
# y = circles[0][0][0]-circles[0][0][1]
# w = h = 2*circles[0][0][2]
# center = (int(circles[0][0][0]),int(circles[0][0][1]))
# radius = circles[0][0][2]
# trans_centers = (int(center[0]-x),int(center[1]-y))
#
# flags = cv2.INTER_CUBIC + cv2.WARP_POLAR_LINEAR
# linear_polar_image = cv2.warpPolar(image, trans_centers, center, radius*2, flags)

cv2.imshow("",image)
cv2.waitKey()

