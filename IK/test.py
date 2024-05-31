from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# 设置图例字号
mpl.rcParams['legend.fontsize'] = 10

# 方式2：设置三维图形模式
fig = plt.figure()
ax = fig.add_axes(Axes3D(fig))

# 测试数据
# x = np.linspace(-4 * np.pi, 4 * np.pi, 30)
# y = x + np.random.randn(x.shape[-1]) * 2.5
# z = x * x
x=y=z = np.array([1,100])

ax.scatter(x,y,z) # 画出(x,y,z)的散点图。
ax.plot(x, y, z, label='parametric curve')

# X1 = X2 = np.arange(-5,15,1)
# X1,X2 = np.meshgrid(X1,X2)
#
# Z = 1/2*X1**2
#
# plt.ion()
# fig = plt.figure()
#
# azim = -60
# elev = 30
#
# for i in range(3000):
#     fig.clf()
#     ax = fig.add_axes(Axes3D(fig))
#
#     ax.view_init(elev,azim)
#
#     ax.plot_surface(X1,X2,Z,cmap=mpl.cm.gist_rainbow)
#     plt.pause(0.001)
#
#     # elev,azim = ax.elev,ax.azim
#
#     Z = Z-X1+2*X2
#
#
# # # 显示图例
# # ax.legend()
# plt.ioff()



# 显示图形
plt.show()