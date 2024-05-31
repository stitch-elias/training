# from PIL import Image
#
# card_image = Image.open(
#     "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Cards/card_peashooter_move.png").resize((37, 51))
#
# card_image_ = card_image.convert('L')
#
# x, y = card_image_.size  # 获得长和宽
# # 设置每个像素点颜色的透明度
# for i in range(x):
#     for k in range(25,y):
#         color = card_image_.getpixel((i, k))
#         color = (color,color,color)
#         card_image.putpixel((i, k), color)
#
# card_image.show()
