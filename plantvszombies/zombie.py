import os
from PIL import Image, ImageTk


class zombie:
    def __init__(self, x, y, map, tag):
        self.x = x
        self.y = y - 30 + 50
        self.rect_r = int(62 * 0.7)
        self.width = int(90 * 0.7)
        self.height = 101
        self.speed = 1
        self.frame_index = 0
        self.frame = []

        self.common_frame = []
        self.dead_frame = []
        self.attack_frame = []
        self.losthead_attack_frame = []
        self.losthead_frame = []

        self.map = map
        self.tag = tag
        self.state = 0

        self.gx = (self.x + self.rect_r - self.map.grid.grid_x) // self.map.grid.grid_width
        self.gy = (self.y + self.height * 0.5 - self.map.grid.grid_y) // self.map.grid.grid_height

        self.healthy = 10

        self.load_images()

        self.image = self.frame[self.frame_index]
        self.tkimage = ImageTk.PhotoImage(self.image)

    def load_images(self):
        for i in range(len(os.listdir(
                "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Zombies/NormalZombie/Zombie"))):
            self.common_frame.append(Image.open(
                "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Zombies/NormalZombie/Zombie/Zombie_{}.png".format(
                    i)).resize((116, 101)))
        for i in range(len(os.listdir(
                "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Zombies/NormalZombie/ZombieDie"))):
            self.dead_frame.append(Image.open(
                "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Zombies/NormalZombie/ZombieDie/ZombieDie_{}.png".format(
                    i)).resize((116, 101)))
        for i in range(len(os.listdir(
                "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Zombies/NormalZombie/ZombieAttack"))):
            self.attack_frame.append(Image.open(
                "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Zombies/NormalZombie/ZombieAttack/ZombieAttack_{}.png".format(
                    i)).resize((116, 101)))
        for i in range(len(os.listdir(
                "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Zombies/NormalZombie/ZombieLostHead"))):
            self.losthead_frame.append(Image.open(
                "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Zombies/NormalZombie/ZombieLostHead/ZombieLostHead_{}.png".format(
                    i)).resize((116, 101)))
        for i in range(len(os.listdir(
                "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Zombies/NormalZombie/ZombieLostHeadAttack"))):
            self.losthead_attack_frame.append(Image.open(
                "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Zombies/NormalZombie/ZombieLostHeadAttack/ZombieLostHeadAttack_{}.png".format(
                    i)).resize((116, 101)))

        self.frame = self.common_frame
        self.image = self.frame[self.frame_index]

    def attacked(self, damage):
        self.healthy -= damage
        if self.healthy == 5:
            if self.state == 2:
                self.frame = self.losthead_attack_frame
            else:
                self.frame = self.losthead_frame
            self.map.grid.head_grid.append(head(self.x + self.rect_r, self.y, self.map.Canvas, self.tag + "_head"))
            self.frame_index = 0
        elif self.healthy == 0:
            self.frame = self.dead_frame
            self.frame_index = 0
            self.state = 1
        self.image = self.frame[self.frame_index].copy()
        x, y = self.image.size  # 获得长和宽
        # 设置每个像素点颜色的透明度
        for i in range(x):
            for k in range(y):
                color = self.image.getpixel((i, k))
                color = (175 + color[0],) + color[1:]
                self.image.putpixel((i, k), color)

    def update(self):
        if self.state != 2:
            self.x -= self.speed
            self.gx = (self.x + self.rect_r - self.map.grid.grid_x) // self.map.grid.grid_width
        self.frame_index += 1
        self.frame_index %= len(self.frame)
        self.image = self.frame[self.frame_index]

    def show(self):
        self.tkimage = ImageTk.PhotoImage(self.image)
        if not self.map.Canvas.find_withtag(self.tag):
            self.map.Canvas.create_image(self.x, self.y, anchor='nw', image=self.tkimage, tag=self.tag)
        else:
            self.map.Canvas.itemconfigure(self.tag, image=self.tkimage)
        self.map.Canvas.moveto(self.tag, self.x, self.y)

    def attack(self, plants):
        for i in plants:
            if i.x < (self.rect_r + self.x) < (i.x + i.width) and self.gy == i.gy:
                if self.state == 0:
                    if self.healthy > 5:
                        self.frame = self.attack_frame
                    else:
                        self.frame = self.losthead_attack_frame
                    self.frame_index = 0
                    self.state = 2
                elif self.state == 2 and self.frame_index % 5 == 4:
                    i.attacked(1)
                    i.show()
                return
        if self.state == 2:
            self.state = 0
            if self.healthy > 5:
                self.frame = self.common_frame
            else:
                self.frame = self.losthead_frame
            self.frame_index = 0

    def delete(self):
        self.map.Canvas.delete(self.tag)


class head:
    def __init__(self, x, y, Canvas, tag):
        self.x = x
        self.y = y
        self.frame_index = 0
        self.frame = []

        self.imgtk = None
        self.image = None
        self.Canvas = Canvas
        self.tag = tag

        self.load_images()

    def load_images(self):
        for i in range(len(os.listdir(
                "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Zombies/NormalZombie/ZombieHead/"))):
            self.frame.append(Image.open(
                "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Zombies/NormalZombie/ZombieHead/ZombieHead_{}.png".format(
                    i)).resize((105, 130)))

    def show(self):
        self.tkimage = ImageTk.PhotoImage(self.image)
        if not self.Canvas.find_withtag(self.tag):
            self.Canvas.create_image(self.x, self.y - 10, anchor='nw', image=self.tkimage, tag=self.tag)
        else:
            self.Canvas.itemconfigure(self.tag, image=self.tkimage)

    def update(self):
        self.frame_index += 1
        self.frame_index %= len(self.frame)
        self.image = self.frame[self.frame_index]

    def delete(self):
        self.Canvas.delete(self.tag)
