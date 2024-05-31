import os
import random

from PIL import Image, ImageTk


class plant:
    def __init__(self, x, y, map, tag):
        self.x = x
        self.y = y + 10
        self.width = 30
        self.height = 30

        self.healthy = 5
        self.frame_index = 0
        self.frame = []

        self.image = None
        self.tkimage = None

        self.map = map
        self.tag = tag

        self.gx = int((self.x + self.width * 0.5 - self.map.grid.grid_x) // self.map.grid.grid_width)
        self.gy = int((self.y + self.height * 0.5 - self.map.grid.grid_y) // self.map.grid.grid_height)

        self.type = "plant"
        self.cost = 25
        self.attack_area = []



    def update(self):
        self.frame_index += 1
        self.frame_index %= len(self.frame)
        self.image = self.frame[self.frame_index]

    def show(self):
        self.tkimage = ImageTk.PhotoImage(self.image)
        if not self.map.Canvas.find_withtag(self.tag):
            self.map.Canvas.create_image(self.x, self.y, anchor='nw', image=self.tkimage, tag=self.tag)
        else:
            self.map.Canvas.itemconfigure(self.tag, image=self.tkimage)

    def load_images(self):
        path = ""
        for i in range(len(os.listdir(path))):
            self.frame.append(Image.open(os.path.join(path, "{}.png".format(i))).resize((50, 50)))
        self.image = self.frame[self.frame_index]

    def attacked(self, damage):
        self.image = self.frame[self.frame_index].copy()
        self.healthy -= damage
        x, y = self.image.size  # 获得长和宽
        # 设置每个像素点颜色的透明度
        for i in range(x):
            for k in range(y):
                color = self.image.getpixel((i, k))
                color = (175 + color[0],) + color[1:]
                self.image.putpixel((i, k), color)

    def reset(self):
        self.image = self.frame[self.frame_index]

    def attack(self, zombie):
        pass

    def delete(self):
        self.map.Canvas.delete(self.tag)

    def shell(self):
        self.map.grid.sun_grid.append(
            sun(self.x, self.y, self.map, "{}_shell_sun_0".format(self.tag), state=1, stopx=self.x - 10,
                stopy=self.y + 40))


class peashooter(plant):
    def __init__(self, x, y, map, tag):
        super(peashooter, self).__init__(x, y, map, tag)
        self.bullet_index = 0
        self.type = "peashooter"
        self.width = 40
        self.attack_area = [(i, self.gy) for i in range(self.gx, self.map.grid.grid_size[0])]
        self.load_images()

        self.cost = 50
        self.re_cost = 40

    def attack(self, zombies):
        self.update()
        for i in zombies:
            if (i.gx, i.gy) in self.attack_area and self.frame_index == 3:
                self.map.grid.bullet_grid.append(
                    bullet(self.x + 25, self.y, self.map, self.tag + "_bullet_{}".format(self.bullet_index)))
                self.bullet_index += 1
                self.bullet_index %= 10
                break

    def load_images(self):
        path = "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Plants/Peashooter/"
        for i in range(len(os.listdir(path))):
            self.frame.append(Image.open(os.path.join(path, "Peashooter_{}.png".format(i))).resize((50, 50)))
        self.image = self.frame[self.frame_index]


class bullet:
    def __init__(self, x, y, map, tag):
        self.x = x
        self.y = y
        self.width = 16
        self.height = 16
        self.speed = 1
        self.damage = 1
        self.state = 0
        self.image = ImageTk.PhotoImage(Image.open(
            "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Bullets/PeaNormal/PeaNormal_0.png").resize(
            (39, 24))
                                        )
        self.crush_image = ImageTk.PhotoImage(Image.open(
            "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Bullets/PeaNormalExplode/PeaNormalExplode_0.png").resize(
            (36, 62))
                                              )
        self.map = map
        self.tag = tag
        self.map.Canvas.create_image(self.x, self.y, anchor='nw', image=self.image, tag=(self.tag))

        self.gx = (self.x - self.map.grid.grid_x) // self.map.grid.grid_width
        self.gy = (self.y - self.map.grid.grid_y) // self.map.grid.grid_height

    def update(self):
        self.x += self.speed
        self.gx = (self.x - self.map.grid.grid_x) // self.map.grid.grid_width

    def show(self):
        self.map.Canvas.moveto(self.tag, self.x, self.y)

    def attack(self, zombies):
        for zombie in zombies:
            if zombie.x + zombie.rect_r <= self.x <= zombie.x + zombie.width and zombie.healthy > 0 and self.state == 0 and self.gy == zombie.gy:
                zombie.attacked(1)
                zombie.show()
                self.map.Canvas.itemconfigure(self.tag, image=self.crush_image)
                self.state = 1

    def delete(self):
        self.map.Canvas.delete(self.tag)


class sun_state:
    move = 0
    standby = 1
    miss = 2
    clicked = 3


class sun:
    def __init__(self, x, y, map, tag, type="common", state=0, stopx=0, stopy=0):
        self.x = x
        self.y = y
        self.stop_x = x
        self.stop_y = y
        self.speed_x = 0
        self.speed_y = 0
        self.state = state

        if self.state == 0:
            self.x = random.randint(10, 80) * 5
            # self.x = 5
            if self.x > 365:
                self.y = 0
            else:
                self.y = 55
            self.stop_x = self.x
            self.stop_y = random.randint(20, 70) * 5
            self.speed_y = 1
            self.speed_x = 0
        elif self.state == 1:
            if stopx != 0 or stopy != 0:
                self.stop_x = stopx
                self.stop_y = stopy
            else:
                self.stop_x = x
                self.stop_y = y
            self.speed_x = 1 if self.stop_x > self.x else -1
            self.speed_y = 1 if self.stop_y > self.y else -1

        self.map = map
        self.tag = tag

        self.size = (40, 40)
        if type == "common":
            self.size = (40, 40)
        elif type == "big":
            self.size = (60, 60)
        elif type == "small":
            self.size = (20, 20)

        self.frame = []
        self.image = None
        self.imgtk = None
        self.load_images()

        self.value = 25
        self.dead_time = 500
        self.live_time = 0
        self.type = "sun"

        self.frame_index = 0

    def show(self):
        self.tkimage = ImageTk.PhotoImage(self.image)
        if not self.map.Canvas.find_withtag(self.tag):
            self.map.Canvas.create_image(self.x, self.y - 10, anchor='nw', image=self.tkimage, tag=self.tag)
        else:
            if self.live_time >= 250:
                self.image = self.image.copy()
                x, y = self.image.size  # 获得长和宽
                # 设置每个像素点颜色的透明度
                for i in range(x):
                    for k in range(y):
                        color = self.image.getpixel((i, k))
                        color = color[:-1] + (color[-1] * 2 // 3,)
                        self.image.putpixel((i, k), color)
                self.tkimage = ImageTk.PhotoImage(self.image)
            self.map.Canvas.itemconfigure(self.tag, image=self.tkimage)
        if not (self.stop_x - 1 < self.x < self.stop_x + 1 and self.stop_y - 1 < self.y < self.stop_y + 1):
            self.map.Canvas.moveto(self.tag, int(self.x), int(self.y))

    def update(self):
        self.frame_index += 1
        self.frame_index %= len(self.frame) * 2
        self.image = self.frame[self.frame_index // 2]

        if not (self.stop_x - 1 < self.x < self.stop_x + 1):
            self.x += self.speed_x
        if not (self.stop_y - 1 < self.y < self.stop_y + 1):
            self.y += self.speed_y

        if (self.stop_x - 1 < self.x < self.stop_x + 1 and self.stop_y - 1 < self.y < self.stop_y + 1):
            if self.state == 2:
                self.state = 4
                self.map.grid.value += self.value
            elif self.state < 2:
                self.state = 3
            elif self.state == 3:
                self.live_time += 1
                if self.live_time == self.dead_time:
                    self.state = 4

    def load_images(self):
        for i in range(len(os.listdir("/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Plants/Sun"))):
            self.frame.append(Image.open(
                "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Plants/Sun/Sun_{}.png".format(
                    i)).resize(self.size))

    def delete(self):
        self.map.Canvas.delete(self.tag)

    def click(self):
        self.state = 2
        self.stop_x = 5
        self.stop_y = 5
        self.speed_x = (self.stop_x - self.x) * 0.1
        self.speed_y = (self.stop_y - self.y) * 0.1
        self.live_time = 0
        return self.value


class sunflower(plant):
    def __init__(self, x, y, map, tag):
        super(sunflower, self).__init__(x, y, map, tag)
        self.type = "sunflower"
        self.rect_r = 0
        self.rect = [25, 0, 0, 0]
        self.width = 40
        self.rect_l = self.x + 25
        self.load_images()

        self.cost = 25
        self.re_cost = 20

        self.time = 0
        self.product = len(self.frame) * 4

        self.sun_index = 0

    def load_images(self):
        path = "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Plants/SunFlower"
        for i in range(len(os.listdir(path))):
            self.frame.append(Image.open(os.path.join(path, "SunFlower_{}.png".format(i))).resize((50, 50)))
        self.image = self.frame[self.frame_index]

    def update(self):
        self.frame_index += 1
        self.frame_index %= len(self.frame)
        self.image = self.frame[self.frame_index]

        self.time += 1

        if self.time % self.product == self.product - 1:
            self.map.grid.sun_grid.append(
                sun(self.x, self.y, self.map, "{}_sun_{}".format(self.tag, self.sun_index), state=1, stopx=self.x - 10,
                    stopy=self.y + 40))
            self.sun_index += 1
            self.map.grid.sun_grid.append(
                sun(self.x, self.y, self.map, "{}_sun_{}".format(self.tag, self.sun_index), state=1, stopx=self.x + 10,
                    stopy=self.y + 40))
            self.sun_index += 1
            self.time = 0
            self.sun_index %= 10

        if self.time > (self.product // 2):
            self.image = self.image.copy()
            x, y = self.image.size  # 获得长和宽
            # 设置每个像素点颜色的透明度
            for i in range(x):
                for k in range(y):
                    color = self.image.getpixel((i, k))
                    color = (color[0] + 175, color[1] + 107) + color[2:]
                    self.image.putpixel((i, k), color)

    def show(self):
        self.tkimage = ImageTk.PhotoImage(self.image)
        if not self.map.Canvas.find_withtag(self.tag):
            self.map.Canvas.create_image(self.x, self.y, anchor='nw', image=self.tkimage, tag=self.tag)
        else:
            self.map.Canvas.itemconfigure(self.tag, image=self.tkimage)
