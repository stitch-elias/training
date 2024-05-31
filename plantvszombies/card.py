from plant import peashooter, sunflower
from PIL import ImageTk, Image


class card:
    def __init__(self, Canvas, x, y, tag):
        self.Canvas = Canvas
        self.tag = tag
        self.cost = 50
        self.x = x
        self.y = y

        self.p_card_image = None
        self.card_image = None
        self.image_plant = None

        self.time = 100
        self.freeze_time = 100

    def show(self):
        self.p_card_image = Image.open(
            "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Cards/card_peashooter_move.png").resize(
            (37, 51))
        self.card_image = ImageTk.PhotoImage(self.p_card_image)
        self.image_plant = Image.open(
            "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Plants/Peashooter/Peashooter_0.png").resize(
            (50, 50))
        x0, y0 = self.image_plant.size  # 获得长和宽
        # 设置每个像素点颜色的透明度
        for i in range(x0):
            for k in range(y0):
                color = self.image_plant.getpixel((i, k))
                color = color[:-1] + (color[-1] // 2,)
                self.image_plant.putpixel((i, k), color)
        self.image_plant = ImageTk.PhotoImage(self.image_plant)
        self.Canvas.create_image(self.x, self.y, anchor='nw', image=self.card_image, tag=(self.tag))

    def move(self, x, y):
        self.Canvas.moveto(self.tag, x, y)
        self.x = x
        self.y = y

    def be_chose(self, e):
        self.Canvas.moveto(self.tag, e.x - 15, e.y - 15)
        self.Canvas.itemconfigure(self.tag, image=self.image_plant)

    def common(self):
        self.Canvas.moveto(self.tag, self.x, self.y)
        self.Canvas.itemconfigure(self.tag, image=self.card_image)

    def create_plant(self, x, y, parent):
        gx = (x - parent.grid.grid_x) // parent.grid.grid_width
        gy = (y - parent.grid.grid_y) // parent.grid.grid_height
        self.time = 0
        return peashooter(x, y, parent, self.tag + "_{}_{}".format(gx, gy))

    def update(self):
        if self.time!=self.freeze_time:
            self.time+=1
            card_image = self.p_card_image.copy()
            card_image_ = card_image.convert('L')
            x, y = card_image_.size  # 获得长和宽
            # 设置每个像素点颜色的透明度
            for i in range(x):
                for k in range(int(self.time*0.01*51), y):
                    color = card_image_.getpixel((i, k))
                    color = (color, color, color)
                    card_image.putpixel((i, k), color)
            self.card_image = ImageTk.PhotoImage(card_image)
            self.Canvas.itemconfigure(self.tag, image=self.card_image)



class peashooter_card(card):
    def __init__(self, Canvas, x, y, tag):
        super(peashooter_card, self).__init__(Canvas, x, y, tag)
        self.show()


class sunflower_card(card):
    def __init__(self, Canvas, x, y, tag):
        super(sunflower_card, self).__init__(Canvas, x, y, tag)
        self.cost = 25
        self.show()

    def show(self):
        self.p_card_image = Image.open(
            "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Cards/card_sunflower.png").resize(
            (37, 51))
        self.card_image = ImageTk.PhotoImage(self.p_card_image)
        self.image_plant = Image.open(
            "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Plants/SunFlower/SunFlower_0.png").resize(
            (50, 50))
        x0, y0 = self.image_plant.size  # 获得长和宽
        # 设置每个像素点颜色的透明度
        for i in range(x0):
            for k in range(y0):
                color = self.image_plant.getpixel((i, k))
                color = color[:-1] + (color[-1] // 2,)
                self.image_plant.putpixel((i, k), color)
        self.image_plant = ImageTk.PhotoImage(self.image_plant)
        self.Canvas.create_image(self.x, self.y, anchor='nw', image=self.card_image, tag=(self.tag))

    def create_plant(self, x, y, parent):
        gx = (x - parent.grid.grid_x) // parent.grid.grid_width
        gy = (y - parent.grid.grid_y) // parent.grid.grid_height
        self.time = 0
        return sunflower(x, y, parent, self.tag + "_{}_{}".format(gx, gy))
