from PIL import ImageTk, Image


class shovel:
    def __init__(self, x, y, map):
        self.x = x
        self.y = y
        self.map = map
        self.type = "shovel"
        self.tag = "shovel"
        self.image = ImageTk.PhotoImage(Image.open(
            "../shovel.png").resize((56, 56))
                                        )
        self.map.Canvas.create_image(self.x, self.y, anchor='nw', image=self.image, tag=(self.tag))

    def delete_plant(self, x, y):
        if (x, y) in self.map.grid.plant_grid.keys():
            self.map.grid.plant_grid[(x, y)].shell()
            self.map.grid.plant_grid.pop((x, y)).delete()

    def be_chose(self, e):
        self.map.Canvas.moveto(self.tag, e.x - 15, e.y - 15)

    def common(self):
        self.map.Canvas.moveto(self.tag, self.x, self.y)


class car:
    def __init__(self, x, y, tag, map):
        self.x = x
        self.y = y
        self.width = 50
        self.type = "car"
        self.tag = tag
        self.map = map
        self.car = ImageTk.PhotoImage(Image.open(
            "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Screen/car.png").resize((50, 40))
                                      )
        self.speed = 10
        self.state = 0
        self.map.Canvas.create_image(self.x, self.y, anchor='nw', image=self.car, tag=(self.tag))

        self.gx = (self.x - self.map.grid.grid_x) // self.map.grid.grid_width
        self.gy = (self.y - self.map.grid.grid_y) // self.map.grid.grid_height

    def update(self):
        if self.state == 1:
            self.x += self.speed
            self.map.Canvas.moveto(self.tag, self.x, self.y)
            if self.x >= 550:
                self.state = 2

    def attack(self, zombies):
        for zombie in zombies:
            if zombie.x + zombie.rect_r <= self.x + self.width <= zombie.x + zombie.width and zombie.healthy > 0 and self.state == 0 and self.gy == zombie.gy:
                self.state = 1
            if zombie.x + zombie.rect_r <= self.x <= zombie.x + zombie.width and zombie.healthy > 0 and self.state == 1 and self.gy == zombie.gy:
                zombie.attacked(zombie.healthy)
                zombie.show()

    def delete(self):
        self.map.Canvas.delete(self.tag)
