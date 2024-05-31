import random
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import configparser
import time
from plant import sun
from zombie import zombie
from card import card, sunflower_card, peashooter_card
from utile import shovel, car


class popup:
    def __init__(self, frame):
        self.frame = frame
        self.top = tk.Toplevel(self.frame)
        self.top.title("Popup")
        self.x, self.y = 500, 500

        self.top.geometry('100x100+100+1300')

        self.top.resizable(False, False)

        self.top.attributes('-topmost', 1)

        # self.top.overrideredirect(True)

        # 窗口移动事件
        self.top.bind("<B1-Motion>", self.move_window)
        # 单击事件
        self.top.bind("<Button-1>", self.get_point)
        # 双击事件
        self.top.bind("<Double-Button-1>", self.close)

        self.top.protocol("WM_DELETE_WINDOW", self.null_action)
        # self.top.protocol("WM_TAKE_FOCUS", self.callback)
        # self.top.bind("<FocusOut>", self.callback)

        self.first_back = ttk.Button(self.top, text="第一关", command=self.enter)
        self.btn_back = ttk.Button(self.top, text="返回", command=self.close_exit)
        self.first_back.pack(side="top", padx=5)
        self.btn_back.pack(side="bottom", padx=5)

    def show(self):
        self.top.deiconify()  # 显示弹出窗口
        self.frame.master.state('withdrawn')
        self.top.wait_window()

    def move_window(self, event):
        """窗口移动事件"""
        new_x = (event.x - self.x) + self.top.winfo_x()
        new_y = (event.y - self.y) + self.top.winfo_y() - 35
        s = f"100x100+{new_x}+{new_y}"
        self.top.geometry(s)

    def get_point(self, event):
        """获取当前窗口位置并保存"""
        self.x, self.y = event.x, event.y
        self.top.deiconify()  # 显示弹出窗口

    def close(self, event):
        self.frame.master.state('normal')
        self.top.destroy()

    def close_exit(self):
        self.frame.master.state('normal')
        self.top.destroy()

    def enter(self):
        """进入第一关"""
        self.frame.master.state('normal')
        self.top.destroy()
        master = self.frame.master
        self.frame.destroy()
        self.frame.pack_forget()
        battle(master)

    def null_action(self):
        """无操作，用于防止关闭窗口"""
        pass

    # def callback(self,e):
    #     print(1)
    #     self.top.state('normal')
    #     self.top.attributes('-topmost',1)


class main_screen:
    def __init__(self, window):
        self.window = window

        self.frame = ttk.Frame(self.window)
        self.frame.pack()

        self.bg_img = None
        self.bg_image = None
        self.btn_img = None
        self.btn_image = None
        self.Canvas = tk.Canvas(self.frame, width=560, height=373, highlightthickness=0, bg="white")
        self.load_image()

        self.Canvas.bind("<ButtonPress>", self.press)
        self.Canvas.bind("<ButtonRelease>", self.btn_event)

        self.Canvas.pack()

        self.choose = 0

    def btn_event(self, e):
        self.btn_img = Image.open(
            "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Screen/Adventure_0.png")
        self.btn_img = self.btn_img.resize((int(self.btn_img.size[0] * 1.3), int(self.btn_img.size[1] * 1.3)),
                                           Image.LANCZOS)
        self.btn_image = ImageTk.PhotoImage(self.btn_img)
        self.Canvas.create_image(290, 50, anchor='nw', image=self.btn_image, tag=("button"))
        if 290 < e.x < 290 + int(self.btn_img.size[0] * 1.3) - 60 and 50 < e.y < 25 + int(
                self.btn_img.size[1] * 1.3) - (0 if e.x - 290 > 100 else 30 - (e.x - 290) * 0.15) and self.choose == 1:
            popup_ = popup(self.frame)
            popup_.show()
        self.choose = 0

    def load_image(self):
        self.bg_img = Image.open(
            "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Screen/MainMenu.png")
        self.bg_img.thumbnail((560, 560), Image.LANCZOS)
        self.bg_image = ImageTk.PhotoImage(self.bg_img)
        self.Canvas.create_image(0, 0, anchor='nw', image=self.bg_image, tag=("bg"))

        self.btn_img = Image.open(
            "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Screen/Adventure_0.png")
        self.btn_img = self.btn_img.resize((int(self.btn_img.size[0] * 1.3), int(self.btn_img.size[1] * 1.3)),
                                           Image.LANCZOS)
        self.btn_image = ImageTk.PhotoImage(self.btn_img)
        self.Canvas.create_image(290, 50, anchor='nw', image=self.btn_image, tag=("button"))

    def press(self, e):
        if 290 < e.x < 290 + int(self.btn_img.size[0] * 1.3) - 60 and 50 < e.y < (
                25 + int(self.btn_img.size[1] * 1.3) - (0 if e.x - 290 > 100 else 30 - (e.x - 290) * 0.15)):
            self.btn_img = Image.open(
                "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Screen/Adventure_1.png")
            self.btn_img = self.btn_img.resize((int(self.btn_img.size[0] * 1.3), int(self.btn_img.size[1] * 1.3)),
                                               Image.LANCZOS)
            self.btn_image = ImageTk.PhotoImage(self.btn_img)
            self.Canvas.create_image(290, 50, anchor='nw', image=self.btn_image, tag=("button"))
            self.choose = 1

    def undo(self, e):
        pass


class choosetable:
    def __init__(self, battle):
        self.Canvas = battle.Canvas
        self.map = ImageTk.PhotoImage(Image.open(
            "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Screen/PanelBackground.png").resize(
            (326, 359))
                                      )
        self.Canvas.create_image(0, -359, anchor='nw', image=self.map, tag=("choosetable"))
        # self.Canvas.tag_raise("plantbar")
        self.id = self.Canvas.after(1, self.appear, [0, -359], [0, 61], [0, 2])

        self.grid_x = 15
        self.grid_y = 90
        self.grid_width = 37
        self.grid_height = 51

        self.grid_size = (8, 5)

        self.plant_list = []

        self.grid = [(self.grid_x + self.grid_width * i, self.grid_y + self.grid_height * j) for j in
                     range(self.grid_size[1]) for i in range(self.grid_size[0])]

        self.btn_img = None

    def appear(self, start, end, speed):
        if self.Canvas.itemcget("choosetable", 'state') == 'hidden':
            self.Canvas.itemconfig("choosetable", state='normal')

        if start != end:
            start[0] += speed[0]
            start[1] += speed[1]
            self.Canvas.moveto("choosetable", start[0], start[1])
            self.Canvas.update()
            self.Canvas.after(1, self.appear, start, end, speed)
        else:
            self.Canvas.after_cancel(self.id)
            for i in range(10):
                self.plant_list.append(peashooter_card(self.Canvas, self.grid[i][0], self.grid[i][1], "card%d" % i))
            self.plant_list.append(sunflower_card(self.Canvas, self.grid[10][0], self.grid[10][1], "card%d" % 10))
            if self.btn_img == None:
                self.btn_img = Image.open(
                    "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Screen/StartButton.png")
                self.btn_img = self.btn_img.resize((int(self.btn_img.size[0] * 0.7), int(self.btn_img.size[1] * 0.7)),
                                                   Image.LANCZOS)
                self.btn_image = ImageTk.PhotoImage(self.btn_img)
                self.Canvas.create_image(109, 383, anchor='nw', image=self.btn_image, tag=("start_button"))
            else:
                if self.Canvas.itemcget("start_button", 'state') == 'hidden':
                    self.Canvas.itemconfig("start_button", state='normal')

            for i in range(len(self.plant_list)):
                if self.Canvas.itemcget(self.plant_list[i].tag, 'state') == 'hidden':
                    self.Canvas.itemconfig(self.plant_list[i].tag, state='normal')

    def click(self, e):
        gx = (e.x - self.grid_x) // self.grid_width
        gy = (e.y - self.grid_y) // self.grid_height
        if 0 <= gx < self.grid_size[0] and 0 <= gy < self.grid_size[1] and gy * self.grid_size[0] + gx < len(
                self.plant_list):
            card = self.plant_list.pop(gy * self.grid_size[0] + gx)
            for i in range(len(self.plant_list)):
                self.plant_list[i].move(self.grid[i][0], self.grid[i][1])
            return card
        return None

    def click_(self, e):
        if 109 <= e.x <= 218 and 383 <= e.y <= 410:
            self.Canvas.itemconfig("start_button", state='hidden')
            self.Canvas.itemconfig("choosetable", state='hidden')
            for i in range(len(self.plant_list)):
                self.Canvas.itemconfig(self.plant_list[i].tag, state='hidden')
            # self.Canvas.itemconfig("plantbar", state='hidden')
            return 1
        return 0

    def add(self, card):
        self.plant_list.append(card)
        for i in range(len(self.plant_list)):
            self.plant_list[i].move(self.grid[i][0], self.grid[i][1])

    def re_appear(self):
        del self.id
        self.id = None
        self.id = self.Canvas.after(1, self.appear, [0, -359], [0, 61], [0, 2])


class plantbar:
    def __init__(self, battle):
        self.battle = battle
        self.Canvas = battle.Canvas
        self.map = ImageTk.PhotoImage(Image.open(
            "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Screen/ChooserBackground.png").resize(
            (365, 61))
                                      )
        self.Canvas.create_image(0, 0, anchor='nw', image=self.map, tag=("plantbar"))

        self.plant_que = []

        self.maxlen = 8

        self.grid_y = 5
        self.grid_x = 55
        self.grid_width = 38
        self.grid_height = 51

        self.grid_size = (8, 5)

        self.grid = [(self.grid_x + self.grid_width * i, self.grid_y + self.grid_height * j) for j in
                     range(self.grid_size[1]) for i in range(self.grid_size[0])]

    def add(self, card):
        if len(self.plant_que) < self.maxlen:
            card.move(self.grid[len(self.plant_que)][0], self.grid[len(self.plant_que)][1])
            self.plant_que.append(card)

    def click(self, e):
        gx = (e.x - self.grid_x) // self.grid_width
        gy = (e.y - self.grid_y) // self.grid_height
        if 0 <= gx < len(self.plant_que) and 0 == gy and len(self.plant_que) > 0:
            card = self.plant_que.pop(gx)
            for i in range(gx, len(self.plant_que)):
                self.plant_que[i].move(self.grid[i][0], self.grid[i][1])
            return card
        return None

    def click_(self, e):
        gx = (e.x - self.grid_x) // self.grid_width
        gy = (e.y - self.grid_y) // self.grid_height
        if gx < len(self.plant_que) and 0 == gy and len(self.plant_que) > 0:
            if self.plant_que[gx].time == self.plant_que[gx].freeze_time:
                return self.plant_que[gx]

    def get_num(self):
        return len(self.plant_que)


class map_grid:
    def __init__(self, parent):
        self.grid_y = 58
        self.grid_x = 24
        self.grid_width = 56
        self.grid_height = 70

        self.grid_size = (9, 5)

        self.grid = [(self.grid_x + self.grid_width * i, self.grid_y + self.grid_height * j) for j in
                     range(self.grid_size[1]) for i in range(self.grid_size[0])]

        self.plant_grid = {}
        self.zombie_grid = []
        self.bullet_grid = []
        self.head_grid = []

        self.sun_grid = []

        self.car_grid = []

        self.value = 50

        self.parent = parent

        self.sun_index = 0

    def get_grid(self, e):
        gx = (e.x - self.grid_x) // self.grid_width
        gy = (e.y - self.grid_y) // self.grid_height
        return (gx, gy)

    def add_plant(self, plant, grid):
        if grid not in self.plant_grid:
            self.plant_grid[grid] = plant

    def update(self, index):
        if index % 30 == 29:
            for i in range(len(self.plant_grid) - 1, -1, -1):
                if self.plant_grid[list(self.plant_grid.keys())[i]].healthy == 0:
                    self.plant_grid.pop(list(self.plant_grid.keys())[i]).delete()
                else:
                    self.plant_grid[list(self.plant_grid.keys())[i]].attack(self.zombie_grid)
                    self.plant_grid[list(self.plant_grid.keys())[i]].update()
                    self.plant_grid[list(self.plant_grid.keys())[i]].show()
        if index % 20 == 19:
            for i in range(len(self.zombie_grid) - 1, -1, -1):
                if self.zombie_grid[i].state == 1 and self.zombie_grid[i].frame_index == (
                        len(self.zombie_grid[i].frame) - 1):
                    self.zombie_grid.pop(i).delete()
                else:
                    self.zombie_grid[i].attack(self.plant_grid.values())
                    self.zombie_grid[i].update()
                    self.zombie_grid[i].show()

        if index % 10 == 9:
            for i in range(len(self.head_grid) - 1, -1, -1):
                if self.head_grid[i].frame_index == (len(self.head_grid[i].frame) - 1):
                    self.head_grid[i].delete()
                    self.head_grid.pop(i)
                else:
                    self.head_grid[i].update()
                    self.head_grid[i].show()

        for i in range(len(self.bullet_grid) - 1, -1, -1):
            if self.bullet_grid[i].x > 550 or self.bullet_grid[i].state == 1:
                self.bullet_grid[i].delete()
                self.bullet_grid.pop(i)
            else:
                self.bullet_grid[i].update()
                self.bullet_grid[i].attack(self.zombie_grid)
                self.bullet_grid[i].show()

        if index % 2 ==0:
            for i in range(len(self.sun_grid) - 1, -1, -1):
                self.sun_grid[i].update()
                self.sun_grid[i].show()
                if self.sun_grid[i].state == 4:
                    self.sun_grid.pop(i).delete()
            for i in range(len(self.car_grid) - 1, -1, -1):
                if self.car_grid[i].state != 2:
                    self.car_grid[i].update()
                    self.car_grid[i].attack(self.zombie_grid)
                else:
                    self.car_grid[i].delete()
                    self.car_grid.pop(i)

        if index % 1000 == 499:
            self.sun_grid.append(sun(0, 0, self.parent, "sun_{}".format(self.sun_index)))
            self.sun_index += 1
            self.sun_index %= 10

        self.parent.Canvas.itemconfigure(self.parent.text, text="{}".format(self.value))

    def click(self, e):
        for i in range(len(self.sun_grid) - 1, -1, -1):
            if self.sun_grid[i].x < e.x < self.sun_grid[i].x + 40 and self.sun_grid[i].y < e.y < self.sun_grid[
                i].y + 40:
                self.sun_grid[i].click()


class battle:
    def __init__(self, window):
        self.window = window

        self.frame = ttk.Frame(self.window)
        self.frame.pack()

        self.Canvas = tk.Canvas(self.frame, width=560, height=420, highlightthickness=0, bg="white")
        self.Canvas.pack()
        self.screen_x = -154

        self.map = ImageTk.PhotoImage(Image.open(
            "/home/qiu/project/pythonplantsvszombies-master/resources/graphics/Items/Background/Background_0.jpg").resize(
            (980, 420))
                                      )
        self.Canvas.create_image(-420, 0, anchor='nw', image=self.map, tag=("map"))
        self.bar = plantbar(self)
        self.table = choosetable(self)

        self.Canvas.bind("<ButtonPress>", self.press)
        self.Canvas.bind("<ButtonRelease>", self.release)
        self.Canvas.bind("<Motion>", self.mouse_move)
        self.state = 0

        self.id = None

        self.choose = None

        self.grid = map_grid(self)

        self.text = None

        self.shovel = None

    def press(self, e):
        print(e)
        if self.state == 0:
            if self.table.click_(e):
                self.state = 1
                self.id = self.Canvas.after(1, self.screen_move, [-420, 0], [self.screen_x, 0], [2, 0])
            else:
                card_ = self.table.click(e)
                if self.bar.get_num() < self.bar.maxlen and card_ is not None:
                    self.bar.add(card_)
                else:
                    card_ = self.bar.click(e)
                    if card_ is not None:
                        self.table.add(card_)

        else:
            self.grid.click(e)

            card_ = self.bar.click_(e)
            if card_ is not None:
                if self.choose is not None:
                    self.choose.common()
                self.choose = card_
                self.window.config(cursor="none")
                return
            if self.shovel.x <= e.x <= self.shovel.x + 56 and self.shovel.y <= e.y <= self.shovel.y + 56:
                if self.choose is not None:
                    self.choose.common()
                self.window.config(cursor="none")
                self.choose = self.shovel
                return

            if self.choose is not None:
                grid = self.grid.get_grid(e)
                if isinstance(self.choose, card):
                    if 0 <= grid[0] < self.grid.grid_size[0] and 0 <= grid[1] < self.grid.grid_size[
                        1] and self.grid.value >= self.choose.cost:
                        self.grid.add_plant(self.choose.create_plant(self.grid.grid_x + grid[0] * self.grid.grid_width,
                                                                     self.grid.grid_y + grid[1] * self.grid.grid_height,
                                                                     self), grid)
                        self.grid.value -= self.choose.cost
                elif isinstance(self.choose, shovel):
                    self.shovel.delete_plant(grid[0], grid[1])
                self.choose.common()
                self.choose = None
                self.window.config(cursor="arrow")

    def mouse_move(self, event):
        if self.choose is not None:
            self.choose.be_chose(event)

    def release(self, e):
        pass

    def screen_move(self, start, end, speed):
        if start != end:
            start[0] += speed[0]
            start[1] += speed[1]
            self.Canvas.moveto("map", start[0], start[1])
            self.Canvas.update()
            self.Canvas.after(1, self.screen_move, start, end, speed)
        else:
            self.Canvas.after_cancel(self.id)
            del self.id
            self.id = None
            self.id = self.Canvas.after(1, self.update, 0)
            self.text = self.Canvas.create_text(40, 50, text="cost")

            self.shovel = shovel(380, 0, self)

            for i in range(self.grid.grid_size[1]):
                self.grid.car_grid.append(
                    car(-25, self.grid.grid_y + 10 + self.grid.grid_height * i, "car_%d" % i, self))

    def update(self, index):
        self.grid.update(index)
        index += 1
        index %= 1000
        self.Canvas.after(10, self.update, index)

        if index%1000==9:
            for i in range(len(self.grid.zombie_grid),-1,-1):
                if not self.Canvas.find_withtag("zombie_%d"%i):
                    print(1)
                    self.grid.zombie_grid.append(zombie(550, random.randint(0,self.grid.grid_size[1]-1)*(self.grid.grid_width+10), self, "zombie_%d"%i))
                    break
        if index%10 ==9:
            for i in self.bar.plant_que:
                i.update()

class game:
    def __init__(self):
        self.level = 0

        self.window = tk.Tk()
        self.window.geometry("560x420+100+1300")

        main_screen(self.window)

    def run(self):
        self.window.mainloop()


a = game()
a.run()
