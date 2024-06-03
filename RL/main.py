# https://github.com/Lizhi-sjtu/DRL-code-pytorch
import queue
import numpy as np
import pandas as pd
import random
import gym
class Qtabel:
    def __init__(self,action):
        self.tabel = pd.DataFrame(columns=action)
        self.actionsize = len(action)
        self.action = {}
        for i in range(self.actionsize):
            self.action[i]=str(action[i])

    def updata(self,state,action,state_n,r,dacay=0.99,lr=0.85):
        action = str(action)
        if state not in list(self.tabel.index):
            self.tabel.loc[state] = [0] * self.actionsize
        if state_n not in list(self.tabel.index):
            self.tabel.loc[state_n] = [0] * self.actionsize
        if state_n == -1:
            self.tabel.loc[state,action] = self.tabel.loc[state,action] + lr*(r-self.tabel.loc[state,action])

        else:
            self.tabel.loc[state,action] = self.tabel.loc[state,action] + lr*(r+dacay*self.tabel.loc[state_n].max()-self.tabel.loc[state,action])

    def choose(self,state,greedy=0.9):
        if random.randint(0,10)*0.1<greedy and state in self.tabel.index:
            action = int(self.tabel.loc[state].idxmax())
            return int(self.action[action])
        else:
            return random.randint(0,3)

    def saveTable(self):
        self.tabel.to_csv("./table.csv",index=True)

    def loadTable(self):
        self.tabel = pd.read_csv("./table.csv", index_col=0)

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1',render_mode="human")
    render = False

    Qlearning = Qtabel(range(4))

    episodes = 1000
    greedy = 2

    Qlearning.loadTable()
    for i in range(episodes):

        state,_ = env.reset()
        rAll = 0

        for j in range(100):
            action = Qlearning.choose(state,greedy)
            state_n,r,d,_,_=env.step(action)

            rAll+=r
            if d==True:
                state_n=-1
                Qlearning.updata(state, action, state_n, r)
                break
            else:
                Qlearning.updata(state, action, state_n, r)

            state = state_n

        if i%100 ==0:
            greedy += 0.1
        if rAll>0:
            print(rAll)
            print(greedy)
            print(i)
            print(j)
            print("---------------------------------")
        # print(Qlearning.tabel)
        # Qlearning.saveTable()



