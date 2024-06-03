import torch
import random
import gym
class DQN(torch.nn.Module):
    def __init__(self,statesize,actionsize):
        super(DQN, self).__init__()
        self.fc1 = torch.nn.Linear(statesize,16)
        self.bn1 = torch.nn.BatchNorm1d(1)
        self.relu1 = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(16,32)
        self.bn2 = torch.nn.BatchNorm1d(1)
        self.relu2 = torch.nn.LeakyReLU()
        self.fc3 = torch.nn.Linear(32,actionsize)

    def forward(self,x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

    def choose(self,state,greedy=0.9):
        if random.randint(0,10)*0.1<greedy and state in self.tabel.index:
            action = int(self.tabel.loc[state].idxmax())
            return int(self.action[action])
        else:
            return random.randint(0,3)

def train(decay=0.99,episodes=5000,greedy=0.5):
    env = gym.make('FrozenLake-v1')
    statesize = env.observation_space.n
    actionsize = env.action_space.n

    net = DQN(statesize,actionsize)
    net.load_state_dict(torch.load("DQN.ckpt"))

    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(net.parameters())



    for j in range(2):
        for i in range(episodes):
            state,_ = env.reset()
            Allr = 0
            for k in range(100):
                with torch.no_grad():
                    Q = net(torch.nn.functional.one_hot(torch.tensor(state).reshape(1,1),statesize).float())
                    if random.randint(1, 10) * 0.1 > greedy:
                        action = env.action_space.sample()
                    else:
                        action = torch.argmax(Q).item()

                    state_n, r, d, _, _ = env.step(action)
                    # if d ==True and r==0:
                    #     r=-10
                    # else:
                    #     r=r*30/(k+1)
                    Q_n = net(torch.nn.functional.one_hot(torch.tensor(state_n).reshape(1,1),statesize).float())
                    targetQ = Q
                    targetQ[0,0,action] = r + decay*torch.max(Q_n)


                tQ = net(torch.nn.functional.one_hot(torch.tensor(state).reshape(1,1),statesize).float())

                loss = loss_fn(tQ,targetQ)

                opt.zero_grad()
                loss.backward()
                opt.step()
                Allr += r
                state = state_n

                if d==True:
                    break
            torch.save(net.state_dict(), "DQN.ckpt")
            # if Allr>0:
            #     print(Allr)
            #     print(k)
            #     print("-------------------------")
        greedy = 0.9

def test():
    env = gym.make('FrozenLake-v1',render_mode="human")
    statesize = env.observation_space.n
    actionsize = env.action_space.n

    net = DQN(statesize,actionsize)

    net.load_state_dict(torch.load("DQN.ckpt"))
    for i in range(10):
        state, _ = env.reset()
        for k in range(100):
            with torch.no_grad():
                net.eval()
                Q = net(torch.nn.functional.one_hot(torch.tensor(state).reshape(1, 1), statesize).float())
                action = torch.argmax(Q).item()
                state_n, r, d, _, _ = env.step(action)
                state = state_n
            if d == True:
                print(r)
                print(k)
                break
if __name__ == '__main__':
    # train()
    test()


