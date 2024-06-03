import torch
import random
import gym
from collections import deque
class DQN(torch.nn.Module):
    def __init__(self,statesize,actionsize):
        super(DQN, self).__init__()
        self.fc1 = torch.nn.Linear(statesize,8)
        self.relu1 = torch.nn.LeakyReLU()
        # self.dp1 = torch.nn.Dropout()
        self.fc2 = torch.nn.Linear(8,16)
        self.relu2 = torch.nn.LeakyReLU()
        # self.dp2 = torch.nn.Dropout()
        # self.fc3 = torch.nn.Linear(16,32)
        # self.relu3 = torch.nn.LeakyReLU()
        # self.fc4 = torch.nn.Linear(32,16)
        # self.relu4 = torch.nn.LeakyReLU()
        # self.fc5 = torch.nn.Linear(16,8)
        # self.relu5 = torch.nn.LeakyReLU()
        self.fc6 = torch.nn.Linear(16,actionsize)

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        # x = self.fc3(x)
        # x = self.relu3(x)
        # x = self.fc4(x)
        # x = self.relu4(x)
        # x = self.fc5(x)
        # x = self.relu5(x)
        # x = self.dp1(x)
        # x = self.fc2(x)
        # x = self.relu2(x)
        # x = self.dp2(x)
        x = self.fc6(x)
        return x

def train(episodes=5000,decay=0.9,greedy=0.):
    env = gym.make('CartPole-v1')
    env = env.unwrapped
    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.n

    net = DQN(statesize,actionsize)
    # net.load_state_dict(torch.load("DDQN.ckpt"))
    targetnet = DQN(statesize,actionsize)
    # targetnet.load_state_dict(torch.load("DDQN.ckpt"))

    memory = deque(maxlen=2000)
    batch=128

    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(net.parameters(),lr=1e-3)

    maxr = 0

    for i in range(episodes):
        state,_ = env.reset()
        allr = 0

        mm = []
        for j in range(10000):
            with torch.no_grad():
                Q = net(torch.tensor(state).reshape(1, -1).float())
                if random.randint(1, 10) * 0.1 > greedy:
                    action = env.action_space.sample()
                else:
                    action = torch.argmax(Q).item()

            state_n, r, d, _, _ = env.step(action)

            mm.append((state,action,state_n,r,d))

            state = state_n

            allr +=r

            if d:
                if allr>0:
                    print(allr)
                    print(i)
                    if len(memory) > batch:
                        print(loss)
                    print("---------------------------")
                break
        if allr>20:
            for m in mm:
                memory.append(m)
        if len(memory)>batch:
            data = random.sample(memory,batch)

            for s,a,s_,r_,d_ in data:
                with torch.no_grad():
                    Q_ = net(torch.tensor(s).reshape(1, -1).float())
                    Q_n = net(torch.tensor(s_).reshape(1, -1).float())
                    targetQ_n = targetnet(torch.tensor(s_).reshape(1, -1).float())
                    targetQ = Q_
                    if d_:
                        targetQ[0,a] = r_
                    else:
                        targetQ[ 0, a] = r_ + decay * targetQ_n[0,torch.argmax(Q_n)]
                targetQ = torch.nn.functional.softmax(targetQ,dim=1)
                TQ = net(torch.tensor(s).reshape(1, -1).float())
                loss = loss_fn(TQ,targetQ)
                opt.zero_grad()
                loss.backward()
                opt.step()
        if i>2500 and allr>20 and len(memory)>batch:
            if greedy<1:
                greedy += 0.001
        if i%100==0:
            targetnet.load_state_dict(net.state_dict())
        # if allr>30:
        #     print("save:",allr)
        if i>1500:
            if allr>maxr:
                torch.save(net.state_dict(), "DDQN.ckpt")
                maxr = allr
        # else:
        #     torch.save(net.state_dict(), "DDQN1.ckpt")
def test():
    env = gym.make('CartPole-v1',render_mode="human")
    env = env.unwrapped
    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.n

    net = DQN(statesize,actionsize)

    net.load_state_dict(torch.load("DDQN.ckpt"))
    for i in range(2):
        state, _ = env.reset()
        # for k in range(100):
        allr=0
        while(True):
            with torch.no_grad():
                net.eval()
                Q = net(torch.tensor(state).reshape(1, -1).float())
                action = torch.argmax(Q).item()
                state_n, r, d, _, _ = env.step(action)
                state = state_n
                print(Q)
                print(action)
                allr+=r
            if d == True:
                print(allr)
                # print(k)
                break

if __name__ == '__main__':
    # train()
    test()
