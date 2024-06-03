import random

import torch
import gym
from torch.utils.data.sampler import BatchSampler,SubsetRandomSampler
import torch.nn.functional as F
import copy
class Actor(torch.nn.Module):
    def __init__(self,statesize,actionsize,orthogonal=True,max_action=2):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(statesize,32)
        self.relu1 = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(32,64)
        self.relu2 = torch.nn.LeakyReLU()
        self.fc6 = torch.nn.Linear(64,actionsize)
        self.tanh = torch.nn.Tanh()

        self.max_action = max_action

        if orthogonal:
            self.orthogonal_init(self.fc1)
            self.orthogonal_init(self.fc2)
            self.orthogonal_init(self.fc6,0.01)

    def orthogonal_init(self,layer,gain=1.0):
        torch.nn.init.orthogonal_(layer.weight,gain=gain)
        torch.nn.init.constant_(layer.bias,0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc6(x)
        x = self.tanh(x)
        return x*self.max_action

class Critic(torch.nn.Module):
    def __init__(self,statesize,actionsize,orthogonal=True):
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(statesize+actionsize,32)
        self.relu1 = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(32,64)
        self.relu2 = torch.nn.LeakyReLU()
        self.fc6 = torch.nn.Linear(64,1)
        if orthogonal:
            self.orthogonal_init(self.fc1)
            self.orthogonal_init(self.fc2)
            self.orthogonal_init(self.fc6)

    def orthogonal_init(self, layer, gain=1.0):
        torch.nn.init.orthogonal_(layer.weight, gain=gain)
        torch.nn.init.constant_(layer.bias, 0)

    def forward(self,state,action):
        x = torch.cat([state,action],1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc6(x)
        return x

class Memery:
    def __init__(self,statesize,actionsize = 1,size=2048):
        self.state = torch.zeros((size,statesize))
        self.action = torch.zeros((size,actionsize))
        self.r = torch.zeros((size,1))
        self.state_n = torch.zeros((size,statesize))
        self.d = torch.zeros((size,1))
        self.index = 0

    def append(self,state,action,r,state_n,d):
        self.state[self.index,:] = torch.tensor(state)
        self.action[self.index, :] = torch.tensor(action)
        self.r[self.index,:] = torch.tensor(r)
        self.state_n[self.index,:] = torch.tensor(state_n)
        self.d[self.index,:] = torch.tensor(d)
        self.index = (self.index+1)%2048

    def __len__(self):
        return self.index

    def __getitem__(self, item):
        return (self.state[item],self.action[item],self.logaction[item],self.r[item],self.state_n[item],self.d[item])

    def __call__(self, *args, **kwargs):
        return (self.state,self.action,self.r,self.state_n,self.d)

class Normalize:
    def __init__(self,shape,RewardScaling=False,gamma=0.9):
        self.mean = torch.zeros((1,shape))
        self.S = torch.zeros((1,shape))
        self.std = torch.zeros((1,shape))
        self.num = 0
        self.RewardScaling = RewardScaling
        self.gamma = gamma
        self.R = torch.zeros((1,shape))
        self.shape = shape

    def updata(self,x):
        x = torch.tensor(x)
        if self.RewardScaling:
            self.R = self.gamma*self.R+x
            x = self.R
        self.num += 1
        if self.num==1:
            self.mean = x
            self.std = x

        else:
            mean_ = self.mean
            self.mean = self.mean + (x-self.mean)/self.num
            self.S = self.S + (x-self.mean)*(x-mean_)
            self.std = torch.sqrt_(self.S/self.num)

    def __call__(self,x,updata=True):
        x = torch.tensor(x)
        if updata:
            self.updata(x)
        return (x-self.mean)/(self.std+1e-8)

    def reset(self):
        self.R = torch.zeros((1,self.shape)).reshape(-1)

def train(episodes=20000,decay=0.99,greedy=0.):
    env = gym.make('Hopper-v4')
    env = env.unwrapped
    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    actor = Actor(statesize,actionsize,True,max_action)
    actor_target = copy.deepcopy(actor)
    # actor.load_state_dict(torch.load("ppoa1.ckpt"))
    critic1 = Critic(statesize,True)
    critic_target1 = copy.deepcopy(critic1)
    critic2 = Critic(statesize,True)
    critic_target2 = copy.deepcopy(critic2)
    # critic.load_state_dict(torch.load("ppoc1.ckpt"))

    loss_fn = torch.nn.MSELoss()
    opt_a = torch.optim.Adam(actor.parameters(),lr=3e-4,eps=1e-5)
    opt_c1 = torch.optim.Adam(critic1.parameters(),lr=3e-4,eps=1e-5)
    opt_c2 = torch.optim.Adam(critic2.parameters(),lr=3e-4,eps=1e-5)

    # state_nomal = Normalize(statesize)
    # reward_normal = Normalize(1,True)

    maxr = 0
    maxloss = 10
    memery = Memery(statesize,actionsize)
    for i in range(episodes):
        state,_ = env.reset()

        # state = state_nomal(state)
        # reward_normal.reset()
        allr = 0
        d = False
        gaussian = 1
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
        while d==False:
            if random.randint(0,10) < 3:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = actor(state).data.numpy().flatten()

            state_n, r, d, _, _ = env.step(action)
            # state_n = state_nomal(state_n)
            # R = reward_normal(r)
            r = (r+8)/8
            state_n = torch.unsqueeze(torch.tensor(state_n, dtype=torch.float), 0)
            memery.append(state,action,r,state_n,d)

            state = state_n

            # allr +=r

            if len(memery) == 2048:
                states,actions,rs,state_ns,ds = memery()
                for _ in range(10):
                    for index in BatchSampler(SubsetRandomSampler(range(2048)),64,False):
                        with torch.no_grad():
                            target_action = actor_target(state_ns[index])
                            Q_1 = critic_target1(state_ns[index],target_action)
                            Q_2 = critic_target2(state_ns[index],target_action)
                            target_Q = rs[index] + decay * (1 - ds[index]) * torch.min(Q_1,Q_2)
                        Q1 = critic1(states[index],action[index])
                        Q2 = critic2(states[index], action[index])
                        c_loss = loss_fn(Q1,target_Q)+loss_fn(Q2,target_Q)
                        opt_c1.zero_grad()
                        opt_c2.zero_grad()
                        c_loss.backward()
                        opt_c1.step()
                        opt_c2.step()

                        for params in critic1.parameters():
                            params.requires_grad = False

                        a_loss = -critic1(states[index],actor(state[index])).mean()
                        opt_a.zero_grad()
                        a_loss.backward()
                        opt_a.step()

                        for params in critic1.parameters():
                            params.requires_grad=True

                        for param,target_param in zip(critic1.parameters(),critic_target1.parameters()):
                            target_param.data.copy_(0.005*param.data+(1-0.005)*target_param)
                        for param,target_param in zip(critic2.parameters(),critic_target2.parameters()):
                            target_param.data.copy_(0.005*param.data+(1-0.005)*target_param)
                        for param,target_param in zip(actor.parameters(),actor_target.parameters()):
                            target_param.data.copy_(0.005*param.data+(1-0.005)*target_param)
                    torch.save(actor.state_dict(), "ppoca1.ckpt")
                    torch.save(critic1.state_dict(), "ppocc1.ckpt")
                    torch.save(critic2.state_dict(), "ppocc1.ckpt")


                lr_a_now = 3e-4 * (1 - i / episodes)
                lr_c_now = 3e-4 * (1 - i / episodes)
                for p in opt_a.param_groups:
                    p['lr'] = lr_a_now
                for p in opt_c1.param_groups:
                    p['lr'] = lr_c_now
                for p in opt_c2.param_groups:
                    p['lr'] = lr_c_now

            # if allr>maxr and i>10000:
            #     torch.save(actor.state_dict(), "ppoa.ckpt")
            #     torch.save(critic.state_dict(), "ppoc.ckpt")
            #     maxr = allr

            # print(i,r)
def test():
    env = gym.make('Hopper-v4',render_mode="human")
    env = env.unwrapped
    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.shape[0]
    net = Actor(statesize,actionsize)

    # net.load_state_dict(torch.load("ppoca1.ckpt"))

    for i in range(1):
        state, _ = env.reset()
        # for k in range(100):
        allr=0
        while(True):
            with torch.no_grad():
                net.eval()
                Q = net(torch.tensor(state).reshape(1, -1).float())
                # action = torch.argmax(Q).item()
                action =Q.detach().numpy().flatten()
                # print(action)
                # action = torch.nn.functional.softmax(Q[0], dim=0).multinomial(num_samples=1, replacement=False).item()
                state_n, r, d, _, _ = env.step(action)
                state = state_n
                print(Q)
                # print(action)
                allr+=r
            if d == True:
                print(allr)
                # print(k)
                break

if __name__ == '__main__':
    # train()
    test()
