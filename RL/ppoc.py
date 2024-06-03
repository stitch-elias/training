import torch
import gym
from torch.utils.data.sampler import BatchSampler,SubsetRandomSampler
from torch.distributions import Categorical,Normal,Beta,kl_divergence
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

        self.std = torch.nn.Parameter(torch.zeros(1,actionsize))

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

    def get_dist(self,x):
        mean = self.forward(x)
        std = torch.exp(self.std.expand_as(mean))
        dist = Normal(mean,std)
        return dist


class ActorBeta(torch.nn.Module):
    def __init__(self, statesize, actionsize, orthogonal=True):
        super(ActorBeta, self).__init__()
        self.fc1 = torch.nn.Linear(statesize, 32)
        self.relu1 = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(32, 64)
        self.relu2 = torch.nn.LeakyReLU()
        self.alpha = torch.nn.Linear(64, actionsize)
        self.beta = torch.nn.Linear(64,actionsize)

        if orthogonal:
            self.orthogonal_init(self.fc1)
            self.orthogonal_init(self.fc2)
            self.orthogonal_init(self.alpha, 0.01)
            self.orthogonal_init(self.beta, 0.01)

    def orthogonal_init(self, layer, gain=1.0):
        torch.nn.init.orthogonal_(layer.weight, gain=gain)
        torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        alpha = F.softplus(self.alpha(x))
        beta = F.softplus(self.beta(x))
        return alpha,beta

    def get_dist(self, x):
        alpha,beta = self.forward(x)
        dist = Beta(alpha, beta)
        return dist

class Critic(torch.nn.Module):
    def __init__(self,statesize,orthogonal=True):
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(statesize,32)
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

    def forward(self,x):
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
        self.logaction = torch.zeros((size,actionsize))
        self.r = torch.zeros((size,1))
        self.state_n = torch.zeros((size,statesize))
        self.d = torch.zeros((size,1))
        self.index = 0

    def append(self,state,action,logaction,r,state_n,d):
        self.state[self.index,:] = torch.tensor(state)
        self.action[self.index, :] = torch.tensor(action)
        self.logaction[self.index,:] = torch.tensor(logaction)
        self.r[self.index,:] = torch.tensor(r)
        self.state_n[self.index,:] = torch.tensor(state_n)
        self.d[self.index,:] = torch.tensor(d)
        self.index += 1

    def __len__(self):
        return self.index

    def __getitem__(self, item):
        return (self.state[item],self.action[item],self.logaction[item],self.r[item],self.state_n[item],self.d[item])

    def __call__(self, *args, **kwargs):
        return (self.state,self.action,self.logaction,self.r,self.state_n,self.d)

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

def train(episodes=200000,decay=0.99,greedy=0.,ppo=0):
    env = gym.make('Pendulum-v1')
    env = env.unwrapped
    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    actor = Actor(statesize,actionsize,True,max_action)
    # actor.load_state_dict(torch.load("ppoca1.ckpt"))
    if ppo:
        actor_old = copy.deepcopy(actor)
        lam = 0.5
    critic = Critic(statesize,True)
    # critic.load_state_dict(torch.load("ppocc1.ckpt"))

    loss_fn = torch.nn.MSELoss()
    opt_a = torch.optim.Adam(actor.parameters(),lr=3e-4,eps=1e-5)
    opt_c = torch.optim.Adam(critic.parameters(),lr=3e-4,eps=1e-5)

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
        for jj in range(200):
            if i < 100000:
                action = torch.tensor(env.action_space.sample())
                dist1 = Normal(action, torch.tensor(2))
                a_logprob = dist1.log_prob(action)
                action,log_action = action.numpy().flatten(), a_logprob.numpy().flatten()
            else:
                with torch.no_grad():
                    dist1 = actor.get_dist(state)
                    a = dist1.sample()
                    if gaussian:
                        a = torch.clamp(a,-max_action,max_action)
                    a_logprob = dist1.log_prob(a)
                action,log_action = a.numpy().flatten(), a_logprob.numpy().flatten()
            state_n, r, d, _, _ = env.step(action)
            # state_n = state_nomal(state_n)
            R = (r+8)/8
            # R = reward_normal(R)
            if jj==200:
                d = True
            state_n = torch.unsqueeze(torch.tensor(state_n, dtype=torch.float), 0)
            memery.append(state,action,log_action,R,state_n,d)

            state = state_n

            # allr +=r

            if len(memery) == 2048:
                states,actions,log_actions,rs,state_ns,ds = memery()

                with torch.no_grad():
                    vs = critic(states.float())
                    vs_ = critic(state_ns.float())

                    td_error = rs + decay*(1-d)*vs_-vs

                    gae = 0
                    adv = []
                    for td,d_ in zip(reversed(td_error.flatten().numpy()),reversed(ds.flatten().numpy())):
                        gae = td + decay*0.95*gae*(1-d_)
                        adv.insert(0, gae)

                    adv = torch.tensor(adv,dtype=torch.float)

                    target_v = adv+vs
                    adv = (adv-torch.mean(adv))/(torch.std(adv)+1e-5)
                for _ in range(4):
                    for index in BatchSampler(SubsetRandomSampler(range(2048)),64,False):
                        dist = actor.get_dist(states[index].float())
                        dist_entropy = dist.entropy().sum(1,keepdim=True)
                        log_dist = dist.log_prob(actions[index])

                        ratios = torch.exp(log_dist.sum(1,keepdim=True)-log_actions[index].sum(1,keepdim=True))

                        surr1 = ratios*adv[index]

                        if ppo:
                            with torch.no_grad():
                                dist_ = actor_old.get_dist(states[index].float())
                                kl = torch.mean(kl_divergence(dist_,dist))
                            loss_a = -torch.mean(surr1-kl*lam)
                            if kl>4*0.01:
                                break
                        else:
                            surr2 = torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * adv[index]
                            loss_a = -torch.min(surr1, surr2) - dist_entropy * 0.01




                        opt_a.zero_grad()
                        loss_a.mean().backward()
                        torch.nn.utils.clip_grad_norm_(actor.parameters(),0.5)
                        opt_a.step()
                        v_p = critic(states[index].float())
                        loss_c = loss_fn(v_p,target_v[index])*0.5
                        # print(loss_a.mean())
                        # print(loss_c)

                        opt_c.zero_grad()
                        loss_c.backward()
                        torch.nn.utils.clip_grad_norm_(critic.parameters(),0.5)
                        opt_c.step()
                        if loss_c<maxloss:
                            maxloss=loss_c
                if ppo:
                    if kl < 0.01 / 1.5:
                        lam /= 2
                    elif kl > 0.01 * 1.5:
                        lam *= 2
                    if lam < 1e-4:
                        lam = 1e-4
                    elif lam > 10:
                        lam = 10
                    actor_old.load_state_dict(actor.state_dict())
                memery.index=0

                lr_a_now = 3e-4 * (1 - i / episodes)
                lr_c_now = 3e-4 * (1 - i / episodes)
                for p in opt_a.param_groups:
                    p['lr'] = lr_a_now
                for p in opt_c.param_groups:
                    p['lr'] = lr_c_now

            # if allr>maxr and i>10000:
            #     torch.save(actor.state_dict(), "ppoa.ckpt")
            #     torch.save(critic.state_dict(), "ppoc.ckpt")
            #     maxr = allr
            # if loss_c < (maxloss + 0.5):
                torch.save(actor.state_dict(), "ppoca1.ckpt")
                torch.save(critic.state_dict(), "ppocc1.ckpt")
            print(i,r)
def test():
    env = gym.make('Pendulum-v1',render_mode="human")
    env = env.unwrapped
    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    net = Actor(statesize,actionsize,max_action=max_action)
    net.load_state_dict(torch.load("ppoca1.ckpt"))

    for i in range(1):
        state, _ = env.reset()
        # for k in range(100):
        allr=0
        while(True):
            with torch.no_grad():
                # net.eval()
                Q = net(torch.tensor(state).reshape(1, -1).float())
                # Q = net.get_dist(torch.tensor(state).reshape(1, -1).float())
                # action = torch.argmax(Q).item()
                Q = torch.clamp(Q,-max_action,max_action)
                action =Q.detach().numpy().flatten()
                # print(Q)
                # action = env.action_space.sample()
                print("w",action)
                # print(action)
                # action = torch.nn.functional.softmax(Q[0], dim=0).multinomial(num_samples=1, replacement=False).item()
                state_n, r, d, _, _ = env.step(action)
                state = state_n
                # print(action)
                allr+=r
            if d == True:
                print(allr)
                # print(k)
                break

if __name__ == '__main__':
    train()
    # test()

# Etonedu11022157120641000005.TIF
# Etonedu11022155530426000033.TIF
# Etonedu11022155480217000017.TIF