import torch
import random
import gym
class Actor(torch.nn.Module):
    def __init__(self,statesize,actionsize):
        super(Actor, self).__init__()
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

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc6(x)
        return x

class Critic(torch.nn.Module):
    def __init__(self,statesize):
        super(Critic, self).__init__()
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
        self.fc6 = torch.nn.Linear(16,1)

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

def train(episodes=200000,decay=0.99):
    env = gym.make('CartPole-v1')
    env = env.unwrapped
    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.n

    actor = Actor(statesize,actionsize)
    # net.load_state_dict(torch.load("DDQN1.ckpt"))
    critic = Critic(statesize)
    # targetnet.load_state_dict(torch.load("DDQN1.ckpt"))

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    opt_a = torch.optim.Adam(actor.parameters(),lr=5e-4)
    opt_c = torch.optim.Adam(critic.parameters(),lr=5e-4)

    maxr = 0

    for i in range(episodes):
        state,_ = env.reset()
        allr = 0

        for j in range(100):

            with torch.no_grad():
                Q = actor(torch.tensor(state).reshape(1, -1).float()).detach()

                action = torch.nn.functional.softmax(Q[0], dim=0).multinomial(num_samples=1, replacement=False).item()

            state_n, r, d, _, _ = env.step(action)

            if d:
                r = -20
            v = critic(torch.tensor(state).reshape(1,-1).float())
            with torch.no_grad():
                v_n = critic(torch.tensor(state_n).reshape(1,-1).float())
                td_error = r+decay*(1-d)*v_n - v

            loss_c = torch.square(td_error-v)
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()



            acts = actor(torch.tensor(state).reshape(1,-1).float())
            loss_a = loss_fn(acts, torch.tensor([action]))
            loss_a = torch.sum(loss_a * td_error)

            opt_a.zero_grad()
            loss_a.backward()
            opt_a.zero_grad()

            state = state_n

            allr +=r

            if d:
                if allr>0:
                    print(allr)
                    print(i)
                    print("---------------------------")
                break

        if i>1500:
            if allr>maxr:
                torch.save(actor.state_dict(), "actor.ckpt")
                torch.save(critic.state_dict(), "critic.ckpt")
                maxr = allr

def test():
    env = gym.make('CartPole-v1',render_mode="human")
    env = env.unwrapped
    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.n

    net = Actor(statesize,actionsize)

    net.load_state_dict(torch.load("actor.ckpt"))
    for i in range(2):
        state, _ = env.reset()
        # for k in range(100):
        allr=0
        while(True):
            with torch.no_grad():
                net.eval()
                Q = net(torch.tensor(state).reshape(1, -1).float())
                action = torch.argmax(Q).item()
                # action = torch.nn.functional.softmax(Q[0], dim=0).multinomial(num_samples=1, replacement=False).item()
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
    train()
    # test()
