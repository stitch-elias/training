import torch
import gym
import random
class PG(torch.nn.Module):
    def __init__(self,statesize,actionsize):
        super(PG, self).__init__()
        self.fc1 = torch.nn.Linear(statesize, 8)
        self.relu1 = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(8, 16)
        self.relu2 = torch.nn.LeakyReLU()
        self.fc3 = torch.nn.Linear(16, actionsize)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def train(episodes=20000):
    env = gym.make('CartPole-v1')
    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.n

    net = PG(statesize,actionsize)
    net.load_state_dict(torch.load("PG.ckpt"))

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    opt = torch.optim.Adam(net.parameters(),lr=1e-3)

    maxr=0
    for j in range(1):
        for i in range(episodes):
            state,_ = env.reset()
            Allr = 0
            states = []
            actions = []
            rewards = []
            while True:
                with torch.no_grad():
                    Q = net(torch.tensor(state).reshape(1,-1).float())

                    action = torch.nn.functional.softmax(Q[0],dim=0).multinomial(num_samples=1,replacement=False).item()

                    state_n, r, d, _, _ = env.step(action)

                    if d:
                        r=-20

                    Allr+=r
                    states.append(torch.tensor(state).reshape(1,-1))
                    actions.append(action)
                    rewards.append(r)

                if d:
                    rs = 0
                    g = torch.zeros(len(rewards))
                    for t in reversed(range(0,len(rewards))):
                        rs = rs*0.9 + rewards[t]
                        g[t] = rs
                    g = g-torch.mean(g)
                    g = g/torch.std(g)
                    acts = net(torch.cat(states,dim=0))
                    loss = loss_fn(acts,torch.tensor(actions))
                    loss = torch.mean(loss*g)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    break

            torch.save(net.state_dict(), "PG.ckpt")
            if Allr>0:
                print(Allr)
                print(loss)
                print(i)
                print("-------------------------")


def test():
    env = gym.make('CartPole-v1',render_mode="human")
    env = env.unwrapped
    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.n

    net = PG(statesize,actionsize)

    net.load_state_dict(torch.load("PG.ckpt"))
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
    # train()
    test()
