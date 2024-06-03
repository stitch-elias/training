import gym
import torch
import random
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import cv2
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def orthogonal_init(layer, gain=1.0, bias=True):
    torch.nn.init.orthogonal_(layer.weight, gain=gain)
    if bias:
        torch.nn.init.constant_(layer.bias, 0)


class AdaptiveAvgPool(torch.nn.Module):
    def __init__(self, out_s, type="Max"):
        super(AdaptiveAvgPool, self).__init__()
        self.out_s = out_s
        self.type = type

    def forward(self, x):
        in_size = torch.tensor(x.shape[2:])
        out_size = torch.tensor(self.out_s)

        str_size = torch.floor_(in_size / out_size)
        kernel_size = in_size - (out_size - 1) * str_size

        stri = (int(str_size[0].item()), int(str_size[1].item()))
        kernel = (int(kernel_size[0].item()), int(kernel_size[1].item()))

        if self.type == 'Avg':
            pool = F.avg_pool2d(x, kernel_size=kernel, stride=stri)
        else:
            pool = F.max_pool2d(x, kernel_size=kernel, stride=stri)
        return pool


class ShuffleBlock(torch.nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class SEBlock(torch.nn.Module):
    def __init__(self, in_c, ratio):
        super(SEBlock, self).__init__()
        self.squeeze = AdaptiveAvgPool((1, 1), type="Avg")
        self.compress = torch.nn.Conv2d(in_c, in_c // ratio, 1, 1, 0)
        self.excitation = torch.nn.Conv2d(in_c // ratio, in_c, 1, 1, 0)

        orthogonal_init(self.compress)
        orthogonal_init(self.excitation)
        self.compress.inited = True
        self.excitation.inited = True

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.leaky_relu(out)
        out = self.excitation(out)
        return torch.sigmoid(out)


class PSA_p(torch.nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PSA_p, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2

        self.conv_q_right = torch.nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = torch.nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                            bias=False)
        self.conv_up = torch.nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = torch.nn.Softmax(dim=2)
        self.sigmoid = torch.nn.Sigmoid()

        self.conv_q_left = torch.nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                           bias=False)  # g
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = torch.nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                           bias=False)  # theta
        self.softmax_left = torch.nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):
        orthogonal_init(self.conv_q_right, bias=False)
        orthogonal_init(self.conv_v_right, bias=False)
        orthogonal_init(self.conv_q_left, bias=False)
        orthogonal_init(self.conv_v_left, bias=False)

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)

        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1, 2))
        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        context = self.conv_up(context)

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out

    def channel_pool(self, x):
        # [N, IC, H, W]
        g_x = self.conv_q_left(x)

        batch, channel, height, width = g_x.size()

        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)

        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        # [N, IC, H*W]
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)

        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = torch.matmul(avg_x, theta_x)
        # [N, 1, H*W]
        context = self.softmax_left(context)

        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)

        out = x * mask_sp

        return out

    def forward(self, x):
        # [N, C, H, W]
        context_channel = self.spatial_pool(x)
        # [N, C, H, W]
        context_spatial = self.channel_pool(x)
        # [N, C, H, W]
        out = context_spatial + context_channel
        return out

class ShuffleBlock2(torch.nn.Module):
    def __init__(self, in_channels, split_ratio=0.5, SE=False):
        super(ShuffleBlock2, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels, in_channels,
                                     kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(in_channels)
        self.conv2 = torch.nn.Conv2d(in_channels, in_channels,
                                     kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(in_channels)
        self.conv3 = torch.nn.Conv2d(in_channels, in_channels,
                                     kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(in_channels)
        self.shuffle = ShuffleBlock()

        self.convD1 = torch.nn.Conv2d(in_channels, in_channels,
                                      kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
        self.bnD1 = torch.nn.BatchNorm2d(in_channels)
        self.convD2 = torch.nn.Conv2d(in_channels, in_channels,
                                      kernel_size=1, bias=False)
        self.bnD2 = torch.nn.BatchNorm2d(in_channels)

        self.is_se = SE
        if SE:
            self.SE = PSA_p(in_channels, in_channels)

        orthogonal_init(self.conv1, bias=False)
        self.conv1.inited = True
        orthogonal_init(self.conv2, bias=False)
        self.conv2.inited = True
        orthogonal_init(self.conv3, bias=False)
        self.conv3.inited = True
        orthogonal_init(self.convD1, bias=False)
        self.convD1.inited = True
        orthogonal_init(self.convD2, bias=False)
        self.convD2.inited = True

    def forward(self, x):

        out1 = F.softplus(self.bn1(self.conv1(x)))
        out1 = F.softplus(self.bn2(self.conv2(out1)))
        out1 = F.softplus(self.bn3(self.conv3(out1)))

        if self.is_se:
            out1 = self.SE(out1)

        out2 = self.bnD1(self.convD1(x))
        out2 = F.softplus(self.bnD2(self.convD2(out2)))

        out = torch.cat([out1, out2], 1)
        out = self.shuffle(out)
        return out


class DQN(torch.nn.Module):
    def __init__(self, statesize, actionsize):
        super(DQN, self).__init__()
        self.Conv1 = torch.nn.Conv2d(statesize, 32, 3, 1, 1)
        self.BN1 = torch.nn.BatchNorm2d(32)
        self.Conv2 = torch.nn.Conv2d(32, 64, 3, 2, 1)
        self.BN2 = torch.nn.BatchNorm2d(64)
        self.shuffle1 = ShuffleBlock2(64, SE=True)
        self.shuffle2 = ShuffleBlock2(128, SE=True)
        self.Conv5 = torch.nn.Conv2d(256, 256, 3, 1, 1)
        self.BN5 = torch.nn.BatchNorm2d(256)
        self.flatten = torch.nn.Flatten()
        self.Linear = torch.nn.Linear(138240, actionsize)

        orthogonal_init(self.Conv1)
        self.Conv1.inited = True
        orthogonal_init(self.Conv2)
        self.Conv2.inited = True
        orthogonal_init(self.Conv5)
        self.Conv5.inited = True
        orthogonal_init(self.Linear)
        self.Linear.inited = True

    def forward(self, x):
        x = F.softplus(self.BN1(self.Conv1(x)))
        x = F.softplus(self.BN2(self.Conv2(x)))
        x = self.shuffle1(x)
        x = self.shuffle2(x)
        x = F.softplus(self.BN5(self.Conv5(x)))
        x = self.flatten(x)
        x = self.Linear(x)
        return x


class Memery:
    def __init__(self, statesize, actionsize, size=2048):
        self.state = torch.zeros((size, statesize[2], statesize[0], statesize[1]))
        self.action = torch.zeros((size, 1), dtype=torch.int64)
        self.r = torch.zeros((size, 1))
        self.state_n = torch.zeros((size, statesize[2], statesize[0], statesize[1]))
        self.d = torch.zeros((size, 1))
        self.index = 0

    def append(self, state, action, r, state_n, d):
        if self.index >= 2049:
            self.state[:-1] = self.state[1:]
            self.state[-1] = state
            self.action[:-1] = self.action[1:].clone()
            self.action[-1] = action
            self.state_n[:-1] = self.state_n[1:].clone()
            self.state_n[-1] = state_n
            self.r[:-1] = self.r[1:].clone()
            self.r[-1] = r
            self.d[:-1] = self.d[1:].clone()
            self.d[-1] = d
        else:
            self.state[self.index, :] = torch.tensor(state)
            self.action[self.index, :] = torch.tensor(action)
            self.r[self.index, :] = torch.tensor(r)
            self.state_n[self.index, :] = torch.tensor(state_n)
            self.d[self.index, :] = torch.tensor(d)
        self.index += 1
        self.index = self.index % 2048

    def __len__(self):
        return self.index

    def __getitem__(self, item):
        return (self.state[item], self.action[item], self.r[item], self.state_n[item], self.d[item])

    def __call__(self, *args, **kwargs):
        return (self.state, self.action, self.r, self.state_n, self.d)


def train(episodes=5000, decay=0.9, greedy=0.):
    env = gym.make("ALE/Breakout-v5")
    env = env.unwrapped
    statesize = env.observation_space.shape
    actionsize = env.action_space.n

    net = DQN(statesize[2], actionsize).cuda()
    # net.load_state_dict(torch.load("bk.ckpt"))
    targetnet = DQN(statesize[2], actionsize)
    # targetnet.load_state_dict(torch.load("bk.ckpt"))

    memory = Memery(statesize, actionsize)
    memory1 = Memery(statesize, actionsize)

    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4, eps=1e-5)

    maxr = 0

    sw = 0
    for i in range(episodes):
        state_, _ = env.reset()
        allr = 0
        state_ = ToTensor()(cv2.cvtColor(state_, cv2.COLOR_BGR2GRAY))
        state = torch.repeat_interleave(state_, 3, dim=0)
        state_n = state.clone()
        lives = 5

        memory1.index = 0
        for j in range(10000):
            dd = 0
            with torch.no_grad():
                if random.randint(1, 10) * 0.1 > greedy:
                    action = env.action_space.sample()
                else:
                    Q = net(state.unsqueeze(0).cuda()).cpu()
                    action = torch.argmax(Q).item()
            state_n_, r, d, info, _ = env.step(action)
            state_n_ = ToTensor()(cv2.cvtColor(state_n_, cv2.COLOR_BGR2GRAY))
            state_n[1:] = state[:-1].clone()
            state_n[0] = state_n_.clone()
            if lives - _['lives']:
                dd = 1
                lives = _['lives']
            if d:
                dd = 1
            memory1.append(state, action, r, state_n, dd)


            state = state_n

            allr += r

            if d:
                if allr > 0:
                    print(allr)
                    print(i)
                    print("---------------------------")
                break
        if allr>0:
            statesq, actionsq, rsq, state_nsq, dsq = memory1()
            for j in range(memory1.index):
                memory.append(statesq[j], actionsq[j], rsq[j], state_nsq[j], dsq[j])
                if len(memory) == 2047:
                    sw = 1
        if sw:
            states, actions, rs, state_ns, ds = memory()
            for index in BatchSampler(SubsetRandomSampler(range(2048)), 64, False):
                with torch.no_grad():
                    states_ = states[index].clone().cuda()
                    state_ns_ = state_ns[index].clone().cuda()
                    Q_ = net(states_).cpu()
                    Q_n = net(state_ns_).cpu()
                    targetQ_n = targetnet(state_ns[index])
                    targetQ = Q_
                    actions_ = actions[index]
                    ds_ = ds[index].flatten()
                    rs_ = rs[index]
                    if sum(ds_ == True) > 0:
                        targetQ[ds_ == True, actions_[ds_ == True].flatten()] = rs_[ds_ == True].flatten()
                    targetQ[ds_ == False, actions_[ds_ == False].flatten()] = rs_[ds_ == False].flatten() + decay * \
                                                                              targetQ_n[ds_ == False, torch.argmax(
                                                                                  Q_n[ds_ == False], dim=1)]
                targetQ = torch.nn.functional.softmax(targetQ, dim=1)
                TQ = net(states_)
                loss = loss_fn(TQ, targetQ.cuda())
                print(loss)
                opt.zero_grad()
                loss.backward()
                opt.step()
        if i > 1000 and allr > 0 and len(memory) > 2048:
            if greedy < 1:
                greedy += 0.001
        if i % 10 == 0:
            targetnet.load_state_dict(net.state_dict())

        if i > 4000:
            if allr > maxr:
                torch.save(net.state_dict(), "bk.ckpt")
                maxr = allr
        else:
            torch.save(net.state_dict(), "bk.ckpt")


def test():
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    env = env.unwrapped
    statesize = env.observation_space.shape
    actionsize = env.action_space.n

    net = DQN(statesize[2], actionsize)

    net.load_state_dict(torch.load("bk.ckpt"))
    for i in range(1):
        state_, _ = env.reset()
        allr = 0
        state_ = ToTensor()(cv2.cvtColor(state_, cv2.COLOR_BGR2GRAY))
        state = torch.repeat_interleave(state_, 3, dim=0)
        state_n = state.clone()
        k = 0
        while True:
            with torch.no_grad():
                net.eval()
                with torch.no_grad():
                    if k == 0:
                        action = 1
                        k += 1
                    else:
                        Q = net(state.unsqueeze(0))
                        action = torch.argmax(Q).item()

                state_n_, r, d, _, _ = env.step(action)
                state_n_ = ToTensor()(cv2.cvtColor(state_n_, cv2.COLOR_BGR2GRAY))
                state_n[1:] = state[:-1].clone()
                state_n[0] = state_n_.clone()
                state = state_n
                # print(Q)
                print(action)
                allr += r
            if d:
                print(allr)
                # print(k)
                break


if __name__ == '__main__':
    # train()
    # test()

    # env = gym.make("ALE/Breakout-v5", render_mode="human")
    #
    # env.reset()
    # done = False
    #
    # action = 1
    # while not done:
    #     # action = env.action_space.sample()
    #
    #     observation, reward, done, info,_ = env.step(action)
    #     print(_)
    #     if action ==0:
    #         break
    #
    # env.close()
    a = [1.699108+1.312427,1.422731+1.579996,0.275719+0.269130]
    print(torch.softmax(torch.tensor(a),dim=0))

    import numpy as np
    g=0
    for i in a:
        g+=np.exp(i)
    for i in a:
        print(np.exp(i)/g)


