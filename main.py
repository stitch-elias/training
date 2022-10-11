from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision import transforms
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import argparse
import logging
from multiprocessing import cpu_count
import datetime
import time

parser = argparse.ArgumentParser()
parser.add_argument("-m","--mode",dest="mode",default="test",required=False,type=str,nargs="?",choices=["train","test"],action="store",help="choose the mode of program")
parser.add_argument("-p","--modelpath",dest="modelpath",default="./ckpt/resnet18.pth",required=False,type=str,nargs="?",action="store",help="the path to load ckpt")
parser.add_argument("-s","--savepath",dest="savepath",default="./models",required=False,type=str,nargs="?",action="store",help="the path to save model trained")
parser.add_argument("-e","--epoch",dest="epoch",default=100,required=False,type=int,nargs="?",action="store",help="the epochs of model training")
parser.add_argument("-a","--amp",dest="amp",default=False,required=False,type=bool,nargs="?",action="store",help="use amp")
parser.add_argument("-t","--transformer",dest="transformer",default=True,required=False,type=bool,nargs="?",action="store",help="use transformer to expand data")
parser.add_argument("-l","--learningrate",dest="learningrate",default=1e-4,required=False,type=float,nargs="?",action="store",help="the step of gradient descent,learning rate of neural network")
parser.add_argument("-r","--L2regularization",dest="weightdecay",default=1e-2,required=False,type=float,nargs="?",action="store",help="Use L2 regularization to suppress overfitting")
parser.add_argument("-d","--dropout",dest="dropout",default=0.3,required=False,type=float,nargs="?",action="store",help="Use dropout to suppress overfitting")
parser.add_argument("-o","--optim",dest="optim",default="Adam",required=False,type=str,nargs="?",choices=["Adam","SGD"],action="store",help="choose the optimizer of neural network")
parser.add_argument("-i","--logpath",dest="logpath",default="./log",required=False,type=str,nargs="?",action="store",help="the path to save logevent")
parser.add_argument("-g","--tensorboard",dest="tensorboard",default=True,required=False,type=bool,nargs="?",action="store",help="use tensorboard")
parser.add_argument("-c","--cuda",dest="cuda",default=0,required=False,type=int,nargs="?",action="store",help="use cuda")
parser.add_argument("-b","--batchsize",dest="batchsize",default=16,required=False,type=int,nargs="?",action="store",help="the batchsize of dataloader")
args = parser.parse_args()

class pytorch_val(nn.Module):
    def __init__(self,device,numclass=10):
        super(pytorch_val, self).__init__()
        self.device = device
        self.numclass = numclass

    def confusionmatrix(self,p, y, thred, beta=1):
        p = F.softmax(p,dim=1) > thred
        y = F.one_hot(y,self.numclass)

        tp = torch.sum(y.mul(p),dim=0)
        fn = torch.sum(y.mul(~p),dim=0)
        fp = torch.sum((1 - y).mul(p),dim=0)
        tn = torch.sum((1 - y).mul(~p),dim=0)

        tprate = tp / (tp + fn)
        fprate = fp / (fp + tn)

        precision = tp / (tp + fp)
        recall = tprate

        accuracy = (tp + tn) / y.shape[0]

        F_meansure = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)

        return tp, tn, fp, fn, tprate, fprate, precision, recall, accuracy, F_meansure
    def ClassificationVerification(self,p, y):
        ROC = []
        PR = []
        for i in range(0, 11):
            tp, tn, fp, fn, tprate, fprate, precision, recall, accuracy, F_meansure = self.confusionmatrix(p, y, i * 0.1)

            if not len(ROC):
                ROC = torch.concat([tprate.unsqueeze(dim=0),fprate.unsqueeze(dim=0)],dim=0).unsqueeze(dim=0)
                PR = torch.concat([recall.unsqueeze(dim=0), precision.unsqueeze(dim=0)],dim=0).unsqueeze(dim=0)
            else:
                ROC = torch.concat([ROC,torch.concat([fprate.unsqueeze(dim=0), tprate.unsqueeze(dim=0)]).unsqueeze(dim=0)],dim=0)
                PR = torch.concat([PR, torch.concat([recall.unsqueeze(dim=0), precision.unsqueeze(dim=0)]).unsqueeze(dim=0)],dim=0)
                # shape = (10,2,classfication)
        ROC = ROC.permute(2,0,1)
        PR = PR.permute(2, 0, 1)

        PRs = []
        ROCs = []
        AP = []
        AUC = []
        for i in range(self.numclass):
            PR[i,:,:] = PR[i,torch.argsort(PR[i,:, 0])]
            ROC[i,:,:] = ROC[i,torch.argsort(ROC[i,:, 0])]

            PRMask = ~torch.any((PR[i,:,:]).isnan(),dim=1)
            PR_ = PR[i,PRMask,:].unique(dim=0)
            PRs.append(PR_)

            ROCMask = ~torch.any((ROC[i,:,:]).isnan(),dim=1)
            ROC_ = ROC[i,ROCMask,:].unique(dim=0)
            ROCs.append(ROC_)

            AP.append(torch.sum((PR_[1:,1]+PR_[:-1,1])*((PR_[1:,0]-PR_[:-1,0]))*0.5,dim=0))
            AUC.append(torch.sum((ROC_[1:, 1] + ROC_[:-1, 1]) * ((ROC_[1:, 0] - ROC_[:-1, 0])) * 0.5, dim=0))

        AP = torch.tensor(AP)
        AUC = torch.tensor(AUC)
        mAP = torch.mean(AP,dim=0)
        mAUC = torch.mean(AUC,dim=0)
        return PRs,AP,mAP,ROCs,AUC,mAUC

def BCconfusionmatrix(p, y, thred, beta=1):
    # p = F.softmax(p) >= thred
    p = p >= thred

    tp = torch.sum(y[:,1].mul(p[:,1]))
    fn = torch.sum(y[:,1].mul(~p[:,1]))
    fp = torch.sum((1 - y[:,1]).mul(p[:,1]))
    tn = torch.sum((1 - y[:,1]).mul(~p[:,1]))

    tprate = tp / (tp + fn)
    fprate = fp / (fp + tn)

    precision = tp / (tp + fp)
    recall = tprate

    accuracy = (tp + tn) / y.shape[0]

    # 如果β>1, 召回率有更大影响，如果0<β<1,精确率有更大影响。当β=1的时候，精确率和召回率影响力相同，即F1-score
    F_meansure = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)

    return tp, tn, fp, fn, tprate, fprate, precision, recall, accuracy, F_meansure

def BCAUC(p,y):

    np, ip = torch.sort(p[:,1])
    sumrankp = 0.
    for i in torch.unique(np):
        rank = torch.argwhere(torch.eq(np, i)) + 1

        if rank.shape[0] == 1:
            sumrankp += (rank * y[ip[rank - 1],1]).item()
        else:
            sumrankp += torch.mean(rank.float()) * torch.sum(y[ip[rank - 1], 1])

    M = torch.sum(y[:, 1])
    N = y.shape[0] - M
    auc = (sumrankp - (M + 1) * M * 0.5) / (M * N)

    return auc

def BCVerification(p,y):
    ROC =[]
    PR = []
    for i in range(0,10):
        tp, tn, fp, fn, tprate, fprate, precision, recall, accuracy, F_meansure = BCconfusionmatrix(p,y,i*0.1)
        ROC.append([tprate,fprate])
        PR.append([recall,precision])

    PR1 = torch.tensor(PR)
    PR1 = PR1[torch.argsort(PR1[:,0])]

    ap = 0.
    for i in range(1,PR1.shape[0]):
        ap += (PR1[i,1]+PR1[i-1,1])*(PR1[i,0]-PR1[i-1,0])*0.5

    return ROC,PR,ap

class log:
    def __init__(self,path):
        # 第一步，创建一个logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)  # Log等级总开关  此时是INFO

        # 第二步，创建一个handler，用于写入日志文件
        logfile = path +'/log/'+datetime.datetime.now().strftime('%Y-%m-%d %H:%M')+".txt"
        fh = logging.FileHandler(logfile, mode='a')  # open的打开模式这里可以进行参考
        fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关

        # 第三步，再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)  # 输出到console的log等级的开关

        # 第四步，定义handler的输出格式（时间，文件，行数，错误级别，错误提示）
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 第五步，将logger添加到handler里面
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def info(self,message):
        self.logger.info(message)

    def warning(self,message):
        self.logger.warning(message)

    def error(self,message):
        self.logger.error(message)


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                     padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if abs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                padding=0, dilation=self.conv.dilation, groups=self.conv.groups)

            return out_normal - self.theta * out_diff

class SpatialAttention(nn.Module):
    def __init__(self, kernel=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)


class AdaptiveAvgPool(nn.Module):
    def __init__(self,out_s,type="Max"):
        super(AdaptiveAvgPool, self).__init__()
        self.out_s = out_s
        self.type = type
    def forward(self,x):
        in_size = torch.tensor(x.shape[2:])
        out_size = torch.tensor(self.out_s)

        str_size = torch.floor_(in_size/out_size)
        kernel_size = in_size-(out_size-1)*str_size

        stri = (int(str_size[0].item()),int(str_size[1].item()))
        kernel = (int(kernel_size[0].item()),int(kernel_size[1].item()))

        if self.type == 'Avg':
            pool = F.avg_pool2d(x,kernel_size=kernel,stride=stri)
        else:
            pool = F.max_pool2d(x,kernel_size=kernel,stride=stri)
        return pool

class SEBlock(nn.Module):
    def __init__(self,in_c,ratio):
        super(SEBlock, self).__init__()
        self.squeeze = AdaptiveAvgPool((1, 1),type="Avg")
        self.compress = nn.Conv2d(in_c, in_c // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_c // ratio, in_c, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.leaky_relu(out)
        out = self.excitation(out)
        return torch.sigmoid(out)


class BottleBlock(nn.Module):
    def __init__(self,IC,HC,kernel=3,bias=False,SE=False):
        super(BottleBlock, self).__init__()
        self.conv1 = nn.Conv2d(IC,HC,kernel_size=1,bias=bias)
        self.bn1 = nn.BatchNorm2d(HC)
        self.relu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(HC,HC,kernel_size=kernel,stride=1,padding=kernel//2,bias=bias)
        self.bn2 = nn.BatchNorm2d(HC)
        self.relu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(HC,IC,kernel_size=1,bias=bias)
        self.bn3 = nn.BatchNorm2d(IC)
        self.relu3 = nn.LeakyReLU()

        self.is_se = SE
        if SE:
            self.SE = SEBlock(IC,16)


    def forward(self,x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.is_se:
            coefficient = self.SE(out)
            out = out*coefficient

        out = out + x
        out = self.relu3(out)
        return out

class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C//g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)

class SplitBlock(nn.Module):
    def __init__(self, ratio):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        c = int(x.size(1) * self.ratio)   # [0,1,2,3]  size(1) = 2 取得是维度上的值
        return x[:, :c, :, :], x[:, c:, :, :]

class ShuffleBlock1(nn.Module):
    def __init__(self, in_channels, split_ratio=0.5,SE=False):
        super(ShuffleBlock1, self).__init__()
        self.split = SplitBlock(split_ratio)
        in_channels = int(in_channels * split_ratio)
        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.shuffle = ShuffleBlock()

        self.is_se = SE
        if SE:
            self.SE = SEBlock(in_channels,16)


    def forward(self, x):
        x1, x2 = self.split(x)
        out = F.leaky_relu(self.bn1(self.conv1(x2)))
        out = self.bn2(self.conv2(out))
        out = F.leaky_relu(self.bn3(self.conv3(out)))

        if self.is_se:
            coefficient = self.SE(out)
            out = out*coefficient

        out = torch.cat([x1, out], 1)
        out = self.shuffle(out)
        return out


class ShuffleBlock2(nn.Module):
    def __init__(self, in_channels, split_ratio=0.5, SE=False):
        super(ShuffleBlock2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.shuffle = ShuffleBlock()

        self.convD1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
        self.bnD1 = nn.BatchNorm2d(in_channels)
        self.convD2 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=1, bias=False)
        self.bnD2 = nn.BatchNorm2d(in_channels)

        self.is_se = SE
        if SE:
            self.SE = SEBlock(in_channels, 16)

    def forward(self, x):

        out1 = F.relu(self.bn1(self.conv1(x)))
        out1 = self.bn2(self.conv2(out1))
        out1 = F.leaky_relu(self.bn3(self.conv3(out1)))

        if self.is_se:
            coefficient = self.SE(out1)
            out1 = out1*coefficient

        out2 = self.bnD1(self.convD1(x))
        out2 = F.leaky_relu(self.bnD2(self.convD2(out2)))

        out = torch.cat([out1, out2], 1)
        out = self.shuffle(out)
        return out

class MixBlock(nn.Module):
    def __init__(self,in_channels):
        super(MixBlock, self).__init__()
        self.convL1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=1, bias=False)
        self.bnL1 = nn.BatchNorm2d(in_channels)
        self.reluL1 = nn.LeakyReLU()

        self.convL2 = nn.Conv2d(in_channels*2, in_channels,
                               kernel_size=1, bias=False)
        self.bnL2 = nn.BatchNorm2d(in_channels)
        self.reluL2 = nn.LeakyReLU()

        self.convR1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=1, bias=False)
        self.bnR1 = nn.BatchNorm2d(in_channels)
        self.reluR1 = nn.LeakyReLU()

        self.convR2 = nn.Conv2d(in_channels*2, in_channels,
                               kernel_size=1, bias=False)
        self.bnR2 = nn.BatchNorm2d(in_channels)
        self.reluR2 = nn.LeakyReLU()

    def forward(self,x1,x2):
        out1 = self.reluL1(self.bnL1(self.convL1(x1)))
        out2 = self.reluR1(self.bnR1(self.convR1(x2)))

        out = torch.concat([out1,out2],1)

        outL = self.reluL2(self.bnL2(self.convL2(out)))
        outR = self.reluR2(self.bnR2(self.convR2(out)))

        return outL,outR

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,2,1)
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(32)

        self.resL2 = BottleBlock(32,16,SE=True)
        self.convL2 = nn.Conv2d(32,128,3,2,1)
        self.reluL2 = nn.LeakyReLU()
        self.bnL2 = nn.BatchNorm2d(128)

        self.resL3 = nn.Sequential(
            BottleBlock(128,64,SE=True),
            BottleBlock(128, 64,SE=True),
        )
        self.convL3 = nn.Conv2d(128,256,3,2,1)
        self.reluL3 = nn.LeakyReLU()
        self.bnL3 = nn.BatchNorm2d(256)

        # self.CDConvR2 = Conv2d_cd(32,64,3,1,1)
        # self.reluR2 = nn.LeakyReLU()
        # self.bnR2 = nn.BatchNorm2d(64)
        #
        # self.CDConvR3 = Conv2d_cd(64,128,3,2,1)
        # self.reluR3 = nn.LeakyReLU()
        # self.bnR3 = nn.BatchNorm2d(128)
        #
        # self.CDConvR4 = Conv2d_cd(128,128,3,1,1)
        # self.reluR4 = nn.LeakyReLU()
        # self.bnR4 = nn.BatchNorm2d(128)
        #
        # self.CDConvR5 = Conv2d_cd(128,256,3,2,1)
        # self.reluR5 = nn.LeakyReLU()
        # self.bnR5 = nn.BatchNorm2d(256)
        #
        # self.Mix = MixBlock(128)

        # self.lastConv = nn.Conv2d(512,512,3,1,1)
        # self.lastbn = nn.BatchNorm2d(512)
        # self.lastrelu = nn.LeakyReLU()

        self.pool = AdaptiveAvgPool(1,"Max")

        self.flatten = nn.Flatten()

        self.Linear = nn.Linear(256,10)

        self.softmax = nn.Softmax()

    def forward(self,x):
        out = self.relu1(self.bn1(self.conv1(x)))

        outL = self.bnL2(self.bnL2(self.convL2(self.resL2(out))))

        # outR = self.bnR2(self.bnR2(self.CDConvR2(out)))
        # outR = self.bnR3(self.bnR3(self.CDConvR3(outR)))

        # outL1,outR1 = self.Mix(outL,outR)

        outL = self.bnL3(self.bnL3(self.convL3(self.resL3(outL))))

        # outR = self.bnR4(self.bnR4(self.CDConvR4(outR+outR1)))
        # outR = self.bnR5(self.bnR5(self.CDConvR5(outR)))

        # out = torch.concat([outL,outR],1)

        # out = self.lastrelu(self.lastbn(self.lastConv(out)))

        out = self.pool(outL)

        out = self.Linear(self.flatten(out))

        # out = self.softmax(out)

        return out



class trainer:
    def __init__(self,args):

        global transform
        if args.transformer:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        self.trainloader = DataLoader(MNIST('./data/', train=True, download=True,
                             transform=transform),batch_size=args.batchsize,shuffle=True,drop_last=True,num_workers=cpu_count())
        self.testloader = DataLoader(MNIST('./data/', train=False, download=True,
                             transform=transform),batch_size=args.batchsize,shuffle=True,drop_last=True,num_workers=cpu_count())

        # self.net = resnet18(progress=False)
        # if args.modelpath is not None:
        #     self.net.load_state_dict(torch.load(args.modelpath))

        # self.net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.net.fc = nn.Linear(in_features=512, out_features=10, bias=True)
        # for name, para in self.net.named_parameters():
        #     if "conv1.weight" != name and "fc.weight" != name:
        #         para.requires_grad = False

        self.net = Net()

        if args.cuda is not None:
            if torch.cuda.is_available():
                self.device = torch.device(args.cuda)
                if args.amp:
                    self.amp = GradScaler()
                else:
                    self.amp = None

        if args.weightdecay != 0:
            weightdecay = args.weightdecay
        else:
            weightdecay = 0

        if args.optim == "Adam":
            self.optim = torch.optim.Adam(self.net.parameters(),lr=args.learningrate,weight_decay=weightdecay)
        else:
            self.optim = torch.optim.SGD(self.net.parameters(),lr=args.learningrate,weight_decay=weightdecay)

        self.loss_fn = nn.CrossEntropyLoss()

        self.savepath = args.savepath

        # if args.tensorboard:
        #     self.tensorboard = SummaryWriter(args.logpath+"/tensorboard/"+datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))

        self.epoch = args.epoch

        # self.Log = log(args.logpath)

        self.val = pytorch_val(self.device)

    def train(self,is_val=True):
        self.net = self.net.to(self.device)
        for i in range(self.epoch):
            self.net.train()
            sumloss = 0
            k = 0
            for index,(x,y) in enumerate(self.trainloader):
                x,y = x.to(self.device),y.to(self.device)

                self.optim.zero_grad()

                if self.amp is not None:
                    with autocast():
                        pred = self.net(x)
                        loss = self.loss_fn(pred, y)
                    self.amp.scale(loss).backward()
                    self.amp.step(self.optim)
                    self.amp.update()

                else:
                    pred = self.net(x)

                    loss = self.loss_fn(pred, y)

                    loss.backward()
                    self.optim.step()

                sumloss += loss.detach().cpu().item()

                k += 16
            trainavgloss = sumloss/k

            if is_val:
                self.net.eval()
                with torch.no_grad():
                    # acc=0
                    k1=0
                    sumloss = 0

                    ps = []
                    ys = []
                    t0 = time.time()
                    for index, (x, y) in enumerate(self.trainloader):
                        x, y = x.to(self.device), y.to(self.device)
                        pred = self.net(x)
                        loss = self.loss_fn(pred, y)

                        sumloss += loss.detach().cpu().item()

                        # pred = torch.argmax(F.softmax(pred,dim=1),dim=1).detach().cpu()
                        # y = y.detach().cpu()
                        # acc += torch.sum(torch.eq(pred,y)).item()

                        if len(ps):
                            ps = torch.concat([ps,pred],dim=0)
                            ys = torch.concat([ys,y],dim=0)
                        else:
                            ps = pred
                            ys = y

                        k1 += 16

                    # acc = acc/k1
                    avgloss = sumloss/k1


                    t1 = time.time()
                    tp, tn, fp, fn, tprate, fprate, precision, recall, accuracy, F_meansure = self.val.confusionmatrix(ps,ys,0.5)

                    PR, AP, mAP, ROC, AUC, mAUC = self.val.ClassificationVerification(ps,ys)

                    print("num"+str(k1)+" time0:"+str(t1-t0)+" time1:"+str(time.time()-t1))




                # self.Log.info("epoch:"+str(i)+"-------trainloss:"+str(trainavgloss)+"-----------testloss:"+str(avgloss)+"---------testacc:"+str(acc))
                # self.tensorboard.add_scalar("trainloss",trainavgloss,i)
                # self.tensorboard.add_scalar("testloss",avgloss,i)
                # self.tensorboard.add_scalar("acc",acc,i)

                print("epoch:"+str(i)+"-------trainloss:"+str(trainavgloss)+"-----------testloss:"+str(avgloss)+"---------testacc:"+str(accuracy))
                print("epoch:"+str(i)+"-------F1:"+str(torch.mean(F_meansure).item())+"-----------mAP:"+str(mAP.item())+"---------mAUC:"+str(mAUC.item()))

                # torch.save(self.net.state_dict(),self.savepath+"/latest.ckpt")
                #
                # if sacc<acc:
                #     print("save")
                #     torch.save(self.net.state_dict(), self.savepath + "/best.ckpt")
                #     sacc = acc

if __name__ == '__main__':
    tr = trainer(args)
    tr.train()
