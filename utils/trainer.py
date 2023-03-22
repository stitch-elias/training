from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import os
from multiprocessing import cpu_count
import datetime
import time
from tqdm import tqdm

from dataset.dataset import celebA,mcelebA,dataset_collate
from models.F3net import F3Net
from utils.log import log
from utils.evalution import evalution
from utils.utils import WarmupCosineLR
from torch.optim.lr_scheduler import StepLR,MultiStepLR
from models.MultiFTNet import MultiFTNet1,Arc_Net
from models.LGSC import algsc
from models.convnext import TripletLoss,reg_loss
from models.encode import nLGSC
from models.multiFasnet import train_net


class trainer:
    def __init__(self,config):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # self.traincaleba = mcelebA(config.get("path", "datas_path"), train=True,
        #                                      transform=transform)
        # self.testceleba = mcelebA(config.get("path", "datas_path"), train=True,
        #                                     transform=transform)

        self.traincaleba = mcelebA(config.get("path", "datas_path"), train=True,
                                             transform=transform)
        self.testceleba = mcelebA(config.get("path", "datas_path"), train=True,
                                            transform=transform)

        self.trainbatch = config.getint("train", "batchsize")
        self.testbatch = config.getint("test", "batchsize")


        # self.net = F3Net(num_classes=2, mode="Mix")
        # self.net = MultiFTNet1(num_classes=4)

        self.net = train_net()

        # self.net.load_state_dict(torch.load("ckpt/LGSCbest.ckpt"))
        # self.net.load(config.get("path", "models_save_path"))

        self.cuda = config.get("train", "cuda")[1:-1].split(",")

        self.eval = 1

        if len(self.cuda[0]):
            if torch.cuda.is_available():
                if config.getint("train", "dataparallel"):
                    dist.init_process_group(backend="gloo|nccl")
                    local_rank = int(os.environ["LOCAL_RANK"])
                    self.net = DDP(self.net, device_ids=int(self.cuda[local_rank]),
                                   output_device=int(self.cuda[local_rank]))
                    self.eval = local_rank
                    celeba = celebA(config.get("path", "datas_path"), train=True,
                                    transform=transform)
                    train_sampler = torch.utils.data.distributed.DistributedSampler(celeba)
                    self.trainloader = DataLoader(celeba,
                                                  batch_size=self.trainbatch, shuffle=False,
                                                  sampler=train_sampler,drop_last=True)
                    self.device = torch.device(int(self.cuda[local_rank]))
                else:
                    self.device = torch.device(int(self.cuda[0]))
                    self.net.to(self.device)

                if config.getint("train", "amp"):
                    self.amp = GradScaler()
                else:
                    self.amp = None

        if config.get("train", "optim") == "Adam":
            self.optim = torch.optim.Adam(self.net.parameters(), lr=config.getfloat("train", "lerning_rate"), weight_decay=config.getfloat("train", "weightdecay"))
        else:
            self.optim = torch.optim.SGD(self.net.parameters(), lr=config.getfloat("train", "lerning_rate"), weight_decay=config.getfloat("train", "weightdecay"),momentum=config.getfloat("train", "momentum"))

        # self.LR = WarmupCosineLR(self.optim,1e-9,config.getfloat("train", "lerning_rate"),10,20,1e-4,1.5,0.1)
        self.LR = MultiStepLR(self.optim,[100, 150, 220],gamma=0.1,last_epoch=-1)

        self.loss_fn = nn.NLLLoss(reduction="sum")
        self.loss_fn1 = nn.MSELoss()
        self.loss_fn2 = nn.CrossEntropyLoss()
        # self.loss_fn2 = reg_loss()
        # self.ft_criterion = nn.MSELoss(reduction="sum")
        # self.tloss = nn.CrossEntropyLoss(reduction="sum")

        self.savepath = config.get("path", "models_save_path")

        if config.getint("train", "tensorboard"):
            self.tensorboard = SummaryWriter(config.get("path", "tensorboard_path")+"/"+datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))

        self.epoch = config.getint("train", "epoch")

        self.Log = log(config.get("path", "log_path"))

        # self.val = evalution(self.device,numclass=2)

    def train(self, is_val=True):
        hisacc = 0
        num = 0
        self.trainloader = DataLoader(self.traincaleba, batch_size=self.trainbatch,
                                      shuffle=True, drop_last=True, num_workers=cpu_count() - 1)
        # self.testloader = DataLoader(self.testceleba, batch_size=self.testbatch, drop_last=True,
        #                              num_workers=cpu_count() - 1)
        for i in range(self.epoch):

            self.net.train()
            sumloss = 0
            acc = 0
            with tqdm(total=(len(self.traincaleba)//self.trainbatch+1), desc=f'Epoch {i + 1}/{self.epoch}', postfix=dict, mininterval=0.3) as pbar:
                for index, (x,x1, y,ft,ft1) in enumerate(self.trainloader):
                    x, y= x.to(self.device), y.to(self.device)
                    x1, ft,ft1= x1.to(self.device), ft.to(self.device), ft1.to(self.device)

                    self.optim.zero_grad()

                    if self.amp is not None:
                        pass
                    #     with autocast():
                    #         pred,ft,llbp = self.net(x)
                    #         loss = 0.4*self.loss_fn(llbp, y) + 0.4*self.ft_criterion(ft,tf_img[:,:1,:,:]) + 0.2*self.tloss(pred,y)
                    #     self.amp.scale(loss).backward()
                    #     self.amp.step(self.optim)
                    #     self.amp.update()

                    else:
                        # pred, ft, llbp = self.net(x)
                        # loss = self.loss_fn(llbp, y) + self.ft_criterion(ft, tf_img[:,:1,:,:]) + self.tloss(
                        #     pred, y)
                        arc,(f0,f1),pred = self.net(x,x1)
                        # print(pred)
                        # print(y)
                        # trloss = 0
                        # for feat in out:
                        #     feat = F.avg_pool2d(feat,3,2,1)
                        #     feat = feat.reshape(-1,feat.shape[2]*feat.shape[3])
                            # trloss += self.loss_fn(feat,y)
                        loss = self.loss_fn(arc,y) + self.loss_fn1(f0,ft) + self.loss_fn1(f1,ft1) +self.loss_fn2(pred,y)

                        acc += torch.sum(torch.eq(torch.argmax(pred, dim=1), y)).item()

                        loss.backward()
                        self.optim.step()
                    self.LR.step()

                    sumloss += loss.detach().cpu().item()
                    # print(loss)
                    if num%100 ==0:
                        self.tensorboard.add_scalar("trloss", loss.detach().cpu().item(), num//100)
                    num+=1

                    pbar.set_postfix(**{'loss': loss.detach().cpu().item() / (self.trainbatch)})
                    pbar.update(1)

                # self.traincaleba.step()
                gamma = 1 / (index * self.testbatch )
                acc = acc * gamma
                trainavgloss = sumloss / (index*self.trainbatch)
                print("avgloss:",trainavgloss,"acc:",acc)
                if hisacc < acc:
                    print("save:", acc)
                    self.net.save(self.savepath,True)
                    hisacc = acc
                self.net.save(self.savepath)

            if is_val and self.eval == 0:
                self.net.eval()
                with torch.no_grad():
                    acc = 0
                    sumloss = 0

                    ps = []
                    ys = []
                    t0 = time.time()
                    with tqdm(total=(len(self.testceleba)//self.testbatch+1), desc=f'Epoch {i + 1}/{self.epoch}', postfix=dict,
                              mininterval=0.3) as pbar:
                        for index, (x, y) in enumerate(self.testloader):
                            x, y = x.to(self.device), y.to(self.device)
                            feature,pred = self.net(x,False)
                            loss = self.loss_fn1(pred,y)

                            sumloss += loss.detach().cpu().item()

                            acc += torch.sum(torch.eq(torch.argmax(pred, dim=1), y)).item()

                            ps.append(pred)
                            ys.append(y)

                            pbar.set_postfix(**{'val_loss': loss.detach().cpu().item() / (self.testbatch*2)})
                            pbar.update(1)

                        ps = torch.concat(ps, dim=0)
                        ys = torch.concat(ys, dim=0)
                        gamma = 1/(index*self.testbatch*2)
                        acc = acc * gamma
                        avgloss = sumloss *gamma

                        t1 = time.time()

                        ps = F.softmax(ps)
                        ys = F.one_hot(ys, 2)

                        tp, tn, fp, fn, tprate, fprate, precision, recall, accuracy, F_meansure = self.val.confusionmatrix(
                            ps, ys, torch.max(ps) / 2, softmax=False, onehot=False)

                        PR, AP, mAP, ROC, AUC, mAUC = self.val.ClassificationVerification(ps, ys, softmax=False,
                                                                                          onehot=False)
                        avgtime = (t1 - t0)*gamma

                # self.testceleba.step()
                self.Log.info("epoch:"+str(i)+"-------trainloss:"+str(trainavgloss)+"-----------testloss:"+str(avgloss)+"---------testacc:"+str(acc)+"-------avgtime:" + str(avgtime))
                self.tensorboard.add_scalar("trainloss",trainavgloss,i)
                self.tensorboard.add_scalar("testloss",avgloss,i)
                self.tensorboard.add_scalar("acc",acc,i)
                print("epoch:" + str(i) + "-------trainloss:" + str(trainavgloss) + "-----------testloss:" + str(
                    avgloss) + "---------testacc:" + str(acc))
                print("epoch:" + str(i) + "-------F1:" + str(torch.mean(F_meansure).item()) + "-----------mAP:" + str(
                    mAP.item()) + "---------mAUC:" + str(mAUC.item()))
                print("avgtime:" + str(avgtime))

                if hisacc<acc:
                    print("save:",acc)
                    torch.save(self.net.state_dict(), self.savepath + "/LGSCbest.ckpt")
                    # self.net.save(self.savepath,True)
                    hisacc = acc
            # torch.save(self.net.state_dict(), self.savepath + "/LGSClatest.ckpt")
            self.net.save(self.savepath)


        dist.destroy_process_group()
