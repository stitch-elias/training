from torch import nn
import torch.nn.functional as F
import torch
class evalution(nn.Module):
    def __init__(self,device,numclass=10):
        super(evalution, self).__init__()
        self.device = device
        self.numclass = numclass

    def confusionmatrix(self,p, y, thred, beta=1,softmax=True,onehot=True):

        if softmax:
            p = F.softmax(p,dim=1)
        p = p > thred
        # print(y.shape)
        if onehot:
            y = F.one_hot(y,self.numclass)
        # print(p.shape)
        # print(y.shape)
        tp = torch.sum(y.mul(p),dim=0)
        fn = torch.sum(y.mul(~p),dim=0)
        fp = torch.sum((1 - y).mul(p),dim=0)
        tn = torch.sum((1 - y).mul(~p),dim=0)

        tprate = tp / (tp + fn)
        fprate = fp / (fp + tn)

        precision = tp / (tp + fp)
        recall = tprate

        accuracy = (tp + tn) / (tp+tn+fp+fn)

        F_meansure = (1 + beta * beta) * precision * recall / (beta * beta * (precision + recall))

        return tp, tn, fp, fn, tprate, fprate, precision, recall, accuracy, F_meansure
    def ClassificationVerification(self,p, y,softmax=True ,onehot=True,count=100):
        ROC = []
        PR = []

        if softmax:
            p = F.softmax(p, dim=1)

        index = torch.argsort(p[:,1])
        p = p[index]
        y = y[index]
        num = len(p)//count
        for i in range(count):
            tp, tn, fp, fn, tprate, fprate, precision, recall, accuracy, F_meansure = self.confusionmatrix(p, y, p[num*i],softmax=False,onehot=onehot)
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