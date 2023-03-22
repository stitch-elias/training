import numpy as np
from torch.optim import lr_scheduler
class WarmupCosineLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_min, lr_max, warm_up=0, T_max=10, start_ratio=0.1, T_mult=1, lr_gamma=1):
        """
        Description:
            - get warmup consine lr scheduler

        Arguments:
            - optimizer: (torch.optim.*), torch optimizer
            - lr_min: (float), minimum learning rate
            - lr_max: (float), maximum learning rate
            - warm_up: (int),  warm_up epoch or iteration
            - T_max: (int), maximum epoch or iteration
            - start_ratio: (float), to control epoch 0 lr, if ratio=0, then epoch 0 lr is lr_min

        Example:
            <<< epochs = 100
            <<< warm_up = 5
            <<< cosine_lr = WarmupCosineLR(optimizer, 1e-9, 1e-3, warm_up, epochs)
            <<< lrs = []
            <<< for epoch in range(epochs):
            <<<     optimizer.step()
            <<<     lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
            <<<     cosine_lr.step()
            <<< plt.plot(lrs, color='r')
            <<< plt.show()

        """
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warm_up = warm_up
        self.T_max = T_max
        self.start_ratio = start_ratio
        self.cur = 0  # current epoch or iteration
        self.T_mult = T_mult
        self.lr_gamma = lr_gamma
        super().__init__(optimizer, -1)

    def get_lr(self):
        if self.lr_gamma <= self.lr_min:
            lr = self.lr_min
        else:
            if (self.warm_up == 0) & (self.cur== 0):
                lr = self.lr_max
            elif (self.warm_up != 0) & (self.cur <= self.warm_up):
                if self.cur == 0:
                    lr = self.lr_min + (self.lr_max - self.lr_min) * (self.cur + self.start_ratio) / self.warm_up
                else:
                    lr = self.lr_min + (self.lr_max - self.lr_min) * (self.cur) / self.warm_up
                    # print(f'{self.cur} -> {lr}')
            else:
                # this works fine
                lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 * \
                     (np.cos((self.cur - self.warm_up) / (self.T_max - self.warm_up) * np.pi) + 1)

            self.cur += 1
            if lr==self.lr_max:
                self.T_max = int(self.T_max*self.T_mult)
                self.cur=self.warm_up+1

            if lr==self.lr_min:
                self.lr_max = self.lr_max*self.lr_gamma

        return [lr for base_lr in self.base_lrs]