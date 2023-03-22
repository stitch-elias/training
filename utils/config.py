import os
import configparser

class config:
    def __init__(self,path="config.ini"):
        self.conf = configparser.ConfigParser()
        self.curpath = os.getcwd()
        self.path = os.path.join(self.curpath,path)

        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                self.conf.write(f)

        self.conf.read(self.path)

    def init(self):
        self.conf = configparser.ConfigParser()
        self.conf.add_section("path")
        self.conf.set("path","datas_path","/home/qiu/ddata/CelebA_Spoof/Data")
        self.conf.set("path","pre-trained_models_path",os.path.join(self.curpath,"ckpt"))
        self.conf.set("path","models_save_path",os.path.join(self.curpath,"ckpt"))
        self.conf.set("path","log_path",os.path.join(self.curpath,"logs/event"))
        self.conf.set("path","tensorboard_path",os.path.join(self.curpath,"logs/tensorboard"))

        self.conf.add_section("models")
        self.conf.set("models","net","F3net")
        self.conf.set("models","baseline","xception")
        self.conf.set("models","mode","Mix")
        self.conf.set("models","num_classes","1")
        self.conf.set("models","img_width","299")
        self.conf.set("models","img_height","299")
        self.conf.set("models","dropout","0.3")

        self.conf.add_section("train")
        self.conf.set("train","epoch","1000")
        self.conf.set("train","lerning_rate","1e-4")
        self.conf.set("train","batchsize","16")
        self.conf.set("train","loss_fn","CrossEntropy")
        self.conf.set("train","optim","SGD")
        self.conf.set("train","cuda","[0]")
        self.conf.set("train","weightdecay","1e-2")
        self.conf.set("train","amp","1")
        self.conf.set("train","scheduler","WarmupCosineLR")
        self.conf.set("train","tensorboard","1")
        self.conf.set("train","dataparallel","1")

        self.conf.add_section("test")
        self.conf.set("test","batchsize","16")
        self.conf.set("test","cuda","[0]")
        self.conf.set("test","evaluation","[acc]")

        with open(self.path,"w") as f:
            self.conf.write(f)

    def getpath(self):
        return self.path

    def get(self,section,option):
        return self.conf.get(section,option)

    def getint(self,section,option):
        return self.conf.getint(section,option)

    def getfloat(self,section,option):
        return self.conf.getfloat(section,option)