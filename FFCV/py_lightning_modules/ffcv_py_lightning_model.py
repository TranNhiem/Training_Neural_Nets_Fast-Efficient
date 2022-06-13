
## Tran Nhiem -- 2022/05 
from typing import Optional, Sequence 
from torch.optim import SGD, Adam, AdamW
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, LinearLR, CosineAnnealingLR,CosineAnnealingWarmRestarts
import pytorch_lightning as pl
import torch.nn.functional as F 
from torchmetrics import Accuracy 
from neural_nets_architecture.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
#from efficient_training_neural_Nets.neural_nets_architecture.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

def checkModel(dict, key): 
    """Checking Available ResNet Models"""
    if key in dict.keys(): 
        print(f"Your {key} is available")
    else: 
        print(f"Your {key}  is current not support :( please check the name Carefully !!")

def checkOptimizer(dict, key): 
    """Checking Available Optimizer"""
    if key in dict.keys(): 
        print(f"Your {key} is available")
    else: 
        print(f"Your {key}  is current not support :( please check the name Carefully !!")



model_available={
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnet152': ResNet152,
}

class lightning_model(pl.LightningModule): 
    def __init__(self, 
        backbone_architecture: str, 
        batch_size: int ,  
        lr: float, 
        scheduler: str,
        weight_decay: float, 
        num_classes: int , 
        arch_width_mu: int or float, 
        optimizer_type: str,
        momentum: float,
        epochs: int, 
        data_length: int, 
        lr_decay_steps: Optional[Sequence[int]] = None,
        task: Optional[str]="classification",
        metric: str = "Accuracy", 
 
        **kwargs
    ):
        super().__init__()
        self.backbone_architecture= backbone_architecture
        self.width_mult=arch_width_mu
        self.batch_size= batch_size 
        self.optimizer_type= optimizer_type
        self.momentum=momentum
        self.num_classes= num_classes
        self.lr = lr
        self.epochs=epochs
        self.data_length=data_length
        self.weight_decay = weight_decay
        self.lr_decay_steps = lr_decay_steps
        self.task = task
        self.scheduler = scheduler
        print(self.lr)
        self.__build_model()
        self.accuracy_1= Accuracy()
        self.accuracy_5= Accuracy(top_k=5)
        self.metric=metric
        self.criterion = nn.CrossEntropyLoss()
    
    
    @property
    def optimizer_config(self): 
        optimizer_available={
            'SGD':  SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay = self.weight_decay, nesterov=True), 
            'Adam': Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay= self.weight_decay),
            'AdamW': AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay= self.weight_decay),
        }
        return optimizer_available

    def __build_model(self): 
        checkModel(model_available, self.backbone_architecture)
        self.model=model_available[self.backbone_architecture](num_classes=self.num_classes, wm=self.width_mult, m_transfer=False )

    def forward(self, x): 
        x=self.model(x)
        return x

    def training_step(self, batch, batch_idx): 
        x, y = batch
        y_logits=self.forward(x)
        train_loss=F.cross_entropy(y_logits, y)
 
        if self.metric == 'Accuracy':   
            acc1= self.accuracy_1(y_logits, y)
            acc5 = self.accuracy_5(y_logits, y)
            log = {"train_loss": train_loss, "train_acc1": acc1, "train_acc5": acc5}
        else:
            raise ValueError("The metric is not support yet")
        
        self.log_dict(log, on_epoch=True, sync_dist=True)
        return train_loss
        
    def validation_step(self, batch, batch_idx): 
        x, y = batch
       
        batch_size=x.size(0)
        y_logits=self.forward(x)
        val_loss=F.cross_entropy(y_logits, y)
        if self.metric == 'Accuracy':   
            acc1= self.accuracy_1(y_logits, y)
            acc5 = self.accuracy_5(y_logits, y)
            #result = {"batch_size": batch_size,"val_loss": val_loss, "val_acc1": acc1, "val_acc5": acc5}
            result = {"val_loss": val_loss, "val_acc1": acc1, "val_acc5": acc5}
        else:
            raise ValueError("The metric is not support yet")
        self.log_dict(result,on_epoch=True, sync_dist=True)
        
        return val_loss
 
    def test_step(self, batch, batch_idx): 
        x, y = batch
        batch_size=x.size(0)
        y_logits=self.forward(x)
        test_loss=F.cross_entropy(y_logits, y)
        #train_loss = F.nll_loss(y_logits, y)

        if self.metric == 'Accuracy':   
            acc1= self.accuracy_1(y_logits, y)
            acc5 = self.accuracy_5(y_logits, y)
            result = {"test_loss": test_loss, "test_acc1": acc1, "test_acc5": acc5}
            
            #result = {"batch_size": batch_size,"test_loss": val_loss, "test_acc1": acc1, "test_acc5": acc5}
        else:
            raise ValueError("The metric is not support yet")
        
        self.log_dict(result,on_epoch=True, sync_dist=True)
        
        return test_loss


    def configure_optimizers(self):

        optimizer_available=self.optimizer_config
        checkOptimizer(optimizer_available, self.optimizer_type)
        optimizer=optimizer_available[self.optimizer_type]
      
       ## Configure Learning Rate Schedule
        if self.scheduler == 'step':
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps, gamma=0.5)
            return [optimizer], [scheduler]
        elif self.scheduler == 'reduce_plateau':
            #scheduler = ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
            scheduler = {"scheduler": ReduceLROnPlateau(optimizer, patience=8, factor=0.5,),  "monitor": "val_loss"}
            return [optimizer], [scheduler]
        elif self.scheduler=="linear": 
            scheduler = LinearLR( optimizer, start_factor=0.5, total_iters=self.epochs/2)
            return [optimizer], [scheduler]
        elif self.scheduler=="cosineAnnealing": 
            scheduler = CosineAnnealingLR(optimizer, eta_min=1e-8,T_max=int((self.data_length/self.batch_size)*self.epochs) )
            return [optimizer], [scheduler]
        elif self.scheduler=="ConsineAnl_warmup": 
            scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=int(((self.data_length/self.batch_size)*self.epochs)/2), T_mult=1, eta_min=1e-8)
            return [optimizer], [scheduler]
        else: 
            print('you are not implementing any Learning schedule')
            return optimizer

    def __weighted_mean(self, outputs, key, batch_size_key):
        value=0
        n=0
        for out in outputs: 
            value += out[batch_size_key] +out[key]
            n+= out[batch_size_key]
        value=value/n 
        return value.squeeze(0)

if __name__=="__main__": 
    print("Hoooray ~ You are testing the lightning Module ~")
    print("Oh wait you are not define any testing module yet")
