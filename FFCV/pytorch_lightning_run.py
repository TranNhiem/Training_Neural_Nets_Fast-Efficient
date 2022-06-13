from py_lightning_modules.ffcv_py_lightning_model import lightning_model
from py_lightning_modules.ffcv_py_lightning_datamodule import FFCV_DataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Trainer, seed_everything
import argparse 


##**************************************************************************
## Setting Some of Hyperparameters 
##**************************************************************************

parser= argparse.ArgumentParser()

## Dataset Define
parser.add_argument('--dataloader_type', type=str, default='ffcv', choices=['ffcv', 'others'], required=False)
parser.add_argument('--dataset_name', type=str, default='CIFAR100')
parser.add_argument('--dataset_mean', type=list, default=[125.307, 122.961, 113.8575])
parser.add_argument('--dataset_std', type=list, default=[51.5865, 50.847, 51.255] )
parser.add_argument('--dataset_length', type=int, default=50000, required=False)

parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--data_path', type=str, default='/img_data/FFCV_dataset/CIFAR/')
## For ffcv dataset directory should be specify the correct name 
parser.add_argument('--data_train_dir', type=str, default='/img_data/FFCV_dataset/CIFAR/train/cifar100.beton')
parser.add_argument('--data_val_dir', type=str, default='/img_data/FFCV_dataset/CIFAR/val/cifar100.beton')

parser.add_argument('--img_size', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=146)

## Training Hyperparameters
parser.add_argument('--seed', type=int, default=100,help='random seed')
parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'resnet50'])
parser.add_argument('--width_mult', type=float, default=1)
parser.add_argument('--m_transfer', type=bool, default=False)
parser.add_argument('--metric', type=str, default='Accuracy')

parser.add_argument('--lr', default=0.003719, type=float, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--optimizer_type', default='Adam', )#choices=['SGD', 'Adam','AdamW' 'MuSGD', 'MuAdam', 'MuAdamW'])
parser.add_argument('--momentum', type=float, default=0.9)

parser.add_argument("--lr_scheduler",type= str, default="ConsineAnl_warmup",)# choices=['step', 'reduce_plateau', 'linear', 'cosineAnnealing', 'ConsineAnl_warmup'], help="learning rate schedule")
parser.add_argument("--steps", type= list, default=[30, 50, 80, 120],  help="learning rate schedule")#[30,50,90, 120]
parser.add_argument('--epochs', type=int, default=129)
parser.add_argument('--num_workers', type=float, default=8)
parser.add_argument('--gpus', type=list, default=[1])



## Visualization and Debug Setting
parser.add_argument("--method", type=str, default="FFCV_py_lightning")
parser.add_argument("--job_type", type=str, default="Faster Training")
args = parser.parse_args()


class Dataset_Trainer():

    def __init__(self, **kwargs):         
        ## Dataloader arguments    
        self.dataset_name = args.dataset_name
        self.data_length = args.dataset_length
        self.num_classes=args.num_classes
        self.data_mean=args.dataset_mean
        self.data_std= args.dataset_std
        self.data_train_dir= args.data_train_dir
        self.data_val_dir= args.data_val_dir
        self.img_size= args.img_size
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        ## Training Hyperparameters Arugments
        self.arch=args.arch
        self.arch_width_mu=args.width_mult
        self.gpus = args.gpus
        self.max_epochs=args.epochs 

        #self.optimizer_type=args.optimizer_type
        self.momentum= args.momentum
        self.lr = args.lr
        self.weight_decay = args.weight_decay   
        self.lr_decay_steps=args.steps
        self.scheduler = args.lr_scheduler
        self.optimizer_type=args.optimizer_type
        self.metric=args.metric
        
        
        self.dataloader = FFCV_DataModule(
            data_train_dir=self.data_train_dir,
            data_val_dir=self.data_val_dir,
            data_path=args.data_path,
            dataset_name = self.dataset_name,
            dataset_std= self.data_std,
            dataset_mean= self.data_mean, 
            dataloader_type=args.dataloader_type,
            img_size = self.img_size,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            distributed=True, # Using DataParallel for multiple GPUs
            fit_mem=False, # Cache all dataset into memory
        )

        self.pl_model = lightning_model(
            backbone_architecture=self.arch,
            arch_width_mu=self.arch_width_mu,
            m_transfer= args.m_transfer,
            num_classes=self.num_classes,
            batch_size = self.batch_size,
            data_length=self.data_length,
            epochs=self.max_epochs, 
            lr = self.lr,
            optimizer_type=self.optimizer_type,
            momentum=self.momentum,
            weight_decay = self.weight_decay,
            scheduler = self.scheduler,
            lr_decay_steps = self.lr_decay_steps,
            metric=self.metric
        )

        self.wandb_logger = WandbLogger(
            name = f'{args.method} {self.dataset_name} arch={self.arch} optimizer={self.optimizer_type} lr={self.lr} lr_schedule={self.scheduler} wd={self.weight_decay} batch_size {self.batch_size}',
            project = 'training_efficient',
            entity = 'mlbrl',
            group = self.dataset_name,
            job_type = args.job_type,
            offline = False,
        )
        callbacks_list=[]
        self.wandb_logger.watch(self.pl_model, log="gradients",  log_freq = 50)
        self.wandb_logger.log_hyperparams(args)
        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="step")
        # callbacks_list.append(lr_monitor)
        

        self.trainer = Trainer(
            #fast_dev_run=True,
            accelerator = 'gpu',
            gpus = self.gpus,
            logger = self.wandb_logger,
            #max_steps = self.max_steps,
            max_epochs=self.max_epochs,
            strategy = 'ddp',
            precision=16,

            #callbacks= callbacks_list,
            #replace_sampler_ddp=True,
        )

    def run(self):
        #seed_everything(10)
        print(f"Start Training : {self.dataset_name}")
        # for x, y in self.dataloader.train_dataloader(): 
        #     print(x.shape)
        self.trainer.fit(self.pl_model, self.dataloader)

        print("End Training")

run_experiment=Dataset_Trainer()
run_experiment.run()