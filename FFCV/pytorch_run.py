"""
Fast training script for CIFAR-10 using FFCV.
For tutorial, see https://docs.ffcv.io/ffcv_examples/cifar10.html.

First, from the same directory, run:

    `python write_datasets.py --data.train_dataset [TRAIN_PATH] \
                              --data.val_dataset [VAL_PATH]`

to generate the FFCV-formatted versions of CIFAR.

Then, simply run this to train models with default hyperparameters:

    `python train_cifar.py --config-file default_config.yaml`

You can override arguments as follows:

    `python train_cifar.py --config-file default_config.yaml \
                           --training.lr 0.2 --training.num_workers 4 ... [etc]`

or by using a different config file.
"""
from argparse import ArgumentParser
from typing import List
import time
import numpy as np
from tqdm import tqdm

import torch as ch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, lr_scheduler, Adam
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, LinearLR, CosineAnnealingLR,CosineAnnealingWarmRestarts

import torchvision

from fastargs import get_current_config, Param, Section
from fastargs.decorators import param
from fastargs.validation import And, OneOf

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter
from neural_nets_architecture.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, resnet18, resnet50
import wandb 

Section('training', 'Hyperparameters').params(
    lr=Param(float, 'The learning rate to use',default=0.003791, required=False),
    epochs=Param(int, 'Number of epochs to run for',default=100,  required=False),
    lr_peak_epoch=Param(int, 'Peak epoch for cyclic lr',default=5, required=False),
    data_length=Param(int, 'number of images of the dataset',default=50000, required=False),
    batch_size=Param(int, 'Batch size', default=512),
    momentum=Param(float, 'Momentum for SGD', default=0.9),
    weight_decay=Param(float, 'l2 weight decay', default=5e-4),
    label_smoothing=Param(float, 'Value of label smoothing', default=0.1),
    num_workers=Param(int, 'The number of workers', default=32),
    num_classes=Param(int, 'number of class', default=100),
    lr_tta=Param(bool, 'Test time augmentation by averaging with horizontally flipped version', default=True)
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', default= '/img_data/FFCV_dataset/CIFAR/train/cifar100.beton',required=False),
    val_dataset=Param(str, '.dat file to use for validation', default='/img_data/FFCV_dataset/CIFAR/val/cifar100.beton', required=False),
)

@param('data.train_dataset')
@param('data.val_dataset')
@param('training.batch_size')
@param('training.num_workers')

def make_dataloaders(train_dataset=None, val_dataset=None, batch_size=None, num_workers=None):
    paths = {
        'train': train_dataset,
        'test': val_dataset
    }

    start_time = time.time()
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    loaders = {}

    for name in ['train', 'test']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda'), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                # RandomTranslate(padding=2, fill=tuple(map(int, CIFAR_MEAN))),
                # Cutout(4, tuple(map(int, CIFAR_MEAN))),
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice('cuda', non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        
        ordering = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(paths[name], batch_size=batch_size, num_workers=num_workers,
                               order=ordering, drop_last=(name == 'train'),
                               pipelines={'image': image_pipeline, 'label': label_pipeline})

    return loaders, start_time

# Model ResNet 18
device = ch.device(0)
model = resnet18(num_classes=100,wm=1, m_transfer=False )
#model=model.to(device)
model=ch.nn.DataParallel(model).to(device)

##************************************************
## Initialization Wandb
##************************************************
wandb.init( 
            name = 'FFCV_Py_Cifar100_Res18_2GPU_A100',
            project = 'training_efficient',
            entity = 'mlbrl',
            group = 'CIFAR100',
            job_type = "Faster Training",
            
            )


@param('training.lr')
@param('training.batch_size')
@param('training.data_length')
@param('training.epochs')
@param('training.momentum')
@param('training.weight_decay')
@param('training.label_smoothing')
@param('training.lr_peak_epoch')
def train(model, loaders, lr=None, epochs=None, batch_size=None, data_length=None, label_smoothing=None,
          momentum=None, weight_decay=None, lr_peak_epoch=None):
    #opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    opt = Adam(model.parameters(), lr=lr,  weight_decay=weight_decay)
    iters_per_epoch = len(loaders['train'])
    # Cyclic LR with single triangle
    lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                            [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                            [0, 1, 0])
    #scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scheduler = CosineAnnealingWarmRestarts(opt,T_0=int(((data_length/batch_size)*epochs)/2), T_mult=1, eta_min=1e-8)
        
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)

    for iter_ in range(epochs):
        wandb.log({"epoch": iter_} )
        for ims, labs in tqdm(loaders['train']):
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims)
                loss = loss_fn(out, labs)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            wandb.log({'train_loss': loss})
       

@param('training.lr_tta')
def evaluate(model, loaders, lr_tta=False):
    model.eval()
    with ch.no_grad():
        for name in ['train', 'test']:
            total_correct, total_num = 0., 0.
            for ims, labs in tqdm(loaders[name]):
                with autocast():
                    out = model(ims)
                    if lr_tta:
                        out += model(ims.flip(-1))
                    total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                    total_num += ims.shape[0]
            print(f'{name} accuracy: {total_correct / total_num * 100:.1f}%')
            wandb.log({'accuracy'+str(name): total_correct / total_num * 100})

if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='Fast CIFAR-100 training')
    config.augment_argparse(parser)
    # Also loads from args.config_path if provided
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    loaders, start_time = make_dataloaders()
    model = model
    train(model, loaders)
    print(f'Total time: {time.time() - start_time:.5f}')
    evaluate(model, loaders)
