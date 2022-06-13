
# Tran Nhiem -- 2022/05
from typing import Optional, Sequence, List, Any, Callable, List, Type

import torch
import numpy as np
import os 
import torchvision
from torchvision import transforms as transform_lib
import pytorch_lightning as pl
from ffcv.fields import IntField, RGBImageField
from ffcv.pipeline.operation import Operation
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder, RandomResizedCropRGBImageDecoder
from ffcv.loader import Loader, OrderOption 
from ffcv.transforms import ToDevice, ToTensor, ToTorchImage, Convert, RandomHorizontalFlip, Cutout, RandomTranslate, NormalizeImage
from ffcv.transforms.common import Squeeze
from pathlib import Path
from torch.utils.data import Subset

torchvision_dataset={
    "CIFAR100": torchvision.datasets.CIFAR100, 
    "CIFAR10": torchvision.datasets.CIFAR10,
}
def checkData(dict, key): 
    """Checking Your input dataset Wherether existing in current support torch_vision dataset"""
    if key in dict.keys(): 
        print(f"Your {key} dataset is current support in Torchvision build available dataset")
        return True
    else: 
        print(f"Your {key} dataset is current not support :( please check the name Carefully !!")
        return False



class FFCV_DataModule(pl.LightningDataModule): 

    def __init__(self,
                dataset_name: str, data_train_dir: str,data_val_dir: str,  data_path: str,  img_size: int, batch_size: int, num_workers: int, distributed: bool,   fit_mem: bool, 
                ## Arguments for data normalization  
                dataloader_type: str, subset: Optional[int]= 0,  dataset_std: Optional[Sequence[float]] = None, dataset_mean: Optional[Sequence[float]] = None, 
                ##Arguments for data Augmentation 
                crop_size: Optional[int]= None, min_scale: Optional[float] =0.08,  max_scale: Optional[float] =1.0,
                RandAug: Optional[Sequence[int]] = None,): 

        self.dataset_name= dataset_name
        self.data_train_dir= data_train_dir
        self.data_val_dir=data_val_dir
        self.data_path=data_path
        self.subset=subset
        self.dataloader_type=dataloader_type
        ## Dataset Normalization
        if (dataset_std != None) & (dataset_mean!=None): 
            print("Using Custome dataset mean & std")
            self.std=np.array(dataset_std)
            self.mean=np.array(dataset_mean)
        else: 
            print("Using ImageNet mean & std statistic")
            self.mean= np.array([0.485, 0.456, 0.406]) * 255
            self.std = np.array([0.229, 0.224, 0.225]) * 255

        self.img_size= img_size
        if crop_size !=None:
            self.crop_size=crop_size
        else:
            self.crop_size=img_size
        self.min_scale=min_scale
        self.max_scale=max_scale

        self.distributed= distributed
        self.fit_mem=fit_mem
        self.batch_size = int(batch_size), 
        self.num_workers = int(num_workers), 
        super().__init__()

        self.dataset_transforms = {
            
            "train": self.train_transform,
            "val": self.val_transform,
            "test": self.val_transform,
        }
    
    def image_piplines(self, ):

        """args:
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
            crop_size (int, optional): size of the crop will resize the same for all images 
            device (int, optional): gpu device of the current process. Defaults to 0.
        """
        device = torch.device('cuda')
        ## Label Data Pipeline
        label_pipeline: List[Operation]= [IntDecoder(), ToTensor(), ToDevice(device, non_blocking=True), Squeeze()]
        ## Image data Pipeline 
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
     
        image_pipeline.extend([

                                    #RandomResizedCropRGBImageDecoder((self.crop_size, self.crop_size), scale=(self.min_scale, self.max_scale)),
                                    ToTensor(), 
                                    ToDevice(device, non_blocking=True), 
                                    ToTorchImage(),
                                    #Convert(torch.float16),
                                    #transforms.Resize( (self.img_size, self.img_size), InterpolationMode.BICUBIC),
                                    #transforms.Normalize(self.mean, self.std)
                                    #transforms.Normalize(CIFAR_MEAN, CIFAR_STD, type=np.float16)
                                    NormalizeImage(mean=self.mean, std=self.std, type=np.float16)
                                    ])

        return {"image": image_pipeline, "label": label_pipeline}

    def ffcv_dataloader(self, mode='train'): 
      
        ## Sampling Data 
        ordering = OrderOption.RANDOM if self.distributed else OrderOption.SEQUENTIAL 

        if mode=="train": 
            #data_convert_path='/home/rick/efficient_training/efficient_training_neural_Nets/benchmark_training/train/cifa10_train.beton'
            data_convert_path=self.data_train_dir
        elif mode=="val": 
            #data_convert_path='/home/rick/efficient_training/efficient_training_neural_Nets/benchmark_training/val/cifar_val.beton'
            data_convert_path=self.data_val_dir

        #Get image and Label Pipelien 
        image_label_pipeline=self.image_piplines()

        loaders= Loader(data_convert_path, num_workers= self.num_workers,  batch_size= self.batch_size,
                            distributed=self.distributed, order= ordering,
                         pipelines=image_label_pipeline,  os_cache=self.fit_mem,)
        
        return loaders

    def standard_pytorch(self, mode="train"): 
        
        if not os.path.isdir(self.data_path):
            print("creating : ", self.data_path)
            os.makedirs(self.data_path)

        if checkData(torchvision_dataset, self.dataset_name):
            if mode=="train":    
                dataset_=torchvision_dataset[self.dataset_name](root=self.data_path, train=True, download=True, transform=self.dataset_transforms[mode])
            else: 
                dataset_=torchvision_dataset[self.dataset_name](root=self.data_path, train=False, download=True, transform=self.dataset_transforms[mode])
        ### Future Continue Extend splitting capability for Splitting Dataset with DistributeSampler)
        else: 
            
            data_path=os.path.join(self.data_path, mode)
            dataset_= ImageFolder(data_path, transforms=self.dataset_transforms[mode])
        
        if self.subset >0: dataset_=Subset(dataset_, range(subset))
        

        return torch.utils.data.DataLoader(
            dataset=dataset_,
            batch_size= self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )


    def train_dataloader(self):
        if self.dataloader_type=='ffcv':
            print('You are using ffcv dataloader')
            return self.ffcv_dataloader(mode="train")
        else: 
            print('You are using standard dataloader')
            return self.standard_pytorch(mode='train')
       
    def val_dataloader(self):
        if self.dataloader_type=='ffcv':
            print('You are using ffcv dataloader')
            return self.ffcv_dataloader(mode="val")
        else: 
            return self.standard_pytorch(mode='val')
       
       
    def test_dataloader(self):

        if self.dataloader_type=='ffcv':
            return self.ffcv_dataloader(mode="test")
        else: 
            return self.standard_pytorch(mode='test')
       
    
    @property 
    def train_transform(self) -> Callable:
        """The standard imagenet transforms.
        """
        preprocessing = transform_lib.Compose(
            [
                transform_lib.RandomResizedCrop(self.crop_size),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.ToTensor(),
                transform_lib.Normalize(mean=self.mean/255.0, std=self.std/255.0),
            ]
        )

        return preprocessing
    
    @property 
    def val_transform(self) -> Callable:
        """The standard imagenet transforms for validation.
        """

        preprocessing = transform_lib.Compose(
            [
                transform_lib.Resize(self.img_size + 32),
                transform_lib.CenterCrop(self.img_size),
                transform_lib.ToTensor(),
                transform_lib.Normalize(mean=self.mean/255.0, std=self.std/255.0),

            ]
        )
        return preprocessing
