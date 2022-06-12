
# Tran Nhiem -- 2022/05
from typing import Optional, Sequence, List, Any, Callable, List, Type

import torch
import numpy as np
import os 
import torchvision
import pytorch_lightning as pl
from ffcv.fields import IntField, RGBImageField
from ffcv.pipeline.operation import Operation
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder, RandomResizedCropRGBImageDecoder
from ffcv.loader import Loader, OrderOption 
from ffcv.transforms import ToDevice, ToTensor, ToTorchImage, Convert, RandomHorizontalFlip, Cutout, RandomTranslate, NormalizeImage
from ffcv.transforms.common import Squeeze
from pathlib import Path


class FFCV_DataModule(pl.LightningDataModule): 

    def __init__(self,
                data_dir: str, dataset_name: str, img_size: int, batch_size: int, num_workers: int, distributed: bool,   fit_mem: bool, 
              dataset_std: Optional[Sequence[float]] = None, dataset_mean: Optional[Sequence[float]] = None, 
              RandAug: Optional[Sequence[int]] = None,): 

        self.data_dir= data_dir
        self.dataset_name= dataset_name
        self.img_size= img_size
        self.mean= dataset_mean
        self.std= dataset_std
        self.distributed= distributed
        self.fit_mem=fit_mem
        self.batch_size = batch_size, 
        self.num_workers = num_workers, 
        super().__init__()


    def image_piplines(
        min_scale: float=0.08, 
        max_scale: float =1.0, 
        crop_size: int=32, 
        device: int =0, ):

        """args:
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
            crop_size (int, optional): size of the crop will resize the same for all images 
            device (int, optional): gpu device of the current process. Defaults to 0.
        """
        ## Label Data Pipeline
        label_pipeline: List[Operation]= [IntDecoder(), ToTensor(), ToDevice(device, non_blocking=True), Squeeze()]
        ## Image data Pipeline 
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        # ImageNet_MEAN = np.array([0.485, 0.456, 0.406]) * 255
        # ImageNet_STD = np.array([0.229, 0.224, 0.225]) * 255
        CIFAR_MEAN = [125.307, 122.961, 113.8575]
        CIFAR_STD = [51.5865, 50.847, 51.255] 

        image_pipeline.extend([
                                    RandomResizedCropRGBImageDecoder((crop_size, crop_size), scale=(min_scale, max_scale)),
                                    ToTensor(), 
                                    ToDevice(0, non_blocking=True), 
                                    ToTorchImage(),
                                    #Convert(torch.float16),
                                    #transforms.Resize( (self.img_size, self.img_size), InterpolationMode.BICUBIC),
                                    #transforms.Normalize(self.mean, self.std)
                                    #transforms.Normalize(CIFAR_MEAN, CIFAR_STD, type=np.float16)
                                    NormalizeImage(mean=CIFAR_MEAN, std=CIFAR_STD, type=np.float16)
                                    ])

        return {"image": image_pipeline, "label": label_pipeline}



    def ffcv_dataloader(self, mode='train'): 
      
        device = torch.device('cuda')
        ## Lable data Pipeline 
        
        label_pipeline: List[Operation]= [IntDecoder(), ToTensor(),ToDevice(device), Squeeze()]#
        ## Image data Pipeline 
        #image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        image_pipeline: List[Operation] = []

        CIFAR_MEAN =  np.array([125.307, 122.961, 113.8575])
        CIFAR_STD = np.array([51.5865, 50.847, 51.255]) 
        # CIFAR_MEAN = np.array([0.485, 0.456, 0.406]) * 255
        # CIFAR_STD = np.array([0.229, 0.224, 0.225]) * 255
        image_pipeline.extend([
                                RandomResizedCropRGBImageDecoder((self.img_size, self.img_size)),
                                ToTensor(), 
                                ToDevice(device), #test
                                ToTorchImage(),
                                #Convert(torch.float16),
                                #transforms.Resize( (self.img_size, self.img_size), InterpolationMode.BICUBIC),
                                #transforms.Normalize(self.mean, self.std)
                                #transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
                                NormalizeImage(mean=CIFAR_MEAN, std=CIFAR_STD, type=np.float16)
                                ])

        ordering = OrderOption.RANDOM if self.distributed else OrderOption.SEQUENTIAL 
    
        data_path = self.data_path.joinpath(mode)
        files=os.listdir(data_path)[0]
        data_convert_path= Path.joinpath(data_path,str(files))
        if mode=="train": 
            #data_convert_path='/home/rick/efficient_training/efficient_training_neural_Nets/benchmark_training/train/cifa10_train.beton'
            data_convert_path='/home/rick/efficient_training/dataset_benchmark/cifa100_train.beton'
        elif mode=="val": 
            #data_convert_path='/home/rick/efficient_training/efficient_training_neural_Nets/benchmark_training/val/cifar_val.beton'
            data_convert_path='/home/rick/efficient_training/dataset_benchmark/cifa100_val.beton'

        #print(data_convert_path)
        #print('test', data_convert_path[0])
        loaders= Loader(data_convert_path, num_workers= self.num_workers,  batch_size= self.batch_size,#self.batch_size, 
                        distributed=self.distributed, order= ordering,
                         pipelines={'image': image_pipeline, 'label': label_pipeline},  os_cache=self.fit_mem,
                    )

        return loaders


    @property
    def data_path(self):
        return Path(self.data_dir)

    def train_dataloader(self):
        return self.ffcv_dataloader(mode="train")
       
    def val_dataloader(self):
        return self.ffcv_dataloader(mode="val")
       
    def test_dataloader(self):
        return self.ffcv_dataloader(mode="test")
       

