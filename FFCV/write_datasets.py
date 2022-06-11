
import time 
import os 
import torch 
import numpy as np 
from tqdm import tqdm
from typing import Optional, Sequence, List, Any, Callable, List, Type 
import argparse 


from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision.utils.data import Subset 
## Import packages from FFCV 
from ffcv.writer import DatasetWriter 
from ffcv.fields import IntField, RGBImageField 



def write_dataset(): 
    pass 
def checkKey(dict, key): 
    """Checking Your input dataset Wherether existing in current support torch_vision dataset"""
    if key in dict.keys(): 
        print(f"Your {key} dataset is current support in Torchvision build available dataset")
    else: 
        print(f"Your {key} dataset is current not support :( please check the name Carefully !!")

torchvision_dataset={
    "CIFAR100": datasets.CIFAR100, 
    "CIFAR10": datasets.CIFAR10,

}

def main(dataset_dir: str, dataset_name: str, write_path: str, max_resolution: int, num_workers: int, 
        chunk_size: int, subset: int, jpeg_quality: float, write_mode: str, compress_probability: float, torchvision_data: bool =False,
        ): 
    '''
    args: 
    ""('--torchvision_data', type=str, default=True,)
     ""('--dataset_name', type=str, default='CIFAR10')
     ""('--write_path', type=str, default='./FFCV_dataset/CIFAR10/')
     ""('--data_dir', type=str, default='./CIFAR10/')
     ""('--write_mode', type=str, default='smart', help='Mode: raw, smart or jpg',)
     ""('--img_size', type=int, default=32)
     ""('--num_workers', type=int, default=10)
     ""('--jpeg_quality', type=float, default=90, help="the quality of jpeg Images")
     ""('--chunk_size', type=int, default=100, help="Chunck_size for writing Images")
     ""('--max_resolution', type=int, default=32, help="'Max image side length'")
     ""('--compress_probability', type=float, default=None, help='compress probability')
     ""('--subset', help='How many images to use (-1 for all)', default=-1 )
    '''

    if not os.path.isdir(dataset_dir):
        print("creat : ", dataset_dir)
        os.makedirs(dataset_dir)
    if not os.path.isdir(write_path):
        print("creat : ", write_path)
        os.makedirs(write_path)
    
    if torchvision_data:
        checkKey(torchvision_dataset, dataset_name)
        dataset_=torchvision_dataset['dataset_name'](root=dataset_dir, )
    else: 
        dataset_= ImageFolder(root=dataset_dir)

    if subset >0: dataset_=Subset(dataset_, range(subset))

    writer= DatasetWriter(write_path, {'image': RGBImageField(write_mode=write_mode, max_resolution=max_resolution,
                                                            compress_probability=compress_probability, jpeg_quality=jpeg_quality
                                                            ), 'label': IntField(), },
                            num_workers=num_workers,
                            )

    writer.from_indexed_dataset(dataset_, chunk_size=chunk_size)



if __name__=="__main__": 
    parser= argparse.ArgumentParser()
    ## Dataset Define
     ""('--torchvision_data', type=str, default=True,)
     ""('--dataset_name', type=str, default='CIFAR10')
     ""('--write_path', type=str, default='./FFCV_dataset/CIFAR10/')
     ""('--data_dir', type=str, default='./CIFAR10/')
     ""('--write_mode', type=str, default='smart', help='Mode: raw, smart or jpg',)
     ""('--img_size', type=int, default=32)
     ""('--num_workers', type=int, default=10)
     ""('--jpeg_quality', type=float, default=90, help="the quality of jpeg Images")
     ""('--chunk_size', type=int, default=100, help="Chunck_size for writing Images")
     ""('--max_resolution', type=int, default=32, help="'Max image side length'")
     ""('--compress_probability', type=float, default=None, help='compress probability')
     ""('--subset', help='How many images to use (-1 for all)', default=-1 )
    args = parser.parse_args() 

    print("Hoooray~~~ You are writing Test CIFAR10 dataset ")
    main(dataset_dir= args.data_dir, dataset_name=args.dataset_name, write_path=args.write_path, max_resolution=args.max_resolution, num_workers=args.num_workers, 
        chunk_size=args.chunk_size, subset=args.subset, jpeg_quality=args.jpeg_quality, write_mode=args.write_mode, compress_probability=args.compress_probability, torchvision_data=args.orchvision_data,)
    print("Awesome~~~ Dataset completed --> Please check your folder")
    print("Directory as follow: ", args.data_dir)
