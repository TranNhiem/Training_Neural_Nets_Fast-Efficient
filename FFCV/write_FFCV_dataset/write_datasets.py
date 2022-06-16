
import time 
import os 
import torch 
import numpy as np 
from tqdm import tqdm
from typing import Optional, Sequence, List, Any, Callable, List, Type 
import argparse 


from torchvision import datasets
from torchvision.datasets import ImageFolder
#from torchvision.utils.data import Subset 
from torch.utils.data import Subset
## Import packages from FFCV 
import ffcv
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

def write_ffcv_DATA(dataset_dir: str, dataset_name: str, make_write_path: str, write_file_name: str, max_resolution: int, num_workers: int, 
        chunk_size: int, subset: int, jpeg_quality: float, write_mode: str, compress_probability: float, torchvision_data: bool =False,data_mode: str ='train', 
        ): 
    '''
    args: 
    ""('--torchvision_data', type=str, default=True,)
     ""('--dataset_name', type=str, default='CIFAR10')
     ""('--write_path', type=str, default='/FFCV_dataset/CIFAR10/')
     ""('--data_dir', type=str, default='./CIFAR10/')
     ""('--write_mode', type=str, default='smart', help='Mode: raw, smart or jpg',)
     ""('--img_size', type=int, default=32)
     ""('--num_workers', type=int, default=10)
     ""('--jpeg_quality', type=float, default=90, help="the quality of jpeg Images")
     ""('--chunk_size', type=int, default=100, help="Chunck_size for writing Images")
     ""('--max_resolution', type=int, default=32, help="'Max image side length'")
     ""('--compress_probability', type=float, default=None, help='compress probability')
     This option is mostly useful for users who wish to achieve storage/speed trade-offs between jpg and raw
     ""('--subset', help='How many images to use (-1 for all)', default=-1 )
    '''

    if not os.path.isdir(dataset_dir):
        print("creating : ", dataset_dir)
        os.makedirs(dataset_dir)
    if not os.path.isdir(make_write_path):
        print("creating : ", make_write_path)
        os.makedirs(make_write_path)
    
    write_path= os.path.join(make_write_path,write_file_name)

    if torchvision_data:
        checkKey(torchvision_dataset, dataset_name)
        if data_mode =="train": 
            dataset_=torchvision_dataset[dataset_name](root=dataset_dir, train=True, download=True)
        else: 
            dataset_=torchvision_dataset[dataset_name](root=dataset_dir, train=False, download=True)
        ### Future Continue Extend splitting capability for Splitting Dataset with DistributeSampler

    else: 
        dataset_= ImageFolder(root=dataset_dir)

    if subset >0: dataset_=Subset(dataset_, range(subset))

    writer= DatasetWriter(write_path, {'image': RGBImageField(write_mode=write_mode, max_resolution=max_resolution,
                                                            compress_probability=compress_probability, jpeg_quality=jpeg_quality
                                                            ), 'label': IntField(), },
                            num_workers=num_workers,
                            )

    writer.from_indexed_dataset(dataset_, chunksize=chunk_size)



if __name__=="__main__": 

    parser= argparse.ArgumentParser()
    ## Dataset Define
    parser.add_argument('--torchvision_data', type=str, default=True,)
    parser.add_argument('--dataset_name', type=str, default='CIFAR100')
    parser.add_argument('--make_write_path', type=str, default='/img_data/FFCV_dataset/CIFAR/train/')
    parser.add_argument('--write_file_name', type=str, default='cifar100.beton')
    parser.add_argument('--data_dir', type=str, default='./CIFAR10/')
    parser.add_argument('--write_mode', type=str, default='smart', help='Mode: raw, smart or jpg',)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--jpeg_quality', type=float, default=90, help="the quality of jpeg Images")
    parser.add_argument('--chunk_size', type=int, default=100, help="Chunck_size for writing Images")
    parser.add_argument('--max_resolution', type=int, default=32, help="'Max image side length'")
    parser.add_argument('--compress_probability', type=float, default=None, help='compress probability')
    parser.add_argument('--subset', help='How many images to use (-1 for all)', default=-1 )
    args = parser.parse_args() 

    print("Hoooray~~~ You are writing Test CIFAR10 dataset ")
    write_ffcv_DATA(dataset_dir= args.data_dir, dataset_name=args.dataset_name, make_write_path=args.make_write_path,write_file_name=args.make_write_path, max_resolution=args.max_resolution, num_workers=args.num_workers, 
        chunk_size=args.chunk_size, subset=args.subset, jpeg_quality=args.jpeg_quality, write_mode=args.write_mode, compress_probability=args.compress_probability, torchvision_data=args.orchvision_data,)
    print("Awesome~~~ Dataset completed --> Please check your folder")
    print("Directory as follow: ", args.data_dir)
