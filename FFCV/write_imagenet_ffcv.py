from write_FFCV_dataset.write_datasets import write_ffcv_DATA
import argparse 

parser= argparse.ArgumentParser()
## Dataset Define
parser.add_argument('--torchvision_data', type=str, default= True,)
parser.add_argument('--dataset_name', type=str, default='CIFAR100')
parser.add_argument('--make_write_path', type=str, default='/img_data/FFCV_dataset/CIFAR100/train/')
parser.add_argument('--write_file_name', type=str, default='CIFAR_train.beton')
parser.add_argument('--data_dir', type=str, default='/img_data/train/')
parser.add_argument('--write_mode', type=str, default='jpg', help='Mode: raw, smart or jpg',)
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--num_workers', type=int, default=80)
parser.add_argument('--jpeg_quality', type=float, default=90, help="the quality of jpeg Images")
parser.add_argument('--chunk_size', type=int, default=300, help="Chunck_size for writing Images")
parser.add_argument('--max_resolution', type=int, default=32, help="'Max image side length'")
parser.add_argument('--compress_probability', type=float, default=None, help='compress probability')
parser.add_argument('--subset', help='How many images to use (-1 for all)', default=0 )
args = parser.parse_args() 

write_ffcv_DATA(dataset_dir= args.data_dir, dataset_name=args.dataset_name, make_write_path=args.make_write_path,write_file_name=args.write_file_name, max_resolution=args.max_resolution, num_workers=args.num_workers, 
        chunk_size=args.chunk_size, subset=args.subset, jpeg_quality=args.jpeg_quality, write_mode=args.write_mode, compress_probability=args.compress_probability, torchvision_data=args.torchvision_data,data_mode='train')