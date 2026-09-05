import argparse
import torch
from pathlib import Path
from utils.utils import *
def parse_arguments():
    parser=argparse.ArgumentParser()

    parser.add_argument('--content_dir',type=str,default=r'D:\Projects\Neural Style Transfer\content_data',
                        help='Location of content dataset')
    parser.add_argument('--style_dir',type=str,default=r'D:\Projects\Neural Style Transfer\style_data',
                            help='Location of style dataset')
    parser.add_argument('--vgg',type=str,default=r'D:\Projects\Neural Style Transfer\vgg_normalised.pth',
                            help='Location of pre-trained VGG')
    parser.add_argument('--experiment',type=str,default=r'experiment1',
                            help='Name of experiment')


    return parser.parse_args()

def main():
    args=parse_arguments()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saved_dir=Path('experiment')/args.experiment
    saved_dir.mkdir(exist_ok=True,parents=True)

    with open(saved_dir/'args.txt','w') as args_file:
        for key,value in vars(args).items():
            args_file.write(f'{key}:{value}\n')


    content_transform=None
    style_transform=None



    content_dataset=ImageFolderDataset(args.content_dir,content_transform)
    style_dataset=ImageFolderDataset(args.content_dir,style_transform)


if __name__=='__main__':
    main()

