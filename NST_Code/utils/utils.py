from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

class ImageFolderDataset(Dataset):
    def __init__(self,root:str,transform=None):
        super(ImageFolderDataset,self).__init__()
        self.root=root
        self.transform=transform
        self.files=list(os.listdir(root))
        self.files=[p for p in self.files if p.endswith(('.jpg','.png','.jpeg'))]


    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        image_path=os.path.join(self.root,self.files[idx])
        image=Image.open(image_path)

        if self.transform:
            image=self.transform(image)

        return image

def get_transform(size,crop,final_size):

    transform_list = []

    transform_list.append(
        transforms.Resize((final_size, final_size))
    )

    transform_list.append(
        transforms.ToTensor()
    )

    return transforms.Compose(transform_list)

def adaptive_instance_normalization(content_feat,style_feat):
    pass

def calc_mean_std(feat,eps=1e-5):
    pass

