from torch.utils.data import Dataset
import os
from PIL import Image

class ImageFolderDataset(Dataset):
    def __init__(self,root:str,transform=None):
        super(ImageFolderDataset,self).__init__()
        self.root=root
        self.tranform=transform
        self.files=list(os.listdir(root))
        self.files=[p for p in self.files if p.endswith('.jpg','.png','.jpeg')]


    def _len_(self):
        return len(self.files)

    def _getitem_(self,idx):
        image_path=os.path.join(self.root,self.files[idx])
        image=Image.open(image_path)

        if self.tranform:
            image=self.tranform(image)

        return image