import torch 
from PIL import Image 
import os


class FlickrDataset():
    def __init__(self, root_dir, imgs, labels, vocab, transforms, imgs_idx):
        self.root_dir = root_dir
        self.transforms = transforms
        self.imgs_idx = imgs_idx
        self.imgs = imgs
        self.captions = labels
        self.vocab = vocab
        self.len = self.imgs
        
    def __len__(self):
        return self.len
 
    def __getitem__(self, index):
        
        img_id = self.imgs[index]
        captions = self.captions[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)
            
        return img, captions
        

        
