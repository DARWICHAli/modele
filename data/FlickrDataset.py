import torch 
from PIL import Image 
import os


class FlickrDataset():
    def __init__(self, root_dir, imgs, labels, vocab, transforms, imgs_idx):
        self.root_dir = root_dir
        self.transforms = transforms
        self.imgs = imgs
        self.captions = labels
        self.imgs_idx = imgs_idx
        self.vocab = vocab
        

        

    def __len__(self):
        return (len(self.imgs))

    
    def __getitem__(self, index):
        
        caption = self.captions[index]
        img_id = self.imgs[index]

        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return (img, torch.tensor(numericalized_caption), torch.tensor([self.imgs_idx[img_id]]))
        

        
