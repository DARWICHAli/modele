import torch 
from torch.nn.utils.rnn import pad_sequence

class MyCollate:
    
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim = 0)
        img_ids = [item[2].unsqueeze(0) for item in batch]
        img_ids = torch.cat(img_ids, dim = 0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first = False, padding_value = self.pad_idx)

        return imgs, targets, img_ids
