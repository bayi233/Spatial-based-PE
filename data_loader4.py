import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root  # 'E:/datasets/COCO/images/resized2014'
        self.coco = COCO(json)  # #'E:/datasets/COCO/annotations/captions_train2014.josn'
        self.ids = list(self.coco.anns.keys())  # list of ids
        self.id=[]
        for item in self.ids:
            img_id = self.coco.anns[item]['image_id']
            if img_id not in self.id:
                self.id.append(img_id)
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        # ann_id = self.ids[index]
        img_id=self.id[index]
        # caption = coco.anns[ann_id]['caption']  # get the sentence or caption of each annotation
        # img_id = coco.anns[ann_id]['image_id']  # get the image_id of each annotation by ann_id,e.g. img_id = 0000002742

        path = coco.loadImgs(img_id)[0]['file_name']  # get the image_filename from
        # the dict coco.loadImgs(img_id)[0] by img_id
        # conjoin the root directory of img and img_name, to open the image and convert to 'RGB'
        image = Image.open(os.path.join(self.root, path)).convert('RGB')  # get the image by image_path
        if self.transform is not None:
            image = self.transform(image)  # convert image to pytorch tensor image_data,include nomoralize

        # Convert caption (string) to word ids.
        # tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        # caption = []
        # caption.append(vocab('<start>'))
        # caption.extend([vocab(token) for token in tokens])
        # caption.append(vocab('<end>'))
        # target = torch.Tensor(caption)  # convert caption's ids to pytorch tensor

        return image, img_id   # target,

    def __len__(self):
        return len(self.id)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    # data.sort(key=lambda x: len(x[1]), reverse=True)  ##??????tuple???????????????????????????caption?????????????????????

    images,img_id = zip(*data)  ##???????????????[(image1,caption1),(image2,caption2),...]   ???????????????????????????[(image1,image2,....),
    ##(caption1,caption2.....)]

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)  ###?????????0????????????
    # img_ids=torch.tensor(img_id)   ###
    img_ids=torch.tensor(img_id)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    # lengths = [len(cap) for cap in captions]  ##??????????????????????????????????????????caption?????????
    # targets = torch.zeros(len(captions), max(lengths)).long()  ##????????????len(captions),max(lenths) ???????????????0???long tensor
    # for i, cap in enumerate(captions):  ##i?????????   cap:caption??????
    #     end = lengths[i]  ##end =??????caption?????????
    #     targets[i, :end] = cap[:end]  ##????????????caption
    return images, img_ids   #targets, lengths,


def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader