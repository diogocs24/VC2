import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class ChessReD2KDataset(Dataset):
    def __init__(self, images_dir, image_infos, pieces, transform=None):
        self.images_dir = images_dir
        self.image_infos = image_infos
        self.pieces = pieces
        self.transform = transform

        self.pieces_per_image = {}
        for piece in pieces:
            img_id = piece['image_id']
            if img_id not in self.pieces_per_image:
                self.pieces_per_image[img_id] = []
            self.pieces_per_image[img_id].append(piece)

    def __len__(self):
        return len(self.image_infos)

    def __getitem__(self, idx):
        img_info = self.image_infos[idx]
        img_path = os.path.join(self.images_dir, img_info['path'])
        image = Image.open(img_path).convert("RGB")

        pieces = self.pieces_per_image.get(img_info['id'], [])

        boxes = []
        labels = []
        for p in pieces:
            boxes.append(p['bbox'])
            labels.append(p['category_id'])

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_info['id']])
        }

        return image, target
