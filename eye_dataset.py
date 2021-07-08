import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class EyeDataset(Dataset):
    training_file = 'trainingset.mat'
    training_file_normalsied = 'trainingset_normalised.mat'
    test_file = 'testset.mat'
    test_file_normalsied = 'testset_normalised.mat'
    def __init__(
            self,
            root_dir,
            image_dir,
            mask_dir,
            isPNG=False,
            transform=None):
        self.dataset_path = root_dir
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.isPNG = isPNG
        self.transform = transform
        mask_full_path = os.path.join(self.dataset_path, self.mask_dir)
        self.mask_file_list = [f for f in os.listdir(mask_full_path) if os.path.isfile(os.path.join(mask_full_path, f))]
        random.shuffle(self.mask_file_list)
        

    def __len__(self):
        return len(self.mask_file_list)
    def __getitem__(self, index):
        file_name =  self.mask_file_list[index].rsplit('.', 1)[0]
        img_name = os.path.join(self.dataset_path, self.image_dir, file_name+'.jpg')
        if(self.isPNG):
            img_name = os.path.join(self.dataset_path, self.image_dir, file_name+'.png')
        mask_name = os.path.join(self.dataset_path, self.mask_dir, self.mask_file_list[index])
        image = Image.open(img_name)
        mask = Image.open(mask_name)
        image = np.array(image)
        image = np.array(image).astype(np.float32)
        labels = np.array(mask).astype(np.uint8)
        sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample

train_data = EyeDataset('D:/MLDatasets/NN_human_mouse_eyes', 'fullFrames', 'annotation/png', False)
print(train_data.__getitem__(0))