import numpy as np
import os
import nibabel as nib
from torch.utils.data import Dataset
from torchvision import transforms


class ACDCDataset(Dataset):
    def __init__(self, data_dir, region=1):
        """
            - region: denotes which region is represented in the mask;
                        - 1: Left ventricle, 2: Myocardium, 3: Right ventricle
            (currently only one region at a time is supported)
        """
        self.data_dir = data_dir
        self.subjects = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]
        self.region = region

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_folder = os.path.join(self.data_dir, self.subjects[idx])
        image_file = os.path.join(subject_folder, f'{self.subjects[idx]}_4d.nii')
        image_data = nib.load(image_file).get_fdata()
        image_data = image_data[:, :, :, 0]
        mask_file = os.path.join(subject_folder, f'{self.subjects[idx]}_frame01_gt.nii')
        mask_data = nib.load(mask_file).get_fdata()
        mask_data = (mask_data == self.region).astype(np.float32)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            lambda x: x.float()
        ])
        image_data = transform(image_data).unsqueeze(1)
        mask_data = transform(mask_data).unsqueeze(1)
        return image_data, mask_data
