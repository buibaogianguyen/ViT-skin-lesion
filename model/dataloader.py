from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image

class ISIC2019(Dataset):
    def __init__(self, dataset_path, transform):
        metadata_path = os.path.join(dataset_path, 'ISIC_2019_Training_Metadata.csv')
        groundtruth_path = os.path.join(dataset_path, 'ISIC_2019_Training_GroundTruth.csv')

        if not os.path.exists(metadata_path):
               raise FileNotFoundError(f"Metadata file does not exist at {metadata_path}")
        if not os.path.exists(groundtruth_path):
                raise FileNotFoundError(f"Ground truth file does not exist at {groundtruth_path}")
        
        metadata_df = pd.read_csv(metadata_path)
        groundtruth_df = pd.read_csv(groundtruth_path)
        
        self.df = metadata_df.merge(groundtruth_df, left_on='image', right_on='image', how='inner')

        label_columns = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
        self.df['label'] = self.df[label_columns].idxmax(axis=1)

        self.dataset_path = dataset_path
        self.transform = transform

        self.img_dir = os.path.join(dataset_path, 'ISIC_2019_Training_Input')

        self.label_map = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AK': 3, 'BKL': 4, 'DF': 5, 'VASC': 6, 'SCC': 7, 'UNK': 8}

        self.df['label_idx'] = self.df['label'].map(self.label_map)

        if self.df['label_idx'].isna().any():
             raise ValueError("Null labels in ground truth, recheck 'label' column")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['image']

        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")

        if not os.path.exists(img_path):
             raise FileNotFoundError(f"Image {img_id}.jpg was not found in directory {self.img_dir}")
        
        image = Image.open(img_path).convert('RGB')

        label = self.df.iloc[idx]['label_idx']

        if self.transform:
            image = self.transform(image)

        return image, label