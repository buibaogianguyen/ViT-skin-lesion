from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image

class HAM10000(Dataset):
    def __init__(self, dataset_path, transform):
        metadata_path = os.path.join(dataset_path, 'HAM10000_metadata.csv')

        self.img_dirs = [
            os.path.join(dataset_path, 'HAM10000_images_part_1'),
            os.path.join(dataset_path, 'HAM10000_images_part_2')
        ]

        if not os.path.exists(metadata_path):
               raise FileNotFoundError(f"Metadata file does not exist at {metadata_path}")
        
        self.df = pd.read_csv(metadata_path)

        self.dataset_path = dataset_path
        self.transform = transform

        self.label_map = {'mel': 0, 'bcc': 1, 'akiec': 2, 'bkl': 3, 'df': 4, 'vasc': 5, 'nv': 6}

        self.df['label_idx'] = self.df['dx'].map(self.label_map)

        if self.df['label_idx'].isna().any():
             raise ValueError("Null labels in metadata, recheck 'dx' column")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['image_id']

        img_path = None

        for img_dir in self.img_dirs:
            test_path = os.path.join(img_dir, f"{img_id}.jpg")
            if os.path.exists(test_path):
                img_path = test_path
                break
        
        if not img_path:
             raise FileNotFoundError(f"Image {img_id}.jpg was not found in directory {self.img_dirs}")
        
        image = Image.open(img_path).convert('RGB')

        label = self.df.iloc[idx]['label_idx']

        if self.transform:
            image = self.transform(image)

        return image, label