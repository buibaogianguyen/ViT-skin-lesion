# Vision Transformer for Skin Lesion predictions/classification

This project implements a PyTorch-based Vision Transformer (ViT) model trained on the [ISIC-2019 Dataset](https://www.kaggle.com/datasets/andrewmvd/isic-2019), which has 25,331 dermoscopic images for melanoma prediction/classification. This model utilizes transfer learning, loading weights from "vit_base_patch16_224" from the timm library, which was trained on the ImageNet dataset. This allowed the model to reach 83% validation accuracy after 27 epochs on a highly imbalanced dataset. Running train.py will initialize a database file using SQLAlchemy for ease in querying and downstream analysis. The model also integrates FastAPI, although no UI currently exist.

# Navigation
- [Research Paper - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](#research-paper)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [API Run](#api-run)
  - [Database](#database)
- [Performance](#performance)

# Research Paper - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <a id="research-paper"></a>

This model is a Vision Transformer deep-learning architecture proposed in the 2015 research paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929), authored by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. The model achieves up to 83% validation accuracy after 27 epochs, on a highly imbalanced dataset.

<p align="center">
<img src="https://i.postimg.cc/Hxb4tPqJ/image.png" width="800">
</p>

# Requirements

```bash
fastapi>=0.70.0
uvicorn>=0.15.0
sqlalchemy>=1.4.0
pandas>=1.3.0
torch>=1.9.0
torchvision>=0.10.0
Pillow>=8.3.0
kagglehub>=0.1.15
```

- See `requirements.txt` for a full list of dependencies.

# Project Structure

```bash
ViT-skin-lesion/
├── data/
│   └── data_downloader.py
│   └── data_manager.py
├── model/
│   ├── __init__.py
│   ├── vit.py
│   └── dataloader.py
├── .gitattributes
├── README.md
├── best_acc.json
├── database_storing.py
├── inference.py
├── main.py
├── skin_lesions.db
├── train.py
└── vit.pth
```

# Setup

Clone the repository:
```cmd
git clone https://github.com/buibaogianguyen/ViT-skin-lesion.git
cd ViT-skin-lesion
```

Install dependencies:
```cmd
pip install -r requirements.txt
```

If using a GPU, ensure the PyTorch version matches your CUDA toolkit (e.g., for CUDA 11.8, install `torch>=2.0.0+cu118`). Check [PyTorch's official site](https://pytorch.org/) for CUDA-specific installation.

Prepare the dataset:
The ISIC-2019 dataset is automatically downloaded by `data_downloader.py` to the `C:/Users:/{youruser}:/.cache:/kagglehub:/datasets:/andrewmvd:/isic-2019` directory during the first run of `train.py`. Ensure you have an internet connection, and 10GB of remaining disk storage.

# Usage

## Training

To train the model, run:
```cmd
python train.py
```

Configurations:
- **Epochs**: Default is 30 epochs (adjustable in `train.py`).
- **Batch Size**: Currently set to 48, adjust to 16 or 32 if limited memory.
- **Image Resolution**: Resized to 224x224.
- **Hyperparameters**: Learning rate=0.0001, weight decay=0.01, CosineAnnealingLR scheduler.

The script saves:
- The best model checkpoint (`resnet34.pth`) based on validation accuracy.
- Validation metrics (`best_acc.json`) for accuracy.

## Inference

To run the model with a test image, run:
```cmd
python inference.py
```
Remember to include the image path of the image you want to test at Line 41 between the quotation marks.
```python
image_path = "{file-path-here}"
```

## API run

If you want to run the .db file on FastAPI, run:
```cmd
python main.py
```
Then enter http://127.0.0.1:8000/docs

No formal web UI is currently included with the API but will be added in the future.

## Database

The file skin_lesions.db is saved automatically when running train.py by SQLAlchemy, any new inference runs testing an image with the same file name as images from ISIC-2019 will update existing image data in the probabilities section of the database, this records probabilities predicted by the model for each class. New images will be added, not updated. 

.db file may be ran in external applications or software, such as [SQLite](https://sqlite.org/).

# Performance

This shows the performance of the model over the first 30 epochs of training in terms of Validation Accuracy over epochs. The model reached a relatively good validation accuracy of 83%, considering the dataset being very imbalanced.

<p align="center">
<img src="https://i.postimg.cc/0ypkPG1Y/Vi-T-Validation-Accuracy-over-Epochs.png" width="800">
</p>


# Notes
**Note**: Ensure `checkpoints/vit.pth` exists from training.

# License

This project is licensed under the MIT License. See the `LICENSE` file for details (create one if needed).

# Contributing

Contributions are welcome! Please open an issue or submit a pull request with improvements or bug fixes.
