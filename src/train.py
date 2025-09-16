import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import segmentation_models_pytorch as smp
from dataloader import *
import pytorch_lightning as pl
from model import *

if __name__ == "__main__":
    ENCODER = 'mobilenet_v2'
    DATA_DIR = 'datasets'

    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['silkworm-disease']
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cuda'
    # Select model at here!!!!!!
    model = Segmentation_model("DeepLabV3", "mobilenet_v2", in_channels=3, out_classes=1)
    # create segmentation model with pretrained encoder

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    # Prepare datasets
    x_train_dir = os.path.join(DATA_DIR, 'train',"images")
    y_train_dir = os.path.join(DATA_DIR, 'train',"mask")
    x_valid_dir = os.path.join(DATA_DIR, 'val',"images")
    y_valid_dir = os.path.join(DATA_DIR, 'val',"mask")
    # Create dataloader
    train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,)

    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Training model
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    trainer = pl.Trainer(
        accelerator="auto",
        devices ="auto",
        max_epochs=50,)
    trainer.fit(
    model, 
    train_dataloaders=train_loader, 
    val_dataloaders=valid_loader,
    ckpt_path="lightning_logs/version_1/checkpoints/epoch=38-step=31706.ckpt"
    )
    print("Testing model")
    
    trainer.test(model,dataloaders=valid_loader,verbose=True)
    trainer.save_checkpoint("results/best_weight_DeeplabV3.ckpt")


    