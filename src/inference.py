import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import torch
ENCODER = 'mobilenet_v2'
encodername = 'mobilenet_v2'
name_encoder = 'mobilenet_v2'
ENCODER2 = 'resnet34'
encodername2 = 'resnet34'
name_encoder2 = 'resnet34'
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import segmentation_models_pytorch as smp
from dataloader import *
import pytorch_lightning as pl
from model import *
from utils import *
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
unet_path = 'silkworm-diseases-detection-using-deeplearning-and-computer-vision-technology/logs/Unet/checkpoints/Unet.ckpt'
arch_path = 'Unet'
psp_path = 'silkworm-diseases-detection-using-deeplearning-and-computer-vision-technology/logs/Unet/checkpoints/Unet.ckpt'
deeplabv3_path = 'silkworm-diseases-detection-using-deeplearning-and-computer-vision-technology/logs/Unet/checkpoints/Unet.ckpt'

def inference(checkpoint_path="",arch="",encoder_name="mobilenet_v2",image_path=""):
    mask_name = os.path.basename(image_path).split(".jpg")[0] + "_mask.png"
    mask_path =  os.path.join("all_label","val","mask",mask_name)
    print(mask_path)
    image = cv2.imread(image_path)
    mask_org = cv2.imread(mask_path)
    mask_org = cv2.resize(mask_org,(480,640))
    image = cv2.resize(image,(480,640))
    img_org = image.copy()
    image_org = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    model = Segmentation_model.load_from_checkpoint(checkpoint_path,arch=arch,encoder_name=encoder_name,in_channels=3,out_classes=1 )
    model.eval()
    #####Tiền xử lý dữ liệu#########
    transform_valid = get_validation_augmentation()
    transfrom_pre = get_preprocessing(preprocessing_fn)
    image_ = transform_valid(image=image)
    input_ = transfrom_pre(image=image_["image"])
    print(input_["image"].shape)
    ###############################
    '''
    pass input model
    '''

    # Model inference
    mask_logit = model(torch.tensor(input_["image"]).unsqueeze(0))
    pr_masks = mask_logit.sigmoid()
    output = pr_masks.detach().numpy()[0]
    output = output.transpose(1, 2, 0)
    output = (output * 255).astype(np.uint8)

    # binary mask
    alpha = 0.2
    output = np.where(output > 100, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Blend mask with original image
    mask_overlay = np.zeros_like(image_org)
    mask_overlay = cv2.drawContours(mask_overlay, contours, -1, (0, 0, 255), -1)
    image_with_mask = cv2.addWeighted(image_org, 1 - alpha, mask_overlay, alpha, 0)

    return image_org, mask_org, output, image_with_mask


if __name__ == "__main__":
    model_path = unet_path
    img_path = "all_image/image_1.jpg"
    ENCODER_WEIGHTS = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

   

    image_org,mask_org,mask,image_with_mask = inference(checkpoint_path=model_path,arch=arch_path,encoder_name=name_encoder,image_path=img_path)

    fig = plt.figure(figsize=(20, 5))  # Kích thước của cửa sổ
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 1])  # Chia cửa sổ thành 4 cột

    image_org = cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB)
    image_with_mask = cv2.cvtColor(image_with_mask, cv2.COLOR_BGR2RGB)



    #image oroginal
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image_org)
    ax1.axis('off')
    ax1.set_title('Original Image')

    #mask predict
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mask, cmap='gray')
    ax2.axis('off')
    ax2.set_title('Predicted  Mask')

    # image from model segmentation
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(image_with_mask)
    ax3.axis('off')
    ax3.set_title('Segmentation')

    # mask original
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(mask_org, cmap='gray') 
    ax4.axis('off')
    ax4.set_title('Original Mask')

    plt.tight_layout()
    plt.show()
    



