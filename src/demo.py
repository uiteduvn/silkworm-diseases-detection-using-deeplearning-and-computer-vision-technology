import numpy as np
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
model_path = 'model_saved/best2.pt'
from tkinter import Tk, Label
from PIL import Image, ImageTk
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import tensorflow as tf
def list_image_path(i):
    image_paths = []
    mask_paths = []
    for i in range(1,11):
        image_path = 'images/' + f'image_{i}.jpg'
        mask_path = 'masks/' + f'/image_{i}_mask.png'
        image_paths.append(image_path)
        mask_paths.append(mask_path)
    return image_paths,mask_paths



image_paths,mask_paths = list_image_path(10)
idx = 1
image_path = 'images/' + 'image_1446.jpg'

mask_path = 'labels/' + f'/image_1446_mask.png'
list_of_image=False

def demo1(img_path,model_path,mask_path,image_paths,mask_paths,list_of_image):
    if list_of_image == True:
        for img_path, mask_path in zip(image_paths, mask_paths):
            original_image = mpimg.imread(img_path)
            mask_image = mpimg.imread(mask_path)
            model_yolo = YOLO(model_path)
            results = model_yolo(img_path, boxes=False)
            
            segmentation_mask = results[0].masks.data.cpu().numpy()
            binary_mask = np.zeros((results[0].masks.data.shape[1], results[0].masks.data.shape[2]), dtype=np.uint8)
            for mask in segmentation_mask:
                binary_mask = np.maximum(binary_mask, mask)

            image = Image.fromarray(results[0].plot(boxes=False)[..., ::-1])
            
            fig = plt.figure(figsize=(20, 5))  
            gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 1])  

            
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(original_image)
            ax1.axis('off')
            ax1.set_title('Original Image')

            
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(binary_mask, cmap='gray')
            ax2.axis('off')
            ax2.set_title('Predicted Binary Mask')


            ax3 = fig.add_subplot(gs[0, 2])
            overlay_image = np.array(original_image).copy()

            
            segmentation_mask = results[0].masks.data.cpu().numpy()
            binary_mask = np.zeros(
                (results[0].masks.data.shape[1], results[0].masks.data.shape[2]), dtype=np.uint8
            )
            for mask in segmentation_mask:
                binary_mask = np.maximum(binary_mask, mask)

           
            binary_mask = cv2.resize(binary_mask, (overlay_image.shape[1], overlay_image.shape[0]))

            
            red_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 4), dtype=np.uint8)
            red_mask[binary_mask > 0] = [255, 0, 0, 120]  # (R,G,B,A)

            
            ax3.imshow(overlay_image)
            
            ax3.imshow(red_mask)
            ax3.axis('off')
            ax3.set_title('Segmentation')

            
            ax4 = fig.add_subplot(gs[0, 3])
            ax4.imshow(mask_image, cmap='gray')  
            ax4.axis('off')
            ax4.set_title('Original Mask')

            plt.tight_layout()
            plt.show()
    else:
        original_image = mpimg.imread(img_path)
        mask_image = mpimg.imread(mask_path)
        model_yolo = YOLO(model_path)
        results = model_yolo(img_path, boxes=False)
        
        segmentation_mask = results[0].masks.data.cpu().numpy()
        binary_mask = np.zeros((results[0].masks.data.shape[1], results[0].masks.data.shape[2]), dtype=np.uint8)
        for mask in segmentation_mask:
            binary_mask = np.maximum(binary_mask, mask)

        image = Image.fromarray(results[0].plot(boxes=False)[..., ::-1])
       
        fig = plt.figure(figsize=(20, 5))  
        gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 1])  

        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original_image)
        ax1.axis('off')
        ax1.set_title('Original Image')

        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(binary_mask, cmap='gray')
        ax2.axis('off')
        ax2.set_title('Predicted Binary Mask')

       
        ax3 = fig.add_subplot(gs[0, 2])
        overlay_image = np.array(original_image).copy()

       
        segmentation_mask = results[0].masks.data.cpu().numpy()
        binary_mask = np.zeros(
            (results[0].masks.data.shape[1], results[0].masks.data.shape[2]), dtype=np.uint8
        )
        for mask in segmentation_mask:
            binary_mask = np.maximum(binary_mask, mask)

        
        binary_mask = cv2.resize(binary_mask, (overlay_image.shape[1], overlay_image.shape[0]))

        
        red_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 4), dtype=np.uint8)
        red_mask[binary_mask > 0] = [255, 0, 0, 120]  # (R,G,B,A)

        
        ax3.imshow(overlay_image)
        
        ax3.imshow(red_mask)
        ax3.axis('off')
        ax3.set_title('Segmentation')

        
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(mask_image, cmap='gray')  # Sử dụng colormap 'gray' cho mask
        ax4.axis('off')
        ax4.set_title('Original Mask')

        plt.tight_layout()
        plt.show()

demo1(image_path,model_path,mask_path,image_paths,mask_paths,list_of_image)

