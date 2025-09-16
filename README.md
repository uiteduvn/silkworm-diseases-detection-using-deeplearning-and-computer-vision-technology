# SilkLDP: A Pixel-Level Annotated Dataset and Deep Learning Approach for Silkworm Disease Detection

### Overview
This repository contains the source code, dataset of our paper:
"SilkLDP: A Pixel-Level Annotated Dataset and Deep Learning Approach for Silkworm Disease Detection" Submitted to RIVF 2025 (IEEE Conference on Computing & Communication Technologies).

The project focuses on automating diseased silkworm detection using image segmentation techniques. We constructed a dataset of 4,029 silkworm images with pixel-level annotations and evaluated several popular deep learning models (U-Net, FCN, PSPNet, DeepLab). Our proposed approach — U-Net with ResNet34 encoder — achieved the best performance.

### Repository Structure

├── datasets/   You can download the dataset from https://drive.google.com/drive/folders/1Y_fTj0Mp3nc8Z-zj9HGJFXKFfOZgMiuE?usp=drive_link

├── models/ You can download the model from https://drive.google.com/drive/folders/1Y_fTj0Mp3nc8Z-zj9HGJFXKFfOZgMiuE?usp=drive_link

├── src/

│   ├── build_and_evaluate_model.ipynb

│   ├── train.py

│   ├── dataloader.py

│   ├── model.py

│   ├── inference.py

│   ├── utils.py

│   └── demo.py

├── requirements.txt

└── README.md 

🚀 Organize your dataset as follows:

dataset/
 ├── train/images/
 │    ├── image_1.png
 │    ├── image_2.png
 │    └── ...
 └── train/masks/
      ├── image_1_mask.png
      ├── image_2_mask.png
      └── ...
├── val/images/
 │    ├── image_3.png
 │    ├── image_4.png
 │    └── ...
 └── val/masks/
      ├── image_3_mask.png
      ├── image_4_mask.png
      └── ...   

📊 Results

Pixel Accuracy: 95.59%

IoU: 0.7746

Dice Score: 0.8730

📜 Citation

If you use this code or dataset, please cite our paper:

@inproceedings{SilkwormRIVF2025,
  author    = {HoAnhKhoi, DoTrongHop},
  title     = {SilkLDP: A Pixel-Level Annotated Dataset and Deep Learning Approach for Silkworm Disease Detection},
  booktitle = {Proceedings of the 2025 RIVF International Conference on Computing & Communication Technologies},
  year      = {2025},
  publisher = {IEEE}
}

📬 Contact

For questions or collaborations, please contact: 20521477@gm.uit.edu.vn
