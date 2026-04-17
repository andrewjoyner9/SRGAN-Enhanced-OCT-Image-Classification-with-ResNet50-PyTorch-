SRGAN + Transfer Learning on OCT Images
ECGR-4116/5116 — AI for Biomedical Applications. Classifies OCT images (DME, DRUSEN, NORMAL) by training two ResNet50 models and comparing them. Model A trains directly on 128x128 images. Model B trains on images that were downscaled to 32x32 and super-resolved back to 128x128 using a trained SRGAN. Model B outperformed Model A on all metrics (accuracy 0.84 → 0.91, F1 0.84 → 0.91, AUC 0.96 → 0.98).

Setup
Install dependencies with pip install -r requirements.txt. A CUDA GPU is recommended — CPU works but SRGAN training will be slow. Place the dataset under data/train/DME, data/train/DRUSEN, and data/train/NORMAL. Update DATA_DIR in Cell 1 if your path differs.

Running
Open SRGAN_OCT_Assignment.ipynb and run all cells top to bottom. Cell 1 handles imports and config, Cell 2 loads data and does a 70/30 split, Cell 3 shows augmentation samples, Cell 4 trains Model A, Cell 5 trains the SRGAN, Cell 6 generates SR images and trains Model B, and Cell 7 evaluates both models with confusion matrices and ROC curves.

Checkpoints
Checkpoints are saved periodically under checkpoints/ so training can be resumed after a GPU session timeout. Model A and B save the best weights plus snapshots every 10 epochs. The SRGAN generator and discriminator save every 5 epochs.
