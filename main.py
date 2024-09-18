import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Importing learning rate scheduler
import torch.optim.lr_scheduler as lr_scheduler

from unet import UNet
from dataset import ImgDataset

import cv2 as cv

if __name__ == "__main__":

    # Training Hyperparameters
    lr = 3e-4
    batch_size = 8
    epochs = 1

    # Data Flow
    root = '/Users/aryanbhobe/Desktop/Pytorch/U-Net/data' # root path
    model_save_path = '/Users/aryanbhobe/Desktop/Pytorch/U-Net/models/unet.pth' # model save path

    device = 'cpu'

    train_dataset = ImgDataset(root) # Training Dataset
    val_dataset = ImgDataset(root+"/valid") # Validation Dataset

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    # Initialising the Neural Network
    model = UNet(in_channels=3, num_classes=6).to(device)

    optimizer = optim.AdamW(model.parameters(), lr = lr)

    # Initialising scheduler
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor = 1.0, end_factor = 0.3, total_iters=10)

    criterion = nn.CrossEntropyLoss() # <-- SOURCE OF ISSUES

    # Training Loop
    for epoch in tqdm(range(epochs)):

        model.train()
        train_running_loss = 0

        for idx, img_and_mask in enumerate(tqdm(train_dataloader)):

            img = img_and_mask[0].float().to(device)
            mask = img_and_mask[1].long().to(device)
            mask = mask.squeeze(1)

            print("\nMask shape:", mask.shape," | Mask Type: ", mask.dtype)

            yhat = model(img) # prediction

            print('\nPrediction shape: ',yhat.shape," | Prediction Type: ",yhat.dtype)

            optimizer.zero_grad()
            loss = criterion(yhat, mask) # calculate loss
            train_running_loss += loss.item()

            loss.backward() # apply backpropagation
            optimizer.step() # update weights

        train_loss = train_running_loss / idx+1

        model.eval()

        val_running_loss = 0
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].long().to(device)
                mask = mask.squeeze(1)

                yhat = model(img)
                loss = criterion(yhat, mask)
                val_running_loss += loss.item()

            val_loss = val_running_loss / idx+1

        print("-"*30)
        print(f"Train loss EPOCH {epoch+1}: {train_loss: .4f}")
        print(f"Valid loss EPOCH {epoch+1}: {val_loss: .4f}")

    torch.save(model.state_dict(), model_save_path)