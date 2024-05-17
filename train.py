# imports

# set the configuration
import configparser
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path

from models import UNet
from models.deeplab import get_deeplab


import seg_data

import torch.nn.functional as F


## Dice coefficient

import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def evaluate(model,model_name, dataloader, device, amp = True):
    """Evaluate the validation set

    Return validation score (dice score).
    
    """
    model.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    valid_loss = 0
    criterion = nn.CrossEntropyLoss()
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in dataloader:
            images, mask_true = batch[0], batch[1]

            # move images and labels to correct device and type
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.float32)

            # predict the mask
            mask_pred = model(images)

            if model_name=="deeplab":
                mask_pred = mask_pred['out']

            if model.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                valid_loss += criterion(masks_pred.squeeze(1), mask_true.float())

                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < model.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                valid_loss += criterion(masks_pred, true_masks)
                mask_true = F.one_hot(mask_true.argmax(dim=1).to(torch.long), model.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), model.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

            
            # Per-class dice score
            per_class_dice_score =[]
            for i in range(model.n_classes):
                per_class_dice_score.append(dice_coeff(mask_pred[:,i,...], 
                                                       mask_true[:,i,...], 
                                                       reduce_batch_first=False))

    model.train()
    return dice_score / max(num_val_batches, 1) , valid_loss/ max(num_val_batches, 1), per_class_dice_score

## Read the config
def read_ini(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config




assert len(sys.argv) ==2, "Please make sure you add the .ini file as a configuration"
assert os.path.isfile(sys.argv[1]), "Configuration file not exist"

config = read_ini(sys.argv[1])

img_path = config["DIR"]["image_dir"]
mask_path =config["DIR"]["mask_dir"]
checkpoint_path = config["DIR"]["checkpoint_path"]
log_dir = config["DIR"]["log_dir"]
saved_model = config["DIR"].get('saved_model',None)

is_img_aug = config["PARAMS"].get('img_aug',False)

start_class_i = int(config["PARAMS"].get('start_class_i',0))
model_name = config["PARAMS"]['model']
scale = int(config["PARAMS"]["scale"])

learning_rate = float(config["PARAMS"]['learning_rate'])
batch_size = int(config["PARAMS"]['batch_size'])
val_percent = int(config["PARAMS"]['val_percent'])

# number of classes
n_classes = int(config["PARAMS"].get('n_classes',False))
# Class weight, if not specified, assign None
class_weights = config["PARAMS"].get('class_weights',False)

epochs = int(config["PARAMS"]['epochs'])



### Setting up the model and datasets ###

# config.get("UNET")
if model_name == "unet" and "UNET" in config:
    bilinear = config["UNET"]["bilinear"]



# Read and init datasets
dataset_whole = seg_data.segDataset(img_path = img_path, 
    mask_path = mask_path, n_classes=n_classes,
    scale=scale , start_class_i = start_class_i)

# Split and create dataloader
l=dataset_whole.__len__()
torch.manual_seed(1)
indices = torch.randperm(len(dataset_whole)).tolist()
dataset = torch.utils.data.Subset(dataset_whole, indices[:-int(np.ceil(l*val_percent/100))])
dataset_val = torch.utils.data.Subset(dataset_whole, indices[int(-np.ceil(l*val_percent/100)):])

dataset.dataset.augmentation=is_img_aug
dataset_val.dataset.augmentation=False

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=0, pin_memory=True)


test_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1,
                                        shuffle=False, num_workers=0, pin_memory=True)

print(f'''Dataset info:
        Image folder: {img_path}
        input image scale: {scale}
        Mask folder: {mask_path}
        Dataset length: {l}
        Validation set percentage: {val_percent}
        Batch size: {batch_size}
        Training set: {dataset.__len__()} images
        Validation set: {dataset_val.__len__()} images
    ''')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'''System info:
        Using device: {device}
        CPU cores: {os.cpu_count()}
        GPU count: {torch.cuda.device_count()}
'''
)


# init model and device



### Network ###
if saved_model is not None:
    model = torch.load(saved_model)
else:
    if model_name =='unet':
        model = UNet(n_channels=3, n_classes=n_classes, bilinear=bilinear)
    elif model_name =="deeplab":
        model = get_deeplab(n_classes)
    
# switch NCHW to NHWC
model = model.to(memory_format=torch.channels_last)
model.to(device=device)

# Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
optimizer = optim.RMSprop(model.parameters(),
                            lr=learning_rate, foreach=True)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
# grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

# Loss function
# if class_weights and class_weights!='auto':
#     class_weights = eval(class_weights)
#     assert len(class_weights) == n_classes, "The length of class weights: {} should equal to the number of classes: {}".format(len(class_weights),n_classes)
#     class_weights_tensor = torch.tensor(class_weights)
    
#     print("Using class weight in loss: {}".format(class_weights_tensor))
#     class_weights_tensor = class_weights_tensor.to(device=device)
#     criterion = nn.CrossEntropyLoss(weight= class_weights_tensor) if n_classes > 1 else nn.BCEWithLogitsLoss()
# # Loss function with class weight
# else:
#     print("No class weight in loss")
#     criterion = nn.CrossEntropyLoss() if n_classes > 1 else nn.BCEWithLogitsLoss()
if class_weights and class_weights!='auto':
    class_weights = eval(class_weights)
    assert len(class_weights) == n_classes, "The length of class weights: {} should equal to the number of classes: {}".format(len(class_weights),n_classes)
    class_weights_tensor = torch.tensor(class_weights)
    
    print("Using class weight in loss: {}".format(class_weights_tensor))
    class_weights_tensor = class_weights_tensor.to(device=device)
    criterion = nn.CrossEntropyLoss(weight= class_weights_tensor) 
elif class_weights=='auto':
    print("Use auto-weighting")
    criterion = nn.BCEWithLogitsLoss(reduction="none")
else:
    print("No class weight in loss")
    criterion = nn.CrossEntropyLoss() if n_classes > 1 else nn.BCEWithLogitsLoss()




global_step = 0

scaler=None

# Learning rate warm up
warmup_factor = 1.0 / 1000
warmup_iters = min(1000, len(train_loader) - 1)
warm_up_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=warmup_factor, total_iters=warmup_iters
)

print(
    f'''Model info:
        Model name: {model_name}
        
        Output channels: {n_classes}
        Epochs: {epochs}
        Learning Rate: {learning_rate}
        
''')
 
#{"Bilinear" if model.bilinear else "Transposed conv"} upscaling



# if args.load:
#     state_dict = torch.load(args.load, map_location=device)
#     del state_dict['mask_values']
#     model.load_state_dict(state_dict)
#     logging.info(f'Model loaded from {args.load}')



### Begin training ###
writer = SummaryWriter()

for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0
    print("Training epoch: {}/{}".format(epoch,epochs))
    for batch in train_loader:
        images, true_masks = batch[0], batch[1]

        # assert images.shape[1] == model.n_channels, \
        #     f'Network has been defined with {model.n_channels} input channels, ' \
        #     f'but loaded images have {images.shape[1]} channels. Please check that ' \
        #     'the images are loaded correctly.'
        

        images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        true_masks = true_masks.to(device=device, dtype=torch.float32)

        masks_pred = model(images)

        if model_name =="deeplab":
            masks_pred = masks_pred['out']

        if class_weights == 'auto':
            class_weights_tensor = torch.Tensor([torch.sum(true_masks[:,c_i,...]==1) for c_i in range(true_masks.shape[1])])
            class_weights_tensor = torch.nn.functional.normalize(class_weights_tensor, p=1.0, dim = 0)
            class_weights_tensor = class_weights_tensor.to(device=device, dtype=torch.float32)
        # print(images.shape, true_masks.shape, masks_pred.shape)
        if model.n_classes == 1:
            # Weights can be added for the criterion
            if class_weights == 'auto':
                loss = criterion(masks_pred.squeeze(1), true_masks.float() )
                
                loss = torch.mean(class_weights_tensor[None,:,None,None] * loss)
            else:    
                loss = criterion(masks_pred.squeeze(1), true_masks.float())
            loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
        else:
            if class_weights == 'auto':
                loss = criterion(masks_pred, true_masks)
                loss = torch.mean(class_weights_tensor[None,:,None,None] * loss)
            else: 
                loss = criterion(masks_pred, true_masks)
            loss += dice_loss(
                F.softmax(masks_pred, dim=1).float(),
                F.one_hot(true_masks.argmax(dim=1), model.n_classes).permute(0, 3, 1, 2).float(),
                multiclass=True
            )
        writer.add_scalar('Loss/train', loss, epoch)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if epoch == 0:
            warm_up_scheduler.step()
        
        # Update the global step and epoch loss
        global_step += 1
        epoch_loss += loss.item()

    
    # Validation stage
    val_score,val_loss,per_class_dice_score = evaluate(model,model_name, test_loader, device)
    # writer.add_scalar('Loss/Valid', epoch_loss, epoch)
    writer.add_scalar('Loss/Valid' , val_loss, epoch)
    writer.add_scalar('Dice score/average' , val_score, epoch)

    for i_class in range(len(per_class_dice_score)):
        writer.add_scalar('Dice score/class {}'.format(i_class + 1) ,
                           per_class_dice_score[i_class], epoch)

    scheduler.step(val_score)    

 
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    # state_dict['mask_values'] = dataset.mask_values
    if saved_model is None:
        torch.save(model, str(checkpoint_path + '/{}_checkpoint_epoch{}.pth'.format(model_name, epoch)))
    else:
        saved_model_name = saved_model.split("/")[-1].split(".")[0]
        torch.save(model, str(checkpoint_path + '/{}_plus_epoch{}.pth'.format(saved_model_name, epoch)))
        
    print(f'Checkpoint {epoch} saved! in {checkpoint_path}')
writer.flush()
writer.close()