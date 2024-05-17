# set the configuration
import configparser
import os
import sys
from pathlib import Path

import cv2
import numpy as np

import torch
import torchvision

import seg_data
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F
from PIL import Image 
from torchvision.io import read_image

## Read the config
def read_ini(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config


assert len(sys.argv) ==2, "Please make sure you add the .ini file as a configuration"
assert os.path.isfile(sys.argv[1]), "Configuration file not exist"

config = read_ini(sys.argv[1])


img_path = config["DIR"]["image_dir"]
output_vis_path =config["DIR"]["output_vis_path"]
checkpoint_path = config["DIR"]["checkpoint_path"]
output_path = config["DIR"]["output_path"]

start_class_i = int(config["PARAMS"].get('start_class_i',0))
scale = int(config["PARAMS"]["scale"])


Path(output_path).mkdir(parents=True, exist_ok=True)
Path(output_vis_path).mkdir(parents=True, exist_ok=True)


#assert os.path.isfile(checkpoint_path), "Checkpoint file not exist"


print(f'''Predicting info:
        Reading images from: {img_path}
        Reading the checkpoint from: {checkpoint_path}
        The output directory: {output_path}
''') 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'''System info:
        Using device: {device}
        CPU cores: {os.cpu_count()}
        GPU count: {torch.cuda.device_count()}
''')


dataset_pred = seg_data.segDataset(img_path = img_path,scale=scale,is_train=False, start_class_i = start_class_i)
data_loader_pred = torch.utils.data.DataLoader(dataset_pred, batch_size=1, shuffle=False)


model = torch.load(checkpoint_path)
model = model.to(device)
model.eval()

dataset_pred.img_path
dataset_pred.imgs

## Iterate through the images and save them to the directory.
for idx, (img,img_info) in enumerate(data_loader_pred):
    
    img_name = dataset_pred.imgs[idx]
        
    # print("img info++++++++++",img_info)
    img = img.to(device)  
    out = model(img)
    
    if "deeplab" in checkpoint_path:
        out = out['out']

    
    mask_temp = out[0].argmax(0) ==1
    
    # img_with_mask = draw_segmentation_masks(Image.open(os.path.join(img_path, img_name)).PIL_TO_TENSOR(),
    #                                         masks = mask_temp, alpha=0.5)
    
    # img_with_mask = img_with_mask.detach()
    # img_with_mask = F.to_pil_image(img_with_mask)

    out_temp = out.cpu().detach().numpy()
    seg= out_temp[0].transpose(1, 2, 0).argmax(2)
    
    # output_mask = np.zeros(seg.shape).astype('uint8')
    # output_mask[seg==1]=255

    if scale!=1:
        seg = cv2.resize(seg, (img_info['w'].item(),img_info['h'].item()),
                interpolation = cv2.INTER_NEAREST )
    
    # print(os.path.join(output_path,img_name))
    cv2.imwrite( os.path.join(output_path,img_name),seg.astype('uint8'))


    cv2.imwrite( os.path.join(output_vis_path,img_name), np.interp(seg, [0, np.max(seg)],[1,255]).astype('uint8')  )
    # cv2.imwrite( os.path.join(output_vis_path,img_name), img_with_mask.astype('uint8')  )
    # img_with_mask.save(os.path.join(output_vis_path,img_name))
    # img_pil = F.to_pil_image(img[0])
    # img_with_mask.save(os.path.join(output_vis_path,img_name))
    #  np.interp(img[0].cpu().detach().numpy().transpose(1, 2, 0),[0,1],[1,255]).astype('uint8'))
    
    
    # img_cv = cv2.imread(os.path.join(img_path, img_name))
    
    # alpha = 0.5 # set your desired alpha value

    # # create the overlay mask
    # img_cv = img.cpu().detach().numpy()

    # img_cv= img_cv[0].transpose(1, 2, 0).astype('uint8')
    
    # print(img_cv.shape)
    
    # overlay_mask = np.zeros_like(img_cv)
    # overlay_mask[seg==1] = (0, 0, 255) # set the color of the mask (here, red)

    # # # apply the mask to the image
    # result = cv2.addWeighted(img_cv, 1 - alpha, overlay_mask, alpha, 0)

    # cv2.imwrite( os.path.join(output_vis_path,img_name), result  )
