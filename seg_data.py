import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torchvision.transforms as T
import torch
import torch.utils.data
import albumentations as A


album_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(0.5, 1.5)),  # Adjust contrast limits as needed
    A.RandomGamma(),
    A.Rotate()
])




def contour_to_seg(seg,contour_coords,value, scale):
    """Turn the contour cell from df to a segmentation

    Args:
        seg (_type_): segmentation. shape (height, width)
        contour_coords (_type_): contours in opencv format [[[x11,y11],[x12,y12]] , [[x21,y21],[x22,y22]]]
        value (_type_): The value assign to the seg using this contour_coords
        scale: The scale used to resize the coordinates, e.g. x11=x11//scale 

    Returns:
        _type_: segmentation. shape (height, width)
    """
    contour_coords = eval(contour_coords)
    contour_cv = [(np.array(contour, dtype='int32')//scale).astype('int32') for contour in contour_coords]
    cv2.fillPoly(seg, contour_cv, value)
    return seg

def seg_to_mask(seg,n_cl):
    """Segmentation to mask

    Args:
        seg (_type_): segmentation. shape (height, width), the value can be 0 to (n_cl-1)
        n_cl (_type_): The number of classes for this segmentation

    Returns:
        _type_: mask, shape [height, width, n_cl]. value can only be 0,1
    """
    assert len(seg.shape) ==2 , "Make sure input is [height, width]"

    cl = np.unique(seg)
    h,w =seg.shape
    masks = np.zeros((h, w , n_cl))
    
    for i, c in enumerate(cl):
        masks[:, : , i] = seg == c

    masks = masks.astype('uint8')

    return masks

class segDataset(torch.utils.data.Dataset):
    """Dataset class for segmentation
    """    
    def __init__(self, img_path, mask_path=None, df_path=None, is_train=True,transforms=None,
     n_classes = None,scale=1, start_class_i = 0, augmentation = True):
        """Init function

        Args:
            img_path (_type_): Image folder
            mask_path (_type_, optional): Mask folder. Defaults to None.
            df_path (_type_, optional): dataframe of the image names and segmentation. Defaults to None.
            train (bool, optional): _description_. Defaults to True.
            transforms (_type_, optional): _description_. Defaults to None.
            n_classes: number of classes, the default is 2 (fore ground and back ground)
        """        
        
        self.augmentation = augmentation
        self.transforms = transforms
        self.scale = scale
        self.to_tensor = T.ToTensor()

        self.n_classes = n_classes
        
        self.is_train=is_train

        self.start_class_i = start_class_i
        # load all image files, sorting them to
        # ensure that they are aligned
        self.img_path = img_path
        if mask_path!=None:
   
            self.mask_path = mask_path

            self.imgs = list(sorted(os.listdir(img_path)))
            self.masks = list(sorted(os.listdir(mask_path)))

            self.mode_df =False
        elif df_path!=None:
            self.df = pd.read_csv(df_path,index_col='file')

            self.imgs = self.df.index.values

            self.mode_df =True
        elif df_path ==None:
            self.imgs = list(sorted(os.listdir(img_path)))

    def __getitem__(self, idx):
        transform = T.Compose([T.ToTensor()])
        
        # load images
        img_name_path = os.path.join(self.img_path, self.imgs[idx])
        # PIL read and resize img (RGB)
        # img = Image.open(img_name_path).convert("RGB")
        img = cv2.imread(img_name_path, cv2.IMREAD_COLOR )

        # w,h = img.size 

        h, w = img.shape[:2] 
        w_resized = int(w//self.scale)     
        h_resized = int(h//self.scale)     

        # img = img.resize(( w_resized, h_resized))
        img = cv2.resize(img, (w_resized, h_resized))
        
        if not self.is_train:
            img_info={}
            img_info['w']=w
            img_info['h']=h
            
            return (self.to_tensor(img),img_info)

            # Use images as the labels
        
        if self.mode_df == False:
            mask_name_path = os.path.join(self.mask_path, self.masks[idx])
            # Read and resize Mask (grey value)
            mask = Image.open(mask_name_path).convert('L')
            mask = mask.resize((w_resized , h_resized))
            w_mask,h_mask = mask.size
            mask = np.array(mask)

            if self.augmentation:
                transformed = album_transform(image=img, mask=mask)
                img = transformed['image']
                mask = transformed['mask']

            if self.n_classes!=None:
                # print("""
                # Create the mask using n_classes
                # mask_temp shape (height, width, n classes)
                # """) 


                mask_temp = np.zeros((h_mask,w_mask,self.n_classes))
                for i_class in range(self.n_classes):
         
                    mask_temp[...,i_class] = np.where(mask == i_class + self.start_class_i, 1, mask_temp[...,i_class])

                    # mask_temp[...,i_class] = np.where(mask != i_class, mask_temp[...,i_class], 1)
            
            else:
                # print("""
                # When self.n_classes is none, use the default setting
                # It reads a mask as the segmentation, (0,255)
                # """) 
                
                mask_temp = np.zeros((h_mask,w_mask,2))
                mask_temp[...,0] = np.where(mask == 255, mask_temp[...,0], 1)
                mask_temp[...,1] = np.where(mask != 255, mask_temp[...,1], 1)
            
            mask_tensor = self.to_tensor(mask_temp)
            # mask_tensor = torch.from_numpy(mask_temp.transpose(2, 0, 1))
        
        # Use the csv as the label
        else: 
            row = self.df.loc[self.imgs[idx],:]
            colnames=self.df.columns
            
            seg = np.zeros((h_resized, w_resized))
            for i_col, col in enumerate(colnames):
                contour_coords = row[col]
                value=i_col+1
                seg = contour_to_seg(seg, contour_coords,value,self.scale)

            mask_temp = seg_to_mask(seg,n_cl=len(colnames)+1)
            
            mask_tensor = torch.from_numpy(mask_temp.transpose(2, 0, 1))
    
        # # instances are encoded as different colors
        # obj_ids = np.unique(mask)
        # # first id is the background, so remove it
        # obj_ids = obj_ids[1:]

        # # split the color-encoded mask into a set
        # # of binary masks
        # masks = mask == obj_ids[:, None, None]


        img_tensor = self.to_tensor(img)
        # img_tensor  = torch.from_numpy(img)
        # Customised transformation.
        # if self.transforms is not None:
        #     img, mask = self.transforms(img, mask)

        
        return img_tensor,mask_tensor

    def __len__(self):
        return len(self.imgs)

if __name__=="__main__":
    dataset = segDataset(
    img_path = img_path, 
    mask_path = mask_path,
    scale=20)