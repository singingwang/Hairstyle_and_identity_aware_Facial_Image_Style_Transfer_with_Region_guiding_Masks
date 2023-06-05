import numpy as np
import os
import random 
from PIL import Image

from scipy import ndimage
############# change expr_folder, expr_mask_folder, struc_root

mask_folder = "data/jpeg/cs-mask-HQ"
#expr_folder = "/home/0856609/stargan-v2-master/Ours_assets/src/src_tomale_v34_12_model28"
expr_folder = "/home/0856609/stargan-v2-master/Ours_assets/src/src_tofemale_v90_FID_selected"
expr_mask_folder = "expr/Result_Ours_tofemale_FID_mask"
def toInt255(image):
    #image = np.clip(image * 0.5 + 0.5, 0, 1)
    return (image).astype(np.uint8)
    
def toImage255(image):
    #image = np.clip(image * 0.5 + 0.5, 0, 1)
    return (image*255).astype(np.uint8)
def main():
    
    src_root = "expr/src_male_selected"
    #struc_root = "expr/Result_Ours_tomale"
    #src_root = "/home/0856609/stargan-v2-master/data/celeba_hq/val/male"
    #struc_root = "expr/userstudy/Result_Ours_tomale_v90"
    struc_root = "expr/Result_Ours_tofemale_v90_model22_FID_selected"
    
    #struc_root = "/home/0856609/stargan-v2-master/expr/results/First_starganv2_tofemale"
    
    erosion_range = 15
    base_e = ndimage.generate_binary_structure(2, 1)# 3*3
    struct_e = ndimage.iterate_structure(base_e, 2*erosion_range+1) #31*31
    
    if not os.path.exists(expr_folder):
        os.makedirs(expr_folder)
    
    for n, img in enumerate(os.listdir(struc_root)):
       if n%100 == 0: 
         print(n)
       sf = img[0:6]+".jpg"
       #rf = img[7:13]+".jpg"
       mask = img[0:6]+".jpg"
       sf_path = os.path.join(src_root,sf)
       mask_path = os.path.join(mask_folder,mask)
       struc_path = os.path.join(struc_root,img)
       
       sf_img = Image.open(sf_path)
       mask_img = Image.open(mask_path)
       struc_img = Image.open(struc_path)
       
       sf_img = sf_img.resize((256, 256), Image.BILINEAR)
       
       mask_img = mask_img.resize((256, 256), Image.BILINEAR)
       mask_img = np.array(mask_img)/255
       
       mask_img = ndimage.binary_erosion(mask_img,structure=struct_e).astype(mask_img.dtype)
       mask_img = ndimage.gaussian_filter(mask_img,sigma = 10)
       
       mask_img = np.expand_dims(mask_img,axis = -1)
       mask_img = np.tile(mask_img,[1,1,3])
       
       face = sf_img*mask_img
       
       
       rev_mask = 1-mask_img
       back = struc_img*rev_mask
       final = face+back
       final = toInt255(final)
       
       #Image.fromarray(toImage255(mask_img)).save(os.path.join(expr_mask_folder, "mask_"+sf))
       
       Image.fromarray(final).save(os.path.join(expr_folder, img))

if __name__ == '__main__':
    main()