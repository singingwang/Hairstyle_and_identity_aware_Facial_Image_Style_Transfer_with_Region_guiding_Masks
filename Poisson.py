import numpy as np
import os
import random 
from PIL import Image
from scipy import ndimage

import cv2
import scipy.sparse
from os import path
from scipy.sparse.linalg import spsolve

# for hairstyle Poisson blending
mask_folder = "expr/Result_Ours_tofemale_mask"
expr_folder = "/home/0856609/stargan-v2-master/Ours_assets/src/src_tofemale_Poisson"
src_root = "expr/src"
struc_root = "expr/Result_Ours_tofemale"
    

#expr_mask_folder = "expr/Result_Ours_tomale_mask"
def toInt255(image):
    #image = np.clip(image * 0.5 + 0.5, 0, 1)
    return (image).astype(np.uint8)
    
def toImage255(image):
    #image = np.clip(image * 0.5 + 0.5, 0, 1)
    return (image*255).astype(np.uint8)
    
def Laplacian_matrix(m, n):   
    block = scipy.sparse.lil_matrix((n, n))
    block.setdiag(4)
    block.setdiag(-1, -1)    
    block.setdiag(-1, 1)        
    mat = scipy.sparse.block_diag([block] * m).tolil()    
    mat.setdiag(-1, 1*n)
    mat.setdiag(-1, -1*n)    
    return mat
def Poisson_Blending(source, target, mask):
  y_max, x_max = target.shape[:-1]
  y_min, x_min = 0, 0
  x_range = x_max-x_min
  y_range = y_max-y_min
  mask = np.float32(mask)
  mask = mask/255

# Change mask to 0 and 1
  mask[mask > 0.7] = 1
  mask[mask <= 0.7] = 0
# construct the matrix A
  mat_A = Laplacian_matrix(y_range, x_range) 
  laplacian = mat_A.tocsc()   # Convert this matrix to Compressed Sparse Column format, this is for right-hand side
  for y in range(1, y_range - 1):
    for x in range(1, x_range - 1):
        if mask[y, x] == 0:
            k = x + y * x_range
            mat_A[k, k] = 1
            mat_A[k, k + 1] = 0
            mat_A[k, k - 1] = 0
            mat_A[k, k + x_range] = 0
            mat_A[k, k - x_range] = 0
  mat_A = mat_A.tocsc()   # The matrix A with mask
  mask_flat = mask.flatten()    
  for channel in range(source.shape[2]):
    source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
    target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()        
    # construct b (right-hand side)
    mat_b = laplacian.dot(source_flat)
    # add background
    mat_b[mask_flat == 0] = target_flat[mask_flat == 0]
    # Solve Ax = b
    x = spsolve(mat_A, mat_b)    
    x = x.reshape((y_range, x_range))
    x[x > 255] = 255
    x[x < 0] = 0
    x = x.astype('uint8')    
    target[y_min:y_max, x_min:x_max, channel] = x
  return target
    
def main():
    
    
    #erosion_range = 15
    #base_e = ndimage.generate_binary_structure(2, 1)# 3*3
    #struct_e = ndimage.iterate_structure(base_e, 2*erosion_range+1) #31*31
    
    
    for img in os.listdir(struc_root):
       sf = img[0:6]+".jpg"
       mask = "mask_"+img[0:6]+".jpg"
       sf_path = os.path.join(src_root,sf)
       mask_path = os.path.join(mask_folder,mask)
       struc_path = os.path.join(struc_root,img)
       
       source = cv2.imread(sf_path, cv2.IMREAD_COLOR) 
       target = cv2.imread(struc_path, cv2.IMREAD_COLOR)
       mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
       source = cv2.resize(source, (256, 256), interpolation=cv2.INTER_AREA)
       
       tar = Poisson_Blending(source, target, mask)
       result_path = os.path.join(expr_folder, img)
       cv2.imwrite(result_path, tar)
       """
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
       
       Image.fromarray(toImage255(mask_img)).save(os.path.join(expr_mask_folder, "mask_"+sf))
       
       Image.fromarray(final).save(os.path.join(expr_folder, img))
       """
if __name__ == '__main__':
    main()