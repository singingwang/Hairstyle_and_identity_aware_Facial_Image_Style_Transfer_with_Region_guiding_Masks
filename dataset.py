import os
import random 
import numpy as np 
from PIL import Image
from scipy import ndimage
dataset_folder = "data"
#v31
def get_src_loader(src_data, batch_size, shuffle=True, flip=False):
    
    if shuffle is True:
         weight_src = []
         female_weight = 17943/28000
         male_weight = 10057/28000
         for i in range(17943):
            weight_src.append(male_weight) 
         for j in range(10057):
            weight_src.append(female_weight)
         imgs_data = random.choices(src_data,weights=weight_src,k=len(src_data))
    #    imgs_data = random.sample(src_data, len(src_data))
    else:
        imgs_data = src_data

    imgs_iters = len(src_data)#all source images

    for i in range(0, imgs_iters, batch_size):

        s_index = i
        e_index = i + batch_size

        if e_index > imgs_iters:
            e_index = imgs_iters

        x_images = []
        x_labels = []
        x_tmasks = []

        for fname, x_label, mname in imgs_data[s_index: e_index]:

            x_image = np.load(fname)
            x_tmask = np.load(mname)

            if flip and np.random.rand() > 0.5:
                x_image = np.fliplr(x_image)
                x_tmask = np.fliplr(x_tmask) #水平翻轉

            x_images.append(x_image)
            x_tmasks.append(x_tmask)
            x_labels.append(x_label)

        x_images = np.array(x_images) / 127.5 - 1
        x_labels = np.array(x_labels)
        x_tmasks = np.array(x_tmasks)

        progress = i / float(imgs_iters - 1)

        yield x_images, x_labels, x_tmasks, progress

def get_ref_loader(ref_data, batch_size, shuffle=True, flip=False):
    
    if shuffle is True:
        #imgs_data = random.sample(ref_data, len(ref_data))
        weight_ref = []
        female_weight = 17943/28000
        male_weight = 10057/28000
        for i in range(17943):
            weight_ref.append(male_weight)
        for j in range(10057):
            weight_ref.append(female_weight)
        imgs_data = random.choices(ref_data,weights=weight_ref,k=len(ref_data))
    else:
        imgs_data = ref_data

    imgs_iters = len(ref_data)

    for i in range(0, imgs_iters, batch_size):

        s_index = i
        e_index = i + batch_size

        if e_index > imgs_iters:
            e_index = imgs_iters

        y1_images = []
        y2_images = []
        labels = []

        for f1_name, f2_name, label in imgs_data[s_index: e_index]:

            y1_image = np.load(f1_name)
            y2_image = np.load(f2_name)
 
            if flip and np.random.rand() > 0.5:
                y1_image = np.fliplr(y1_image)

            if flip and np.random.rand() > 0.5:
                y2_image = np.fliplr(y2_image)

            y1_images.append(y1_image)
            y2_images.append(y2_image)

            labels.append(label)

        y1_images = np.array(y1_images) / 127.5 - 1
        y2_images = np.array(y2_images) / 127.5 - 1
        labels = np.array(labels)

        yield y1_images, y2_images, labels

def get_src_data(set_type, image_size):

    root = os.path.join(f"{dataset_folder}/npy{image_size}", set_type)

    domains = os.listdir(root)#[female  , male]  female:17943 (64%)  male:10057 (36%)
    domain_size = len(domains)#2
    
    fnames = []
    labels = []
    mnames = []


    idmask_folder = os.path.join(f"{dataset_folder}/npy256", "mask")

    for domain_idx, domain in enumerate(sorted(domains)): 

        domain_folder = os.path.join(root, domain)        
        label = np.zeros(shape=(domain_size))
        label[domain_idx] = 1  #(1, 0) or (0, 1)

        cls_names = [os.path.join(domain_folder, fname) for fname in os.listdir(domain_folder)]#  [../../../female/XXX.npy, ../../../female/XXX.npy,....]
        msk_names = [os.path.join(idmask_folder, fname) for fname in os.listdir(domain_folder)]

        fnames += cls_names
        labels += [label] * len(cls_names) # labels = [array([1., 0.]), array([1., 0.]), array([1., 0.]),.......array([0., 1.]), array([0., 1.]), ...........]
        mnames += msk_names
        print(label)
        print(len(cls_names))
        print("-------------------")

    assert len(fnames) == len(labels) #(17943+10057=)
    return list(zip(fnames, labels, mnames))

def get_ref_data(set_type, image_size):

    root = os.path.join(f"{dataset_folder}/npy{image_size}", set_type)

    domains = os.listdir(root)
    domain_size = len(domains)
    
    f1_names = []
    f2_names = []
    labels = []

    for domain_idx, domain in enumerate(sorted(domains)):

        domain_folder = os.path.join(root, domain)
        label = np.zeros(shape=(domain_size))
        label[domain_idx] = 1

        fnames = os.listdir(domain_folder)
        rnames = random.sample(fnames, len(fnames))

        f1_names += [os.path.join(domain_folder, fname) for fname in fnames]
        f2_names += [os.path.join(domain_folder, fname) for fname in rnames] 

        labels += [label] * len(fnames)

    assert len(f1_names) == len(f2_names) == len(labels)
    return list(zip(f1_names, f2_names, labels)) 

def get_val_data(image_size):

    root = os.path.join(f"{dataset_folder}/npy{image_size}", "val")

    domains = os.listdir(root) #female, male
    domain_size = len(domains) #2

    val_data = []
    mnames = []


    idmask_folder = os.path.join(f"{dataset_folder}/npy256", "mask")

    for domain_idx, domain in enumerate(sorted(domains)):

        domain_folder = os.path.join(root, domain)

        label = np.zeros(shape=(domain_size))
        label[domain_idx] = 1

        cls_fnames = [os.path.join(domain_folder, fname) for fname in os.listdir(domain_folder)]
        msk_names = [os.path.join(idmask_folder, fname) for fname in os.listdir(domain_folder)]

        val_data.append(list(zip(cls_fnames, [label] * len(cls_fnames),msk_names)))

    return val_data

def get_val_loader(val_data, batch_size, shuffle=True):

    imgs_data = []
    num = 5*batch_size

    for data in val_data:
        imgs_data += data

    if shuffle is True:
        #imgs_data = random.sample(imgs_data, len(imgs_data))
        imgs_data = random.sample(imgs_data, 50)

    imgs_iters = len(imgs_data) #2000

    for i in range(0, imgs_iters, batch_size):
        
        if num != None and i > num:
                break
		
        s_index = i
        e_index = i + batch_size

        if e_index > imgs_iters:
            e_index = imgs_iters

        x_images = []
        x_labels = []
        x_tmasks = []

        for fname, label, mname in imgs_data[s_index: e_index]:
            x_image = np.load(fname) / 127.5 - 1
            x_images.append(x_image)
            x_labels.append(label)
            x_tmask = np.load(mname)
            x_tmasks.append(x_tmask)
        x_tmasks = np.array(x_tmasks)
        y_data = [] # be as reference data

        for data in val_data: #2  
            fname, y_label, mname= data[s_index // len(val_data)]  # len(val_data) = 2
            y_image = np.load(fname) / 127.5 - 1
            y_data.append((y_image, y_label))

        yield x_images, x_labels, x_tmasks, y_data

def save_npy(load_dir, save_dir, resolution):

    load_root = load_dir
    save_root = os.path.join("data", save_dir)

    if not os.path.exists(save_root):
        os.makedirs(save_root) 

    for fname in os.listdir(load_root):
        image = Image.open(os.path.join(load_root, fname))
        image = image.resize((resolution, resolution), Image.BILINEAR)

        i_npy = np.array(image)

        fname = os.path.splitext(fname)[0]
        fname = fname + ".npy"

        np.save(os.path.join(save_root, fname), i_npy)

def save_mask(load_dir, mapping_dir):

    mapping_file = open(mapping_dir, 'r')
    mapping_table = []

    first_line = mapping_file.readline()

    for line in mapping_file.readlines():
        mapping_table.append(line.split()[2])

    id_folder = "data/jpeg/id-mask-HQ"
    #cs_folder = "data/jpeg/cs-mask-HQ"
    cs_folder = "data_print"

    if not os.path.exists(id_folder):
        os.makedirs(id_folder)

    if not os.path.exists(cs_folder):
        os.makedirs(cs_folder)

    id_save_root = os.path.join(id_folder)
    cs_save_root = os.path.join(cs_folder)
    
    
    dilate_range = 20 # original = 20
    
    base = ndimage.generate_binary_structure(2, 1)# 3*3
    struct = ndimage.iterate_structure(base, 2*dilate_range+1) #41*41
     
    erosion_range = 5
    base_e = ndimage.generate_binary_structure(2, 1)# 3*3
    struct_e = ndimage.iterate_structure(base_e, 2*erosion_range+1) #11*11
    
    for i in range(15):
        jpg_folder = os.path.join(load_dir, f"{i}")

        for j in range(2000):

            file_idx = i * 2000 + j

            mask = np.zeros((512, 512))
            
            name = mapping_table[file_idx]

            for anno in ["nose", "l_eye", "r_eye", "l_brow", "r_brow", "mouth", "u_lip", "l_lip"]:

                anno_path = os.path.join(jpg_folder, f"{file_idx:05d}_{anno}.png")

                if os.path.exists(anno_path):
                    anno_mask = Image.open(anno_path).convert('L')
                    anno_mask = np.array(anno_mask)

                    mask = mask + anno_mask

            imask = np.clip(mask, 0, 255)
            store_imask = imask.astype(np.uint8)
            imask_name = "_imask_"+name
            Image.fromarray(store_imask).convert('L').save(os.path.join(cs_save_root, imask_name))

            for anno in ["skin", "neck_l", "neck", "l_ear", "r_ear"]:

                anno_path = os.path.join(jpg_folder, f"{file_idx:05d}_{anno}.png")
                
                if os.path.exists(anno_path):
                    anno_mask = Image.open(anno_path).convert('L')
                    anno_mask = np.array(anno_mask)

                    mask = mask + anno_mask

            # abandon background, eye_g, hair, cloth, and ear_r

            cmask = np.clip(mask, 0, 255)
            
            store_cmask = cmask.astype(np.uint8)
            cmask_name = "_cmask_"+name
            Image.fromarray(store_cmask).convert('L').save(os.path.join(cs_save_root, cmask_name))
            
            #### change the coarse mask
            new_imask = imask/255
            cmask = cmask/255
            
            #dilated_kmask = ndimage.grey_dilation(new_imask, size=(dilate_range, dilate_range), structure=np.zeros((dilate_range, dilate_range)))
            dilated_kmask = ndimage.binary_dilation(new_imask,structure=struct).astype(new_imask.dtype)
            
            store_dilated_kmask = (dilated_kmask*255).astype(np.uint8)
            dilated_kmask_name = "_dilated_kmask_"+name
            Image.fromarray(store_dilated_kmask).convert('L').save(os.path.join(cs_save_root, dilated_kmask_name))
            
            ero_kmask = ndimage.binary_erosion(dilated_kmask,structure=struct_e).astype(dilated_kmask.dtype)
            
            store_ero_kmask = (ero_kmask*255).astype(np.uint8)
            ero_kmask_name = "_ero_kmask_"+name
            Image.fromarray(store_ero_kmask).convert('L').save(os.path.join(cs_save_root, ero_kmask_name))
                       
           
            final_mask = np.multiply(ero_kmask, cmask)
            final_mask = np.clip(final_mask,0,1)
            
            store_final_mask = (final_mask*255).astype(np.uint8)
            final_mask_name = "_final_mask_"+name
            Image.fromarray(store_final_mask).convert('L').save(os.path.join(cs_save_root, final_mask_name))
            
            final_mask_gau = ndimage.gaussian_filter(final_mask,sigma = 15)
            coarse_mask = (final_mask_gau * 255).astype(np.uint8)
            
            #### change the key mask 
            #gau_imask = ndimage.gaussian_filter(new_imask,sigma = 5)
            #final_keymask = (gau_imask * 255).astype(np.uint8)
            
            #Image.fromarray(final_keymask).convert('L').save(os.path.join(id_save_root, ))
            Image.fromarray(coarse_mask).convert('L').save(os.path.join(cs_save_root, mapping_table[file_idx]))

def save_masks_to_npy(id_dir, cs_dir, save_dir, resolution):
    
    save_root = os.path.join("data", save_dir)
    idmk_root = os.path.join("data", id_dir)
    csmk_root = os.path.join("data", cs_dir)

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    for iname, cname in zip(os.listdir(idmk_root), os.listdir(csmk_root)):
        imask = Image.open(os.path.join(idmk_root, iname)).convert("L")
        cmask = Image.open(os.path.join(csmk_root, cname)).convert("L")

        imask = imask.resize((resolution, resolution), Image.BILINEAR)
        cmask = cmask.resize((resolution, resolution), Image.BILINEAR)

        imask = np.array(imask) / 255
        cmask = np.array(cmask) / 255
        
        fname = os.path.splitext(iname)[0]
        fname = fname + ".npy"

        np.save(os.path.join(save_root, fname), np.stack((imask, cmask), axis=-1))

if __name__ == '__main__':
    ### Prepare for face mask and key mask  
    ### Transfer original dataset to face mask and key mask in jpeg
    save_mask(load_dir="/local_path/stargan-v2-master/CelebAMask-HQ/CelebAMask-HQ-mask-anno", mapping_dir="/local_path/stargan-v2-master/CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt")
    ### Transfer jpeg to npy
    save_masks_to_npy(id_dir="jpeg/key-mask-HQ", cs_dir="jpeg/face-mask-HQ", save_dir="npy256/mask", resolution=256)
    
    ### Prepare for Training and Testing set
    #Testing set for female (1000)
    save_npy(load_dir="/Download/CelebA-HQ/val/female", save_dir="npy256/val/female", resolution=256)
    #Testing set for male (1000)
    save_npy(load_dir="/Download/CelebA-HQ/val/male", save_dir="npy256/val/male", resolution=256)
    #Training set for female (17943) 
    save_npy(load_dir="/Download/CelebA-HQ/train/female", save_dir="npy256/train/female", resolution=256)
    #Training set for male (10057)
    save_npy(load_dir="/Download/CelebA-HQ/train/male", save_dir="npy256/train/male", resolution=256)
    