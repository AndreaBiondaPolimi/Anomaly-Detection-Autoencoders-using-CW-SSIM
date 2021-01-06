
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

def load_patches_from_file (file, patch_size, random, n_patches=3, stride=32, cut_size=None, preprocess_limit = 100, resize=None):
    im1 = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    if (resize is not None):
        width = int(im1.shape[1] * resize)
        height = int(im1.shape[0] * resize)
        im1 = cv2.resize(im1, (width, height))

    if (cut_size is not None):
        im1 = im1[cut_size[0]:cut_size[1], cut_size[2]:cut_size[3]]

    cropped = []
    if (random == True):
        for _ in range (n_patches):
            j = np.random.randint(0, im1.shape[0] - patch_size)
            i = np.random.randint(0, im1.shape[1] - patch_size)
            if (check_preprocessing(im1[j:j+patch_size, i:i+patch_size], preprocess_limit)):
                cropped.append(im1[j:j+patch_size, i:i+patch_size])
    else:
        for j in range (int((im1.shape[0] - patch_size) / stride) + 1):
            for i in range (int((im1.shape[1] - patch_size) / stride) + 1):
                cropped.append(im1[(j*stride):(j*stride)+patch_size, (i*stride):(i*stride)+patch_size])

    return cropped, im1


def load_patches_from_file_fixed (file, patch_size, positions):
    im1 = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    patches = []
    for pos in positions:
        patches.append (im1[pos[0]:pos[0]+patch_size, pos[1]:pos[1]+patch_size])
    return patches


def load_patches (folder, patch_size, random=True, n_patches=3, stride=32, cut_size=None, preprocess_limit = 100, resize=None):
    patches = []
    for file in os.listdir(folder):
        if file.endswith(".bmp") or file.endswith(".tif"):
            ret, img = load_patches_from_file(os.path.join(folder, file), patch_size, random, n_patches, stride, cut_size, preprocess_limit, resize)
            for r in ret:
                patches.append(r)
    return patches

def load_gt_from_file (file, cut_size=None):
    im1 = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if (cut_size is not None):
        im1 = im1[cut_size[0]:cut_size[1], cut_size[2]:cut_size[3]]
    return im1/255

def show_patches (patches):
    for img in patches:
        plt.imshow(img)
        plt.show()

def check_preprocessing (patch, preprocess_limit=100):
    return True if np.median(patch) > preprocess_limit else False
    #return True if np.median(patch) > 80 else False


if __name__ == "__main__":
    valid_patches = load_patches_from_file_fixed('Dataset\\SEM_Data\\Anomalous\\images\\ITIA1108.tif', patch_size=64, 
        positions = ((11,468),(131,365),(328,848))) 

    show_patches(valid_patches)