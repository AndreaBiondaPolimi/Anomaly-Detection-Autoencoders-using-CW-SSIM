import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import albumentations as A

def visualize_results (img, res, txt):
    f = plt.figure(figsize=(17, 7))

    f.add_subplot(1,2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title("Image")
    plt.imshow(img)

    f.add_subplot(1,2, 2)
    plt.xticks([])
    plt.yticks([])
    plt.title("Reconstructed")
    plt.imshow(res)

    f.text(.5, .05, txt, ha='center')
    plt.show()

# Preprocess and evaluation ###
def preprocess_data (patches, expand=True):
    x = np.array(patches)
    x = x.astype('float64') / 255.
    if expand:
        x = x[..., tf.newaxis]
    return x

def batch_evaluation (valid_patches, autoencoder, ae_batch_splits):
    x_valid = preprocess_data(valid_patches)
    high = int(len(x_valid) / ae_batch_splits) * ae_batch_splits
    if high == len(x_valid):
        x_batches = np.split(x_valid, ae_batch_splits, axis=0)
    else:
        x_batches = np.split(x_valid[:high], ae_batch_splits, axis=0)
    x_batches.append(x_valid[high:])
    
    y_valid = autoencoder(x_batches[0]).numpy() 
    for i in range(1, len(x_batches)):
        y_valid = np.vstack((y_valid, autoencoder(x_batches[i]).numpy()))  

    y_valid = np.squeeze(y_valid)
    x_valid = np.squeeze(x_valid)
    return x_valid, y_valid

def image_evaluation (valid_img, autoencoder):
    #padding = A.PadIfNeeded(1024, 1024, p=1.0)

    valid_img = valid_img/255
    valid_img = valid_img[tf.newaxis, ..., tf.newaxis]
    reco = autoencoder(valid_img).numpy() 
    reco = np.squeeze(reco)

    return reco

### Postprocess ###
def bg_mask(img, value, mode):
    _,thresh=cv2.threshold(img,value,255,mode)

    def FillHole(mask):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        len_contour = len(contours)
        contour_list = []
        for i in range(len_contour):
            drawing = np.zeros_like(mask, np.uint8)  # create a black image
            img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
            contour_list.append(img_contour)

        out = sum(contour_list)
        return out

    thresh = FillHole(thresh)
    if type(thresh) is int:
        return np.ones(img.shape)
    mask_ = np.ones(thresh.shape)
    mask_[np.where(thresh <= 127)] = 0
    return mask_

def post_reconstruction (y_valid, loss_type):
    if (loss_type == 'cwssim_loss'):
        y_valid = y_valid * -1
        #y_valid = (y_valid - np.min(y_valid)) / (np.max(y_valid) - np.min(y_valid))
    return y_valid


### Performance Evaluations ###
def get_performance (y_true, y_pred):
    y_true = np.array (y_true, dtype=int)
    y_pred = np.array (y_pred, dtype=int)

    iou = iou_coef(y_true, y_pred)
    tpr, fpr = roc_coef (y_true, y_pred)
    ovr = ovr_coef (y_true, y_pred) if fpr <= 0.05 else None
    return iou, tpr, fpr, ovr

def get_roc (y_true, y_pred):
    y_true = np.array (y_true, dtype=int)
    y_pred = np.array (y_pred, dtype=int)

    tpr, fpr = roc_coef (y_true, y_pred)
    return tpr, fpr

def get_ovr (y_true, y_pred):
    y_true = np.array (y_true, dtype=int)
    y_pred = np.array (y_pred, dtype=int)

    ovr = ovr_coef (y_true, y_pred)
    return ovr

def get_iou(y_true, y_pred):
    y_true = np.array (y_true, dtype=int)
    y_pred = np.array (y_pred, dtype=int)

    iou = iou_coef(y_true, y_pred)
    return iou

def iou_coef(y_true, y_pred):
    intersection = np.logical_and(y_true,y_pred) # Logical AND
    union = np.logical_or(y_true,y_pred)    # Logical OR
    IOU = float(np.sum(intersection)/np.sum(union))
    return IOU

def roc_coef (y_true, y_pred):
    tp = np.sum (y_true*y_pred)
    fn = np.sum ((y_true - y_pred).clip(min=0))
    tpr = tp / (tp + fn)

    fp = np.sum ((y_pred - y_true).clip(min=0))
    tn = np.sum ((1-y_true)*(1-y_pred))
    fpr = fp / (fp + tn)

    return tpr, fpr

def ovr_coef (y_true, y_pred):
    r,l = cv2.connectedComponents(np.array(y_true*255, np.uint8))
    blobs = [np.argwhere(l==i) for i in range(1,r)]
    ovrs = []

    for blob in blobs:
        ground = np.zeros_like(y_true, dtype=int)
        for b in blob:
            ground[b[0],b[1]] = 1
        
        tp = np.sum (ground*y_pred)
        ovr = tp / np.sum(ground)
        ovrs.append (ovr)

    return np.mean(ovrs)    