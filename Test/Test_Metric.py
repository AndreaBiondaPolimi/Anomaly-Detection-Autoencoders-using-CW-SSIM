from DataLoader import load_patches
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Steerables.metrics_TF import Metric, Metric_win
import tensorflow as tf

def get_p_at_1 (patches, index, ref):
    batch_ref = [patches[index]] * len(patches)
    batch_eval = patches

    batch_ref = np.expand_dims(batch_ref, axis=-1)
    batch_eval = np.expand_dims(batch_eval, axis=-1)

    stsim_scores_tf = metric_tf.CWSSIM(batch_ref, batch_eval, height=5, orientations=5, full=True)
    #stsim_scores_tf = metric_tf.CWSSIM(batch_ref, batch_eval, height=5, orientations=5)

    print (tf.shape(stsim_scores_tf))
    
    scores = stsim_scores_tf.numpy()
    scores[index] = np.min(scores)

    idx_max = np.argmax(scores)
    print ('index:',index,'idx_max:',idx_max, 'ref:',ref)

    if (idx_max >= ref and idx_max <= ref+2):
        print('1')
        return 1
    else:
        print('0')
        return 0

if __name__ == "__main__":

    patches = load_patches('Dataset\\CUReT_Data', patch_size=128, n_patches=3, random=True, cut_size=(140,340,200,500), preprocess_limit=0)

    """
    for img in patches:
        plt.imshow(img)
        plt.show()
    """

    corrects = 0
    metric_tf = Metric_win (window_size=7)
    #metric_tf = Metric_win ()
    for idx in range (len (patches)):
        corrects += get_p_at_1 (patches, idx, int(idx/3) * 3)
        
    print(corrects/len (patches))


#STSIM1 global 0.85
#STSIM1 win(7) 0.74
#STSIM2 global 0.98
#STSIM2 win(7) 0.78
#STSIM-M       0.95