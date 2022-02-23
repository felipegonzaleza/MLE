from typing import List
import matplotlib.pyplot as plt
import numpy as np
import cv2
from HomomorphicFilter import HomomorphicFilter
from tensorflow_addons.metrics import F1Score
from sklearn.metrics import precision_score, recall_score

import tensorflow as tf
from tensorflow import keras
import matplotlib.cm as cm

COLAB = 'google.colab' in str(get_ipython())

# Show a Grid of RGB/G Images
def show_imgs(imgs: List, title: str=None, cols: int=4, scale: float=1.0,
              gray=False) -> None:
    scaling = scale * 8 / cols
    num_imgs = len(imgs)
    rows = ((num_imgs - 1) // 4) + 1
    height = rows*scaling + (0.45 if title else 0.3)
    plt.figure(figsize=(cols*scaling, height))
    for i, img in enumerate(imgs):
        ax = plt.subplot(rows, cols, i+1)
        ax.axis('off')
        ax.imshow(img, cmap=('gray' if gray else None))
    if COLAB:
      print(' ', title)
    else:
      plt.suptitle(title, fontsize=16)
    plt.tight_layout()

# Show a Table of Images showing Individual Channels
def show_imgs_chs(imgs: List, title: str=None, scale: float=1.0) -> None:
    num_chns = imgs[0].shape[-1]
    num_imgs = len(imgs)
    scaling = scale * 8 / num_chns
    height = num_imgs*scaling + (0.45 if title else 0.3)
    plt.figure(figsize=(num_chns*scaling, height))
    for i, img in enumerate(imgs):
        for ch in range(num_chns):
            ax = plt.subplot(num_imgs, num_chns, i*num_chns + ch+1)
            ax.axis('off')
            ax.imshow(img[:,:,ch], cmap='gray')
    if COLAB:
      print(' ', title)
    else:
      plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
# Show a Grid of Channel Activations
def show_activations(img_chans: np.ndarray, title: str=None, cols: int=8,
                     scale: float=1.0, gray=True) -> None:
    scaling = scale * 8 / cols
    rows = ((img_chans.shape[-1] - 1) // cols) + 1
    height = rows*scaling + (0.45 if title else 0.3)
    plt.figure(figsize=(cols*scaling, height))
    for ch in range(img_chans.shape[-1]):
        ax = plt.subplot(rows, cols, ch+1)
        ax.axis('off')
        ax.imshow(img_chans[..., ch], cmap=('gray' if gray else None))
    if COLAB:
      print(' ', title)
    else:
      plt.suptitle(title, fontsize=16)
    plt.tight_layout()

# RGB to HSI (Hue, Saturation, Intensity) Color Transform
def rgb2hsi(img: np.ndarray) -> np.ndarray:
    assert img.dtype == np.uint8

    eps = np.finfo(float).eps
    img = img.astype(np.float64)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    
    num = 0.5*((r - g) + (r - b))
    den = np.sqrt((r - g)**2 + (r - b)*(g - b))
    H = np.arccos(num/(den + eps))
    H[b > g] = 2*np.pi - H[b > g]
    H = H/(2*np.pi)*255
    
    min_rgb = np.minimum(np.minimum(r, g), b)
    rgb = r + g + b
    rgb[rgb == 0] = eps
    S = (1 - 3*min_rgb/rgb)*255
    
    I = rgb/3
    
    # H[S < 5] = 0
    return np.stack((H, S, I), axis=-1).astype(np.uint8)

# Apply CLAHE Adaptive Contrast Enhancement on image
def clahe(img: np.ndarray, clip=2, clahe_inst=None) -> np.ndarray:
    assert img.dtype == np.uint8
    clahe = clahe_inst or cv2.createCLAHE(clipLimit=clip)
    clh_r = clahe.apply(img[:,:,0])
    clh_g = clahe.apply(img[:,:,1])
    clh_b = clahe.apply(img[:,:,2])
    return np.stack((clh_r, clh_g, clh_b), axis=-1)

# Apply Homomorphic Filter on image
def homomorphic(img: np.ndarray, a=0.99, b=1.5, f_cut=20,
                filt_inst=None) -> np.ndarray:
    assert img.dtype == np.uint8
    filt = filt_inst or HomomorphicFilter(a, b, f_cut)
    img_r = filt.apply(img[:,:,0])
    img_g = filt.apply(img[:,:,1])
    img_b = filt.apply(img[:,:,2])
    return np.stack((img_r, img_g, img_b), axis=-1)

# Compute and Display Multi-Label Prediction-vs-Actual stats
def disp_multilabel_stats(model, classes:List[str], xy_iter, thresh: float) -> None:
    count = 0
    actual = np.zeros(len(classes), dtype='int32')
    predictions = np.zeros(len(classes), dtype='int32')
    for _ in range(len(xy_iter)):
        print('.', end='')
        x, y = next(xy_iter)
        count += len(y)
        preds = model.predict(x)
        predictions += np.sum(preds > thresh, axis=0)
        actual += np.sum(y, axis=0)
    print(f'\nTotal Samples: {count}')
    print(f'Total Positives: {np.sum(actual)} (Actual), {np.sum(predictions)} (Predict)')
    print('CLASS:  ACTUAL#, PREDICT#')
    print('='*25)
    for i, (num_actual, num_pred) in enumerate(zip(actual, predictions)):
        print(f'{classes[i]:15}: {num_actual:3}, {num_pred:3}')

# Compute and Display Multi-Label Precision and Recall scores
def disp_multilabel_scores(model, classes:List[str], xy_iter, thresh: float) -> None:
    actuals = []
    preds = []
    for _ in range(len(xy_iter)):
        print('.', end='')
        x, y = next(xy_iter)
        actuals.append(y)
        preds.append(model.predict(x) > thresh)
    actuals = np.concatenate(actuals)
    preds = np.concatenate(preds)
    print('\nCLASS:    PRECISION, RECALL')
    print('='*27)
    for i, klass in enumerate(classes):
        precision = precision_score(actuals[:, i], preds[:, i])
        recall = recall_score(actuals[:, i], preds[:, i])
        print(f'{klass:15}: {precision*100:.1f}, {recall*100:.1f}')

# Create Grad-CAM heatmap of single image
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Create <heatmap> superimposed on <img>
def superimpose_gradcam(img, heatmap, alpha=0.4):
    # Rescale img and heatmap to a range 0-255
    img = np.uint8(255 * img)
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    return (jet_heatmap * alpha + img).astype(np.uint8)

# Get Heatmaps from a list of images
def get_heatmaps(model, imgs_list, conv_lyr_name, klass: int):
    heatmaps = []
    for img in imgs_list:
        assert isinstance(img, np.ndarray)
        x = tf.constant(np.expand_dims(img, axis=0))
        heatmaps.append(make_gradcam_heatmap(x, model, conv_lyr_name, klass))
    return heatmaps

# Get Superimpositions from a list of images and heatmaps
def get_superimps(imgs, heatmaps):
    return [superimpose_gradcam(img, heatmap)
            for img, heatmap in zip(imgs, heatmaps)]

# Get a list of images, heatmaps and superimpositions for display
def get_gradcam_grid(model, imgs, conv_lyr_name: str, klass: int):
    output = imgs[:]
    imgs_np = np.array(imgs)
    heatmaps = get_heatmaps(model, imgs_np, conv_lyr_name, klass)
    output.extend(heatmaps)
    superimps = get_superimps(imgs_np, heatmaps)
    output.extend(superimps)
    return output

# Return <num> most confident True Positive images of <klass>
def most_confident(model, klass:int , xy_iter, thresh: float, num: int):
    imgs = []
    for _ in range(len(xy_iter)):
        print('.', end='')
        x, y = next(xy_iter)
        preds = model.predict(x)
        for x, y, pred in zip(x, y, preds):
            if y[klass] and pred[klass] > thresh:
                imgs.append((pred[klass], x))
    print(f'\nFound {len(imgs)} True Positives for Class# {klass}')
    imgs.sort(key=lambda x: x[0], reverse=True)
    print('Confidences:', [f'{x[0]:.3f}' for x in imgs[:num]])
    return [tup[1] for tup in imgs[:num]]
