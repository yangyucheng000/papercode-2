import sys
import os
import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.cm as cm 
from matplotlib import rcParams

from mindcv.models import mae, model_factory


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])
colors = cm.rainbow(np.linspace(0, 1, 12))

def compute_distance_matrix(patch_size, num_patches, length):
    """compute_distance_matrix: Computes the distance matrix for the patches in the image

    Args:
        patch_size (int): the size of the patch
        num_patches (int): the number of patches in the image
        length (int): the length of the image

    Returns:
        distance_matrix (np.ndarray): The distance matrix for the patches in the image
    """
    distance_matrix = np.zeros((num_patches, num_patches))
    for i in range(num_patches):
        for j in range(num_patches):
            if i == j:  # zero distance
                continue

            xi, yi = (int(i / length)), (i % length)
            xj, yj = (int(j / length)), (j % length)
            distance_matrix[i, j] = patch_size * np.linalg.norm([xi - xj, yi - yj])

    return distance_matrix


def compute_mean_attention_dist(patch_size, attention_weights, num_cls_tokens=1):
    """compute_mean_attention_dist: Computes the mean attention distance for the image

    Args:
        patch_size (int): the size of the patch
        attention_weights (np.ndarray): The attention weights for the image
        num_cls_tokens (int, optional): The number of class tokens. Defaults to 1.

    Returns:
        mean_distances (np.ndarray): The mean attention distance for the image
    """
    # The attention_weights shape = (batch, num_heads, num_patches, num_patches)
    attention_weights = attention_weights[
        ..., num_cls_tokens:, num_cls_tokens:
    ]  # Removing the CLS token
    num_patches = attention_weights.shape[-1]
    length = int(np.sqrt(num_patches))
    assert length ** 2 == num_patches, "Num patches is not perfect square"

    distance_matrix = compute_distance_matrix(patch_size, num_patches, length)
    h, w = distance_matrix.shape

    distance_matrix = distance_matrix.reshape((1, 1, h, w))
    # The attention_weights along the last axis adds to 1
    # this is due to the fact that they are softmax of the raw logits
    # summation of the (attention_weights * distance_matrix)
    # should result in an average distance per token
    mean_distances = attention_weights * distance_matrix
    mean_distances = np.sum(
        mean_distances, axis=-1
    )  # sum along last axis to get average distance per token
    mean_distances = np.mean(
        mean_distances, axis=-1
    )  # now average across all the tokens

    return mean_distances



def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def prepare_model(arch='mae_b_16_224_pretrain'):
    # load model
    model = model_factory.create_model(
        model_name=arch,
        # num_classes=num_classes,
        in_channels=3,
        drop_rate=0.0,
        drop_path_rate=0.0,
        pretrained=False,
        checkpoint_path='/pathtomae_b_16_224_pretrain-400_1562.ckpt',
        ema=False,
    )
    return model

def run_one_image(img, model):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    attn_weights = model.forward_encoder_test(x.float())
    for i in range(0,12):
        mean_dist=(compute_mean_attention_dist(16,attn_weights[i].detach().numpy()))
        x = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
        plt.scatter(x, mean_dist, color = colors[i], label='Head' + str(i))
        plt.legend(loc="upper right")
    plt.title("Attention Distance", fontsize=20)
    plt.xlabel("Depth")
    plt.ylabel("Attention Distance") 
    plt.show()

def main():
    rcParams['figure.figsize'] = (12, 12)
    img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg' # fox, from ILSVRC2012_val_00046145
    img = Image.open(requests.get(img_url, stream=True).raw)
    img = img.resize((224, 224))
    img = np.array(img) / 255.

    assert img.shape == (224, 224, 3)

    # normalize by ImageNet mean and std
    img = img - imagenet_mean
    img = img / imagenet_std

    # plt.rcParams['figure.figsize'] = [5, 5]
    # show_image(torch.tensor(img))
    model = prepare_model('mae_b_16_224_pretrain')
    print('Model loaded.')

    torch.manual_seed(2)
    run_one_image(img, model)


if __name__ == "__main__":
    main()
