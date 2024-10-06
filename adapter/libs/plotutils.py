import skimage

import torch
import torchvision
import torchvision.transforms as transforms


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Callable, Dict, List, Tuple, Union

mean_image = [0.48145466, 0.4578275, 0.40821073]
std_image = [0.26862954, 0.26130258, 0.27577711]


class NormalizeInverse(torchvision.transforms.Normalize):

    def __init__(self, mean: List[float], std: List[float]) -> None:
        """Reconstructs the images in the input domain by inverting 
        the normalization transformation.

        Args:
            mean: the mean used to normalize the images.
            std: the standard deviation used to normalize the images.
        """
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


def show_grid(dataset: torchvision.datasets.ImageFolder,
              process: Callable = None,
              rand_num=10,
              size_num=10) -> None:
    """Shows a grid with random images taken from the dataset.

    Args:
        dataset: the dataset containing the images.
        process: a function to apply on the images before showing them.        
    """
    fig = plt.figure(figsize=(15, 5))
    indices_random = np.random.randint(rand_num,
                                       size=size_num,
                                       high=len(dataset))
    #indices_random= [x for x in range(8)]

    for count, idx in enumerate(indices_random):
        fig.add_subplot(2, 5, count + 1)
        title = dataset.classes[dataset[idx][1]]
        plt.title(title)
        image_processed = process(
            dataset[idx][0]) if process is not None else dataset[idx][0]
        plt.imshow(transforms.ToPILImage()(image_processed))
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# denormalize = NormalizeInverse(mean_image, std_image)
# show_grid(dataset, process=denormalize)

# 展示图像与所属类别概率
def show_image_and_probs(the_logits,
                         test_set,
                         rand_num=10,
                         size_num=10,
                         process=None,
                         device='cuda'):
    image_input = torch.tensor(
        np.stack([test_set[x][0] for x in range(len(test_set))])).to(device)
    labels = torch.tensor(
        np.stack([test_set[x][1] for x in range(len(test_set))]))
    # the 100.0 works as temperature parameter, raising the softmax confidence
    # text_probs_notens = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    # text_probs = ( 100.0 * image_features @ text_features_ensembled.T).softmax(dim=-1)
    text_probs = (100.0 * the_logits).softmax(dim=-1)

    top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)
    # top_probs_n, top_labels_n = text_probs_notens.cpu().topk(5, dim=-1)

    plt.figure(figsize=(19, 19))
    #taking random index for random sampling
    random_index = np.random.randint(rand_num,
                                     size=size_num,
                                     high=len(image_input))
    image_input_samples = [(image_input[x], labels[x], x)
                           for x in random_index]
    for i, input_sample in enumerate(image_input_samples):

        plt.subplot(len(image_input_samples), 4, 2 * i + 1)
        #denormalizing the image and transforming to PIL image
        plt.title(test_set.classes[input_sample[1].cpu()])
        image_processed = process(input_sample[0].cpu())
        plt.imshow(torchvision.transforms.ToPILImage()(image_processed))

        plt.axis("off")

        plt.subplot(len(image_input_samples), 4, 2 * i + 2)
        y = np.arange(top_probs.shape[-1])
        plt.grid()
        plt.barh(y, top_probs[input_sample[2]])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [
            test_set.classes[index]
            for index in top_labels[input_sample[2]].numpy()
        ])
        # plt.xlabel("probability")
        plt.subplots_adjust(hspace=0.28) # the amount of height reserved for space between subplots
    plt.subplots_adjust(
        wspace=0.30
    )  # 左右子图保留的空间宽度the amount of width reserved for space between subplots
    plt.show()
