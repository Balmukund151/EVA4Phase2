#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

from __future__ import print_function

import copy
import os.path as osp

import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms

from grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def load_images(image_paths):
    images = []
    raw_images = []
    print("Images:",image_paths)
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images


def get_classtable():
    classes = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    #with open("samples/synset_words.txt") as lines:
        #for line in lines:
            #line = line.strip().split(" ", 1)[1]
            #line = line.split(", ", 1)[0].replace(" ", "_")
            #classes.append(line)
    return classes


def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    #raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)


# torchvision models
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


def demo1(image_paths, target_layer, arch, topk, output_dir, cuda, model):
    """
    Visualize model responses given multiple images
    """

    device = get_device(cuda)
    gradcam_list=[]

    # Synset words
    classes = get_classtable()

    # Model from torchvision
    if model is None:
        model = models.__dict__[arch](pretrained=True)
    model.to(device)
    model.eval()

    # Images
    #print(image_paths)
    images, raw_images = load_images(image_paths)
    images = torch.stack(images).to(device)

    """
    Common usage:
    1. Wrap your model with visualization classes defined in grad_cam.py
    2. Run forward() with images
    3. Run backward() with a list of specific classes
    4. Run generate() to export results
    """

    
    # =========================================================================
    print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")
    bp = BackPropagation(model=model)
    probs, ids = bp.forward(images)
    gcam = GradCAM(model=model)
    _ = gcam.forward(images)

    gbp = GuidedBackPropagation(model=model)
    _ = gbp.forward(images)

    for i in range(topk):
        # Guided Backpropagation
        gbp.backward(ids=ids[:, [i]])
        gradients = gbp.generate()

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            #print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))
            
            # Grad-CAM
            filename=osp.join(output_dir,"{}-{}-gradcam-{}.png".format(j, arch, target_layer))
            gradcam_list.append(filename)
            save_gradcam(filename, gcam=regions[j, 0], raw_image=raw_images[j])
    return gradcam_list        


def demo2(image_paths, output_dir, cuda):
    """
    Generate Grad-CAM at different layers of ResNet-152
    """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model
    model = models.resnet152(pretrained=True)
    model.to(device)
    model.eval()

    # The four residual layers
    target_layers = ["relu", "layer1", "layer2", "layer3", "layer4"]
    target_class = 243  # "bull mastif"

    # Images
    images, raw_images = load_images(image_paths)
    images = torch.stack(images).to(device)

    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(images)
    ids_ = torch.LongTensor([[target_class]] * len(images)).to(device)
    gcam.backward(ids=ids_)

    for target_layer in target_layers:
        print("Generating Grad-CAM @{}".format(target_layer))

        # Grad-CAM
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print(
                "\t#{}: {} ({:.5f})".format(
                    j, classes[target_class], float(probs[ids == target_class])
                )
            )

            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, "resnet152", target_layer, classes[target_class]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )


def demo3(image_paths, arch, topk, stride, n_batches, output_dir, cuda):
    """
    Generate occlusion sensitivity maps
    """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model from torchvision
    model = models.__dict__[arch](pretrained=True)
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    # Images
    images, _ = load_images(image_paths)
    images = torch.stack(images).to(device)

    print("Occlusion Sensitivity:")

    patche_sizes = [10, 15, 25, 35, 45, 90]

    logits = model(images)
    probs = F.softmax(logits, dim=1)
    probs, ids = probs.sort(dim=1, descending=True)

    for i in range(topk):
        for p in patche_sizes:
            print("Patch:", p)
            sensitivity = occlusion_sensitivity(
                model, images, ids[:, [i]], patch=p, stride=stride, n_batches=n_batches
            )

            # Save results as image files
            for j in range(len(images)):
                print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

                save_sensitivity(
                    filename=osp.join(
                        output_dir,
                        "{}-{}-sensitivity-{}-{}.png".format(
                            j, arch, p, classes[ids[j, i]]
                        ),
                    ),
                    maps=sensitivity[j],
                )