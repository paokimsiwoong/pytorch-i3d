import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-mode", type=str, help="rgb or flow")
parser.add_argument("-load_model", type=str)
parser.add_argument("-root", type=str)
parser.add_argument("-gpu", type=str)
parser.add_argument("-save_dir", type=str)

args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = "cuda"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms


import numpy as np

from pytorch_i3d import InceptionI3d

# from charades_dataset_full import Charades as Dataset
from charades_dataset_full import Charades_npy as Dataset

from tqdm import tqdm


def run(
    max_steps=64e3,
    mode="rgb",
    root="/data/ephemeral/home/pytorch-i3d/data/rgbs",
    split="/data/ephemeral/home/pytorch-i3d/data/sample.json",
    batch_size=1,
    load_model="/data/ephemeral/home/pytorch-i3d/models/rgb_charades.pt",
    save_dir="/data/ephemeral/home/pytorch-i3d/sample_feat",
):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(split, "training", root, mode, test_transforms, num=-1, save_dir=save_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )

    # val_dataset = Dataset(split, "testing", root, mode, test_transforms, num=-1, save_dir=save_dir)
    # val_dataloader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    # )
    val_dataset = None
    val_dataloader = None

    dataloaders = {"train": dataloader, "val": val_dataloader}
    datasets = {"train": dataset, "val": val_dataset}

    # setup the model
    if mode == "flow":
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(load_model))
    print(f"==>> torch.cuda.is_available(): {torch.cuda.is_available()}")
    i3d.to(device="cuda")

    for phase in ["train"]:
        i3d.train(False)  # Set model to evaluate mode

        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0

        print(f"==>> len(dataloaders[phase]): {len(dataloaders[phase])}")

        # Iterate over data.
        for data in dataloaders[phase]:
            # get the inputs
            inputs, labels, name = data
            if os.path.exists(os.path.join(save_dir, name[0] + ".npy")):
                continue

            b, c, t, h, w = inputs.shape
            if t > 1600:
                features = []
                for start in range(1, t - 56, 1600):
                    end = min(t - 1, start + 1600 + 56)
                    start = max(1, start - 48)
                    ip = Variable(torch.from_numpy(inputs.numpy()[:, :, start:end]).cuda(), volatile=True)
                    features.append(
                        i3d.extract_features(ip).squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy()
                    )
                np.save(os.path.join(save_dir, name[0]), np.concatenate(features, axis=0))
            else:
                # wrap them in Variable
                inputs = Variable(inputs.cuda(), volatile=True)
                features = i3d.extract_features(inputs)
                np.save(
                    os.path.join(save_dir, name[0]),
                    features.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy(),
                )


if __name__ == "__main__":
    # need to add argparse
    for i in tqdm(range(2), total=2):
        if i == 0:
            run()
        else:
            run(
                mode="flow",
                root="/data/ephemeral/home/pytorch-i3d/data/flows",
                split="/data/ephemeral/home/pytorch-i3d/data/sample.json",
                batch_size=1,
                load_model="/data/ephemeral/home/pytorch-i3d/models/flow_charades.pt",
                save_dir="/data/ephemeral/home/pytorch-i3d/sample_feat",
            )
