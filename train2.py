import os.path

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision.transforms import Resize, ToTensor, Compose, ColorJitter
from tqdm.autonotebook import tqdm
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
import shutil
import warnings

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser("train deeplab v3")
    parser.add_argument("--data_path", "-d", type=str, default="D:/vscode/python/VNAI/voc_dataset")
    parser.add_argument("--batchsize", "-b", type=int, default=4)
    parser.add_argument("--image_size", "-i", type=int, default=224)
    parser.add_argument("--lr", "-l", type=float, default=0.01)
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--log_path", "-p", type=str, default="Tensorboard")
    args = parser.parse_args()
    return args


class VOCDataset(VOCSegmentation):
    def __init__(self, root, year, image_set, download, size, transform, target_transform):
        super().__init__(root=root, year=year, image_set=image_set, download=download, transform=transform,
                         target_transform=target_transform)
        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                        'train', 'tvmonitor']
        self.size = size

    def __getitem__(self, item):
        image, target = super().__getitem__(item)
        target = np.array(target)
        target[target == 255] = 0
        return image, target


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True).to(device)

    # print(model)
    # model.backbone.requires_grad_(False)
    # model.classifier.requires_grad_(False)
    # model.aux_classifier.requires_grad_(True)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze last layer
    for param in model.aux_classifier.parameters():
        param.requires_grad = True


    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)
    writer = SummaryWriter(args.log_path)
    transform = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size)),
        ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05)
    ])
    target_transform = Compose([
        Resize((args.image_size, args.image_size))
    ])
    train_dataset = VOCDataset(root=args.data_path, year="2012", image_set="train", download=False,
                               size=args.image_size, transform=transform, target_transform=target_transform)
    test_dataset = VOCDataset(root=args.data_path, year="2012", image_set="val", download=False,
                              size=args.image_size, transform=transform, target_transform=target_transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        drop_last=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batchsize,
        shuffle=False,
        drop_last=False
    )
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        # TRAIN
        model.train()
        progress_bar = tqdm(train_dataloader, colour="yellow")
        for i, (images, targets) in enumerate(progress_bar):
            images = images.to(device)
            targets = targets.to(device)
            # Forward
            output = model(images)
            masks = output["out"]
            targets = torch.squeeze(targets).long()
            loss = criterion(masks, targets)
            progress_bar.set_description("Epoch {}/{}. Loss: {:0.4f}".format(epoch + 1, args.epochs, loss.item()))
            writer.add_scalar("Train/Loss", loss, epoch * len(train_dataloader) + i)
            # Backward
            optimizer.zero_grad()
            loss.requires_grad = True
            loss.backward()
            optimizer.step()

        # TEST
        model.eval()
        all_losses = []
        all_accs = []
        all_ious = []
        progress_bar = tqdm(test_dataloader, colour="green")
        acc_metric = MulticlassAccuracy(num_classes=21).to(device)
        iou_metric = MulticlassJaccardIndex(num_classes=21).to(device)
        with torch.no_grad():
            for images, targets in progress_bar:
                images = images.to(device)
                targets = targets.to(device)
                # Forward
                output = model(images)
                masks = output["out"]
                targets = torch.squeeze(targets, dim=1).long()
                loss = criterion(masks, targets)
                acc = acc_metric(masks, targets)
                iou = iou_metric(masks, targets)
                progress_bar.set_description(
                    "Epoch {}/{}. Loss: {:0.4f}. Accuracy: {:0.4f}. IoU: {:0.4f}".format(epoch + 1, args.epochs,
                                                                                         loss.item(), acc, iou))
                all_losses.append(loss.item())
                all_accs.append(acc.cpu().item())
                all_ious.append(iou.cpu().item())
        loss = np.mean(all_losses)
        acc = np.mean(all_accs)
        iou = np.mean(all_ious)
        writer.add_scalar("Test/Loss", loss, epoch)
        writer.add_scalar("Test/Accuracy", acc, epoch)
        writer.add_scalar("Test/IoU", iou, epoch)


if __name__ == '__main__':
    args = get_args()
    main(args)