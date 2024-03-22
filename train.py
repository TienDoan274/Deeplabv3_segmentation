import torch
import urllib
import argparse
from torchvision.datasets import VOCSegmentation
import cv2
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor,Resize,Compose,PILToTensor,Normalize
import numpy as np
from torch.optim import SGD
import torch.nn as nn
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
from torchmetrics.classification import MulticlassAccuracy,MulticlassJaccardIndex
def get_args():
    parser = argparse.ArgumentParser('train deeplab v3')
    parser.add_argument('--data_path','-d',type=str,default='D:/vscode/python/VNAI/voc_dataset')
    parser.add_argument('--batch_size',type=int,default=8)
    parser.add_argument('--learning_rate','-lr',type=float,default=1e-2)
    parser.add_argument('--epochs','-e',type=int,default=50)
    parser.add_argument('--image_size','-i',type=int,default=224)
    parser.add_argument('--tensorboard_path',type=str,default='Tensorboard')
    parser.add_argument('--save_path',type=str,default='saved_models')
    parser.add_argument('--checkpoint_path',type=str,default=None)


    args = parser.parse_args()
    return args
class VOCdataset(VOCSegmentation):
    def __init__(self, root, year, image_set, download, size,transform,target_transform):
        super().__init__(root = root, year = year, image_set = image_set, download = download,transform=transform,target_transform=target_transform)
        self.classes = ["background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
        self.size = size
    def __len__(self) -> int:
        return super().__len__()
    
    def __getitem__(self, index: int):
        image,target = super().__getitem__(index)
 
        target = np.array(target)

        target[target==255] = 0
        return image,target
def main(args):
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    transform = Compose([
        ToTensor(),
        Resize((args.image_size,args.image_size)),
        # Normalize(mean=[0.485,0.456,0.406],
        #           std=[0.229,0.224,0.225])
    ])
    target_transform = Compose([
        Resize((args.image_size,args.image_size)),
        # Normalize(mean=[0.485,0.456,0.406],
        #     std=[0.229,0.224,0.225])
    ])
    if os.path.isdir(args.tensorboard_path):
        shutil.rmtree(args.tensorboard_path)
    os.makedirs(args.tensorboard_path)
    writer = SummaryWriter(args.tensorboard_path)

    train_dataset = VOCdataset(root='D:/vscode/python/VNAI/voc_dataset',year='2012',image_set='train',download=False,size=224,transform=transform,target_transform=target_transform)
    val_dataset = VOCdataset(root='D:/vscode/python/VNAI/voc_dataset',year='2012',image_set='val',download=False,size=224,transform=transform,target_transform=target_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        drop_last=False
    )
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True).to(device)
    
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze last layer
    for param in model.aux_classifier.parameters():
        param.requires_grad = True
    optimizer = SGD(model.parameters(),lr=args.learning_rate) 
    criterion = nn.CrossEntropyLoss()
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        ckp = torch.load(args.checkpoint_path)
        model = model.load_state_dict(ckp['model'])
        optimizer = optimizer.load_state_dict(ckp['optimizer'])
        start_epoch = ckp['last_epoch']
    else:
        start_epoch = 0
    
    best_acc = 0
    for epoch in range(start_epoch,args.epochs+start_epoch):
        model.train()
        progress_bar = tqdm(train_loader)
        for i,(images,targets) in enumerate(progress_bar):
            
            images = images.to(device)
            output = model(images)
            masks = output['out']
            targets=torch.squeeze(targets).long().to(device)
            
            loss = criterion(masks,targets)
            progress_bar.set_description('Epoch {}/{}.Loss {:.4f}'.format(epoch + 1,args.epochs+start_epoch,loss.item()))
            writer.add_scalar('Train/loss',loss,epoch*len(train_loader)+i)
            optimizer.zero_grad()
            loss.requires_grad = True 
            loss.backward()
            optimizer.step()
        model.eval()
        metric = MulticlassAccuracy(num_classes=21).to(device)
        iou_metric = MulticlassJaccardIndex(num_classes=21).to(device)
        all_losses = []
        all_acc = []
        all_iou = []
        with torch.no_grad():
            val_bar = tqdm(val_loader,colour='yellow')
            for i,(images,targets) in enumerate(val_bar):
                images = images.to(device)
                val_output = model(images)
                val_masks = val_output['out']
                targets = torch.squeeze(targets,dim=1).long().to(device)
                val_loss = criterion(val_masks,targets)
                acc = metric(val_masks,targets)
                iou = iou_metric(val_masks,targets)
                val_bar.set_description('Epoch {}/{}.Val Loss: {:.2f}. Accuracy: {:.2f}. IOU: {:.2f}'.format(epoch + 1,args.epochs+start_epoch,val_loss,acc,iou))
                all_acc.append(acc.item())
                all_iou.append(iou.item())
                all_losses.append(val_loss.item())
            mean_loss = np.mean(all_losses)
            mean_acc = np.mean(all_acc)
            mean_iou = np.mean(all_iou)
            print('Mean loss:',mean_loss)
            print('Mean acc:',mean_acc)
            print('Mean iou:',mean_iou)

            writer.add_scalar('Test/Mean loss',mean_loss,epoch)
            writer.add_scalar('Test/Mean acc',mean_loss,epoch)
            writer.add_scalar('Test/Mean iou',mean_loss,epoch)
            checkpoint = {
                'last_epoch' : epoch+1,
                'model' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'acc' : mean_acc
            }
            torch.save(checkpoint,os.path.join(args.save_path,'last.pt'))
            if best_acc<mean_acc:
                torch.save(checkpoint,os.path.join(args.save_path,'best.pt'))

if __name__ =='__main__':

    args = get_args()
    main(args)