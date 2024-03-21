from torchvision.transforms import Resize,Compose,ToTensor,Normalize
import torch
import cv2
import argparse
from PIL import Image
import numpy as np
def get_args():
    parser = argparse.ArgumentParser('Predict deeplabv3 segmentation')
    parser.add_argument('--image_path',type=str,default='test_image.png')
    parser.add_argument('--save_path',type=str,default='result.png')
    parser.add_argument('--checkpoint_path',type=str,default=None)  

    args = parser.parse_args()
    return args
def main(args):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()
    image = cv2.imread(args.image_path)
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        ckp = torch.load(args.checkpoint_path)
        model = model.load_state_dict(ckp['model'])
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225])
    ])

    image = transform(image)[None,:,:,:].to(device)
    model= model.to(device)
    with torch.no_grad():
        output = model(image)['out'][0]
        pred = output.argmax(0)
    pil_image = Image.fromarray(np.uint8(pred.cpu().numpy())).convert('P')
    pil_image.show()
    pil_image.save(args.save_path)
if __name__ == "__main__":
    args = get_args()
    main(args)