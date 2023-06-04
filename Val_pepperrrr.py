import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

from collections import OrderedDict

from demo.dataset_pepper import Dataset
from metrics import iou_score, dice_coef, accuracy_score
from utils import AverageMeter
from models.baseline.attention_unet import AttU_Net

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='SwinUNet',
                        help='model name')
    parser.add_argument('--model_path', default='/home/felicia/project_pytorch/pytorch-nested-unet-master/',
                        help='model path')
    parser.add_argument('--model_name', default='yemeibing_Pepper_SwinUNet',
                        help='model name')
    parser.add_argument('--test_path', default='/home/felicia/datasets/pepper/yemeibing/Test/',
                        help='test path of the testing iamges')

    args = parser.parse_args()

    return args

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.43
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    output_ =  output > 0.5
    target = target.view(-1).data.cpu().numpy()
    intersection = (output_ * target).sum()

    return (2. * intersection + smooth) / \
        (output_.sum() + target.sum() + smooth)


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    scores = []
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
                dice = dice_coef(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)
                dice = dice_coef(output, target)
                target = target.cpu().numpy()
                output = np.round(output.cpu().numpy()).flatten()
                scores.append(accuracy_score(target.flatten(), output))

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])



def main():
    args = parse_args()
    path = args.model_path + 'output/' + args.model_name +'/config.yml'
    path_test = '/home/felicia/datasets/pepper/yemeibing/Test/'

    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    # model = archs.__dict__[config['arch']](config['num_classes'],
    #                                        config['input_channels'],
    #                                        config['deep_supervision'])

    # NestU-NET
    # model = archs.__dict__[config['arch']](config['num_classes'],
    #                                        config['input_channels'],
    #                                        config['deep_supervision'])
    # Se_PPP_ResUNet
    # model = archs.__dict__[config['arch']](config['input_channels'],
    #                                        config['num_classes'],
    #                                        config['deep_supervision'])
    # AttU_Net
    # model = AttU_Net(config['input_channels'], config['num_classes'])

    # U-NeXt
    # model = MLPGLUNet(config['num_classes'],config['input_channels'],
    #               config['deep_supervision'])

    # model = DoubleMLPUNet_DownAtt(config['input_channels'], config['num_classes'])
    # model = DoubleMLPUNet_DownAtt(config['input_channels'], config['num_classes'])
    # TMUNet
    # model = MLPGLUNet(config['num_classes'])

    model_name = 'SwinUNet'
    base_chan = 32
    num_class = 1
    reduce_size = 8
    block_list = '1234'
    load = False
    pretrained = False

    if model_name == 'UTNet':
        from transformer.utnet import UTNet
        model = UTNet(3, base_chan, num_class, reduce_size=reduce_size,
                    block_list=block_list)
    elif model_name == 'UTNet_encoder':
        from transformer.utnet import UTNet_Encoderonly
        # Apply transformer blocks only in the encoder
        model = UTNet_Encoderonly(3, base_chan, num_class, reduce_size=reduce_size,
                                block_list=block_list)

    elif model_name == 'TransUNet':
        from transformer.transunet import VisionTransformer as ViT_seg
        from transformer.transunet import CONFIGS as CONFIGS_ViT_seg

        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = 4
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(256 / 16), int(256 / 16))
        model = ViT_seg(config_vit, img_size=512, num_classes=1)
        # net.load_from(weights=np.load('./initmodel/R50+ViT-B_16.npz')) # uncomment this to use pretrain model download from TransUnet git repo

    elif model_name == 'ResNet_UTNet':
        from transformer.resnet_utnet import ResNet_UTNet
        model = ResNet_UTNet(3, num_class, reduce_size=reduce_size, block_list=block_list)
    elif model_name == 'AttU_Net':
        model = AttU_Net(config['input_channels'], config['num_classes'])

    elif model_name == 'SwinUNet':
        from transformer.swin_unet import SwinUnet, SwinUnet_config
        swinconfig = SwinUnet_config()
        model = SwinUnet(swinconfig, img_size=512, num_classes=num_class)
        model.load_from('models/swin_tiny_patch4_window7_224.pth')
    elif model_name == 'SCUNet':
        from models.network_scunet import BFUNet as net
        model = net(in_nc=3, config=[2, 2, 2, 2, 2, 2, 2], dim=32)
    elif model_name == 'BFUNet':
        from models.network_scunet import BFUNet as net
        model = net(in_nc=3, config=[2, 2, 2, 2, 2, 2, 2], dim=32)
    else:
        raise NotImplementedError(model_name + " has not been implemented")



    # model = LiverContextNetA(config['input_channels'], config['num_classes'])
    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join(args.test_path, 'image', '*.JPG' ))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load(args.model_path + 'output/%s/model.pth' %
                                     args.model_name))
    model.eval()

    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    # val_dataset = Dataset(
    #     img_ids=val_img_ids,
    #     img_dir=os.path.join('inputs', config['dataset'], 'images'),
    #     mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
    #     img_ext=config['img_ext'],
    #     mask_ext=config['mask_ext'],
    #     num_classes=config['num_classes'],
    #     transform=val_transform)
    val_dataset = Dataset(
        root=args.test_path, #os.path.join(,'image/'),
        img_ext='.JPG',
        mask_ext='.png',
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join(args.test_path, args.model_name, str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            iou = iou_score(output, target)
            avg_meter.update(iou, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    save_path = args.test_path+args.model_name +'/' + str(c) + '/'+ meta['img_id'][i].split('/')[-1][:-4] + '.jpg'
                    predict = (output[i, c] * 255).astype('uint8')
                    ret, binary = cv2.threshold(predict,0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    cv2.imwrite(save_path,binary )

    print('IoU: %.4f' % avg_meter.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()