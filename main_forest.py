import argparse
import os
from collections import OrderedDict
import numpy as np

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import archs
import losses
from demo.dataset_building import Dataset
from metrics import iou_score, dice_coef, accuracy_score
from utils import AverageMeter, str2bool
from models.ours.MLPGLUNet import MLPGLUNet
from transformer.utnet import UTNet, UTNet_Encoderonly

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


# LOSS_NAMES.append('BCEWLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='TransUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                             ' | '.join(ARCH_NAMES) +
                             ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=512, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=512, type=int,
                        help='image height')

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: BCEDiceLoss)')

    # dataset
    # parser.add_argument('--root', default='/home/felicia/datasets/Satellite-Sensor-Imagery/Amazon Forest Dataset/',
    #                     help='dataset name')
    # parser.add_argument('--dataset', default='RGB',
    #                     help='dataset name')
    parser.add_argument('--root', default='/home/felicia/datasets/city_building/Satellite2/',
                        help='dataset name')
    parser.add_argument('--dataset', default='zaoyibing',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.tif',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.tif',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')

    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
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
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()


    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


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
    config = vars(parse_args())

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('output/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('output/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        # criterion = nn.BCEWithLogitsLoss().cuda()
        criterion = nn.BCELoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # create model

    # MODEL = DeepResUNet(2).to(device)
    # MODEL = AttU_Net(1,2).to(device)
    # MODEL = SeResUNet(1, 2, deep_supervision=False).to(device)
    # MODEL = SE_P_AttU_Net(1,2).to(device)
    # MODEL = UNet(1, 2).to(device)
    # vgg_model = VGGNet()
    # MODEL = FCNs(pretrained_net=vgg_model, n_class=2).to(device)
    # MODEL = ResUnetPlusPlus(1,filters=[32, 64, 128, 256, 512]).to(device)
    # MODEL = Res_UNet(1,2).to(device)

    print("=> creating model %s" % config['arch'])
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

    model_name = 'MLPGLUNet'
    base_chan = 32
    num_class = 1
    reduce_size = 8
    block_list = '1234'
    load = False
    pretrained = False

    if model_name == 'UTNet':
        model = UTNet(3, base_chan, num_class, reduce_size=reduce_size,
                    block_list=block_list)
    elif model_name == 'UTNet_encoder':
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
        # from transformer.resnet_utnet import ResNet_UTNet
        # model = ResNet_UTNet(3, num_class, reduce_size=reduce_size, block_list=block_list)
    # elif model_name == 'AttU_Net':
        # model = AttU_Net(config['input_channels'],config['num_classes'])
        from transformer.BBU_Net import BBU_Net
        model = BBU_Net(3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   , num_class, reduce_size=reduce_size, block_list=block_list)
    elif model_name == 'SwinUNet':
        from transformer.swin_unet import SwinUnet, SwinUnet_config
        swinconfig = SwinUnet_config()
        model = SwinUnet(swinconfig, img_size=512, num_classes=num_class)
        model.load_from('models/swin_tiny_patch4_window7_224.pth')
    elif model_name == 'SCUNet':
        from models.network_scunet import MBFUNet as net
        model = net(in_nc=3, config=[2, 2, 2, 2, 2, 2, 2], dim=32)
    elif model_name == 'MLPGLUNet':
        model = MLPGLUNet(config['num_classes'],config['input_channels'],
                  config['deep_supervision'])
    else:
        raise NotImplementedError(model_name + " has not been implemented")

    if load:
        model.load_state_dict(torch.load(load))
        print('Model loaded from {}'.format(load))

    # model = TransMUNet(config['num_classes'])

    # Net = Net.to(device)
    model = model.cuda()
    if int(pretrained):
        model.load_state_dict(torch.load(config['saved_model'], map_location='cpu')['model_weights'])
        best_val_loss = torch.load(config['saved_model'], map_location='cpu')['val_loss']

    # model = model.cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    # img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    # img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    train_transform = Compose([
        transforms.RandomRotate90(),
        transforms.Flip(),
        OneOf([
            transforms.HueSaturationValue(),
            transforms.RandomBrightness(),
            transforms.RandomContrast(),
        ], p=1),
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        root=os.path.join(config['root'], 'Training/'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        root=os.path.join(config['root'], 'Validation/'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])

    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou'], val_log['dice']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv('output/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1
        # gc.collect()

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'output/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
