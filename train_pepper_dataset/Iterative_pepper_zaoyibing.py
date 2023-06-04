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
import sys
import albumentations as albu

sys.path.append('..')

from utils import losses
from utils.PepperDataSet import PepperDataset
from utils.metrics import iou_score, dice_coef, accuracy_score
from utils.utils import AverageMeter, str2bool

LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

########################################################################################################################
# 版权所有：台州学院人工智能小组（2023.04.06）
# 代码功能：training过程的leaf下的早疫病数据集
# 代码使用：只需要修改函数parse_args下的参数，见其上方的代码说明
# 代码修改人：Jiangxiong Fang
########################################################################################################################


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

    # if config['deep_supervision']==False:
    #    model.eval()
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

########################################################################################################################
#版权所有：台州学院人工智能小组（2023.04.06）
# model_path：数据集路径，比如：I:\\pytorch\\pytorch-nested-unet-master\\，注意后面的斜杠不能删掉
# model_name：模型名称，应该与文件的模型名称完全一致
# model_type：模型类型，对应于四个类型：Baseline，MLP，Transformer,Ours
# 其中：  Baseline包括： UNet, Attention UNet(), NestedUNet,Se_PPP_ResUNet
#        MLP：
#        Transformer：
#        Ours:
# epochs:         训练次数
# batch_size：    批次，默认60次
# input_channels：输入的通道数，默认3通道RGB
# num_classes：   分割类别数量（目标的数量），默认是1个目标
# model_type:     模型类型，包括baseline，MLP，transformer，few_shot等，对应于models文件下面的类型
# filesname：     文件名称
#
# 在训练过程中，初次训练需要修改的参数：model_path，model_name，model_type，input_channels，num_classes
# 切换模型需要修改的参数：model_name，model_type
#
########################################################################################################################

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, default='/home/felicia/datasets/pepper/zaoyibing/',
                        help='dataset name')
    parser.add_argument('--model_type', default='Iteration',type=str,
                        help='dataset name')
    parser.add_argument('--filesname', default='pepper_zaoyibing',type=str,
                        help='dataset name')

    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=2, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    # model
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='UNet_AdapMLP_mask_CRE_all',
    #                     choices=ARCH_NAMES,
    #                     help='model architecture: ' +
    #                          ' | '.join(ARCH_NAMES) +
    #                          ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=True, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=512, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=512, type=int,
                        help='image height')

    ####################################################################################################################
    # dataset_type :有val_and_train和only_train
    # val_and_train：验证集和训练集分开
    # only_train   ：只包括训练集
    ####################################################################################################################
    parser.add_argument('--dataset_type', default="val_and_train", type=str,
                        help='image height')
    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: BCEDiceLoss)')
    parser.add_argument('--img_ext', default='.jpg',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
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

    parser.add_argument('--load', default=False, type=str2bool,
                        help='The existing model need loading')
    parser.add_argument('--pretrained', default=False, type=str2bool,
                        help='The existing model need loading')

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

########################################################################################################################
#版权所有：台州学院人工智能小组（2023.04.06）
# 模型输出路径：model_output/pepper_zaoyibing/模型名/
# model_name：模型名称，应该与文件的模型名称完全一致
#
#
#
#
########################################################################################################################

def main():

    config = vars(parse_args())

    # model_name = config['model_name']
    num_class = config['num_classes']
    deep_supervision = config['deep_supervision']


    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        # criterion = nn.BCEWithLogitsLoss().cuda()
        criterion = nn.BCELoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # train_transform = Compose([
    #     transforms.RandomRotate90(),
    #     transforms.Flip(),
    #     OneOf([
    #         transforms.HueSaturationValue(),
    #         transforms.RandomBrightness(),
    #         transforms.RandomContrast(),
    #     ], p=1),
    #     transforms.Resize(config['input_h'], config['input_w']),
    #     transforms.Normalize(),
    # ])

    train_transform = Compose([
        albu.RandomRotate90(),
        albu.Flip(),
        OneOf([
            transforms.HueSaturationValue(),
            transforms.RandomBrightness(),
            transforms.RandomContrast(),
        ], p=1),
        albu.Resize(config['input_h'], config['input_w']),
        albu.Normalize(),
    ])

    # val_transform = Compose([
    #     transforms.Resize(config['input_h'], config['input_w']),
    #     transforms.Normalize(),
    # ])

    val_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        albu.Normalize(),
    ])

    if config['dataset_type'] == 'val_and_train':
        train_dataset = PepperDataset(
            root=os.path.join(config['root'], 'Training/'),
            img_ext=config['img_ext'],
            mask_ext=config['mask_ext'],
            num_classes=config['num_classes'],
            transform=train_transform)
        val_dataset = PepperDataset(
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

    elif config['dataset_type'] == 'only_train':
        from utils.BasicDataset import BasicDataset
        from torch.utils.data import DataLoader, random_split
        img_scale = 1.0
        val_percent = 0.1
        dir_img = os.path.join(config['root'], 'train/')
        dir_mask = os.path.join(config['root'], 'mask/')
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

        # 2. Split into train / validation partitions
        n_val = int(len(dataset) * val_percent)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

        # 3. Create data loaders
        loader_args = dict(batch_size=config['batch_size'], num_workers=os.cpu_count(), pin_memory=True)
        train_loader = DataLoader(train_set, shuffle=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)


    #####################################################################
    # 1. U-Net
    # 2. Se_PPP_ResUNet
    # 3.
    # 4.
    # 5.
    # 6.

    model_stores = ['UNet','AttU_Net','R2U_Net','NestedUNet','Se_PPP_ResUNet','UNeXt']
    for each_model in model_stores:

        config['model_name'] = each_model

        if config['model_name'] is None:
            if config['deep_supervision']:
                config['model_name'] = '%s_%s' % (config['model_type'], each_model)
            else:
                config['model_name'] = '%s_%s' % (config['model_type'], each_model)

        ##### build output path
        model_output_path = '../model_output/%s/%s/%s/' % (config['filesname'], config['model_type'], config['model_name'])
        os.makedirs(model_output_path, exist_ok=True)

        print('-' * 20)
        for key in config:
            print('%s: %s' % (key, config[key]))
        print('-' * 20)

        with open('%s/config.yml' % model_output_path, 'w') as f:
            yaml.dump(config, f)

        if each_model == 'UNet':
            from models.baseline.BasicUNet import UNet as net
            model = net(n_channels=3, n_classes=1, deep_supervision=deep_supervision)
        elif each_model == 'AttU_Net':
            from models.baseline.BasicUNet import AttU_Net as net
            # model = AttU_Net(config['input_channels'], config['num_classes'])
            model = net(n_channels=3, n_classes=1, deep_supervision=deep_supervision)
        elif each_model == 'R2U_Net':
            from models.baseline.BasicUNet import R2U_Net as net
            model = net(n_channels=3, n_classes=1, deep_supervision=deep_supervision)
        elif each_model == 'NestedUNet':
            from models.baseline.BasicUNet import NestedUNet as net
            model = net(n_channels=3, n_classes=1, deep_supervision=deep_supervision)
        elif each_model == 'Se_PPP_ResUNet':
            from models.baseline.Se_PPP_ResUNet import Se_PPP_ResUNet as net
            model = net(n_channels=3, n_classes=1, deep_supervision=deep_supervision)

        elif each_model == 'UNeXt':
            from models.MLP.UNeXt import UNext as net
            model = net(n_channels=3, n_classes=1, deep_supervision=deep_supervision)
        elif each_model == 'UTNet':
            from models.transformer.utnet import UTNet
            base_chan = 32
            reduce_size = 8
            block_list = '1234'
            load = False
            pretrained = False
            model = UTNet(3, base_chan, num_class, reduce_size=reduce_size,
                          block_list=block_list)

        elif each_model == 'UTNet_encoder':
            # Apply transformer blocks only in the encoder
            from models.transformer.utnet import UTNet_Encoderonly
            base_chan = 32
            reduce_size = 8
            block_list = '1234'
            load = False
            pretrained = False
            model = UTNet_Encoderonly(3, base_chan, num_class, reduce_size=reduce_size,
                                      block_list=block_list)

        elif each_model == 'TransUNet':
            from models.transformer.transunet import VisionTransformer as ViT_seg
            from models.transformer.transunet import CONFIGS as CONFIGS_ViT_seg

            base_chan = 32
            reduce_size = 8
            block_list = '1234'
            load = False
            pretrained = False

            config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
            config_vit.n_classes = 4
            config_vit.n_skip = 3
            config_vit.patches.grid = (int(256 / 16), int(256 / 16))
            model = ViT_seg(config_vit, img_size=512, num_classes=1)
            # net.load_from(weights=np.load('../initmodel/R50+ViT-B_16.npz')) # uncomment this to use pretrain model download from TransUnet git repo

        elif each_model == 'ResNet_UTNet':
            from models.transformer.resnet_utnet import ResNet_UTNet
            reduce_size = 8
            block_list = '1234'
            load = False

            model = ResNet_UTNet(3, num_class, reduce_size=reduce_size, block_list=block_list)

        elif each_model == 'SwinUNet':
            from models.transformer.swin_unet import SwinUnet, SwinUnet_config

            swinconfig = SwinUnet_config()
            model = SwinUnet(swinconfig, img_size=512, num_classes=num_class)
            model.load_from('models/swin_tiny_patch4_window7_224.pth')
        elif each_model == 'SCUNet':
            from models.transformer.network_scunet import BFUNet as net
            model = net(in_nc=3, config=[2, 2, 2, 2, 2, 2, 2], dim=32)
        elif each_model == 'SCUNet':
            from models.Ours_MLP.MLPGLUNet import MLPGLUNet
            model = MLPGLUNet(config['num_classes'], config['input_channels'],
                              config['deep_supervision'])

            from models.Ours_MLP.DoubleMLPUNet import DoubleMLPUNet_DownAtt
            model = DoubleMLPUNet_DownAtt(config['input_channels'], config['num_classes'])

            from models.Ours_MLP.MLPGLUNet import MLPGLUNet
            model = MLPGLUNet(config['num_classes'])

        if config['load']:
            model.load_state_dict(torch.load(load))
            print('Model loaded from {}'.format(load))

        # model = TransMUNet(config['num_classes'])

        # create model
        print("=> creating model %s" % config['model_name'])

        # Net = Net.to(device)
        model = model.cuda()
        if int(config['pretrained']):
            model.load_state_dict(torch.load(config['saved_model'], map_location='cpu')['model_weights'])
            best_val_loss = torch.load(config['saved_model'], map_location='cpu')['val_loss']

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
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=[int(e) for e in config['milestones'].split(',')],
                                                 gamma=config['gamma'])
        elif config['scheduler'] == 'ConstantLR':
            scheduler = None
        else:
            raise NotImplementedError

        log = OrderedDict([
            ('epoch', []),
            ('lr', []),
            ('loss', []),
            ('iou', []),
            ('val_loss', []),
            ('val_iou', []),
            ('val_dice', []),
        ])

        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))

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

            pd.DataFrame(log).to_csv('%s/log.csv' % model_output_path, index=False)

            trigger += 1
            # gc.collect()

            if val_log['iou'] > best_iou:
                torch.save(model.state_dict(), '%s/best_model.pth' %
                           model_output_path)
                best_iou = val_log['iou']
                print("=> saved best model")
                trigger = 0

            # early stopping
            if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
                print("=> early stopping")
                break

            torch.cuda.empty_cache()
        torch.save(model.state_dict(), '%s/final_model.pth' %
                   model_output_path)


if __name__ == '__main__':
    main()
