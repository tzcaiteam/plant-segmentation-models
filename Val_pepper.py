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
from scipy import ndimage

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='TransMUNet',
                        help='model name')
    parser.add_argument('--model_path', default='H:\\pytorch\\pytorch-nested-unet-master\\output\\pepper',
                        help='model path')
    parser.add_argument('--model_name', default='AMS_MLP_mask_all_Healthy',
                        help='model name')
    parser.add_argument('--test_path', default='H:\\dataset\\pepper\\jiankang\\Test\\',
                        help='test path of the testing iamges')

    args = parser.parse_args()

    return args

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
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
                output = model(
                    input)
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


# Hausdorff and ASSD evaluation
def get_edge_points(img):
    """
    get edge points of a binary segmentation result
    """
    dim = len(img.shape)
    if (dim==2):
        strt = ndimage.generate_binary_structure(2, 1)
    else:
        strt = ndimage.generate_binary_structure(3, 1)  # 三维结构元素，与中心点相距1个像素点的都是邻域
    ero = ndimage.morphology.binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
    return edge


def binary_hausdorff95(s, g, spacing=None):
    """
    get the hausdorff distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert (image_dim==len(g.shape))
    if (spacing==None):
        spacing = [1.0] * image_dim
    else:
        assert (image_dim==len(spacing))
    img = np.zeros_like(s)
    if (image_dim==2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif (image_dim==3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    dist_list1 = s_dis[g_edge > 0]
    dist_list1 = sorted(dist_list1)
    dist1 = dist_list1[int(len(dist_list1) * 0.95)]
    dist_list2 = g_dis[s_edge > 0]
    dist_list2 = sorted(dist_list2)
    dist2 = dist_list2[int(len(dist_list2) * 0.95)]
    return max(dist1, dist2)


# 平均表面距离
def binary_assd(s, g, spacing=None):
    """
    get the average symetric surface distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert (image_dim==len(g.shape))
    if (spacing==None):
        spacing = [1.0] * image_dim
    else:
        assert (image_dim==len(spacing))
    img = np.zeros_like(s)
    if (image_dim==2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif (image_dim==3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    ns = s_edge.sum()
    ng = g_edge.sum()
    s_dis_g_edge = s_dis * g_edge
    g_dis_s_edge = g_dis * s_edge
    assd = (s_dis_g_edge.sum() + g_dis_s_edge.sum()) / (ns + ng)
    return assd


# relative volume error evaluation
def binary_relative_volume_error(s_volume, g_volume):
    s_v = float(s_volume.sum())
    g_v = float(g_volume.sum())
    assert (g_v > 0)
    rve = abs(s_v - g_v) / g_v
    return rve


def get_evaluation_score(s_volume, g_volume, spacing, metric):
    if (len(s_volume.shape)==4):
        assert (s_volume.shape[0]==1 and g_volume.shape[0]==1)
        s_volume = np.reshape(s_volume, s_volume.shape[1:])
        g_volume = np.reshape(g_volume, g_volume.shape[1:])
    if (s_volume.shape[0]==1):
        s_volume = np.reshape(s_volume, s_volume.shape[1:])
        g_volume = np.reshape(g_volume, g_volume.shape[1:])
    metric_lower = metric.lower()

    if (metric_lower=="hausdorff95"):
        score = binary_hausdorff95(s_volume, g_volume, spacing)

    elif (metric_lower=="rve"):
        score = binary_relative_volume_error(s_volume, g_volume)

    elif (metric_lower=="volume"):
        voxel_size = 1.0
        for dim in range(len(spacing)):
            voxel_size = voxel_size * spacing[dim]
        score = g_volume.sum() * voxel_size
    else:
        raise ValueError("unsupported evaluation metric: {0:}".format(metric))

    return score


def main():
    args = parse_args()
    # path = args.model_path + 'output/' + args.model_name +'/config.yml'
    path = args.model_path + '/' + "Pepper_"+ args.model_name + '/config.yml'
    path_test = 'H:\\dataset\\pepper\\jiankang\\Test\\'

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

    model_name = 'BFUNet'
    base_chan = 32
    num_class = 1
    reduce_size = 8
    block_list = '1234'
    load = False
    pretrained = False

    if model_name == 'UNet':
        from models.baseline.U_Net import U_Net as net
        model = net(img_ch=3, output_ch=1)
    elif model_name == 'UTNet':
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

    elif model_name == 'SwinUNet':
        from transformer.swin_unet import SwinUnet, SwinUnet_config
        swinconfig = SwinUnet_config()
        model = SwinUnet(swinconfig, img_size=512, num_classes=num_class)
        model.load_from('models/swin_tiny_patch4_window7_224.pth')
    elif model_name == 'SCUNet':
        # from models.network_scunet import BFUNet as net
        # model = net(in_nc=3, config=[2, 2, 2, 2, 2, 2, 2], dim=32)
        from models.ours.BAF_Net import BAF_Net as net
        model = net(in_nc=3, config=[2, 2, 2, 2, 2, 2, 2], dim=16)
    elif model_name == 'BFUNet':
        from models.ours.AdaptivemultiscaleMLP import UNet_MLP_mask_all as net
        # model = net(in_nc=3, config=[2, 2, 2, 2, 2, 2, 2], dim=32)  #MLP模型
        # model = net(n_channels = 3,n_classes = 1, basic_chans= 16) #原始U-Net模型
        model = net(n_channels=3, n_classes=1, basic_chans=8)   #通道数减半的U-Net模型

    else:
        raise NotImplementedError(model_name + " has not been implemented")



    # model = LiverContextNetA(config['input_channels'], config['num_classes'])
    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join(args.test_path, 'image', '*.JPG' ))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load(args.model_path + '/%s/model.pth' %
                                     ("Pepper_"+args.model_name)))
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
        TP = 0
        TN = 0
        FN = 0
        FP = 0
        hausdorff = 0
        i = 0
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            avg_meters = {'loss': AverageMeter(),
                          'iou': AverageMeter(),
                          'dice': AverageMeter()}

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            iou = iou_score(output, target)
            avg_meter.update(iou, input.size(0))

            dice = dice_coef(output, target)
            avg_meter.update(dice, input.size(0))

            pred_choice = output.cpu().numpy().flatten()>0.43
            gt_target = target.cpu().numpy().flatten()>0.5

            # TP predict 和 label 同时为1
            TP += ((pred_choice == 1) & (gt_target == 1)).sum()
            # TN predict 和 label 同时为0
            TN += ((pred_choice == 0) & (gt_target == 0)).sum()
            # FN predict 0 label 1
            FN += ((pred_choice == 0) & (gt_target == 1)).sum()
            # FP predict 1 label 0
            FP += ((pred_choice == 1) & (gt_target == 0)).sum()

            # hausdorff += get_evaluation_score(output[i][0].cpu().numpy().astype('float32'), target[i][0].cpu().numpy().astype('float32'), spacing=None, metric='hausdorff95')
            # hausdorff_distance(output[i][0].cpu().numpy(), target[i][0].cpu().numpy(), distance="euclidean")

            output = torch.sigmoid(output).cpu().numpy()
            i = i+1

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    save_path = args.test_path+args.model_name +'/' + str(c) + '/'+ meta['img_id'][i].split('/')[-1][:-4] + '.jpg'
                    predict = (output[i, c] * 255).astype('uint8')
                    ret, binary = cv2.threshold(predict,0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    cv2.imwrite(save_path,binary )
        if TP + FP == 0:
            p = 0
            r = 0
        else:
            p = TP / (TP + FP)
            r = TP / (TP + FN)

        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        recall =TP/(TP+FN)
        spe =TN/(TN+FP)
        pre =TP/(TP+FP)
        # mean_hausdorff = hausdorff / i

    # print("Hausdorff distance test: {0}".format(hausdorff_distance(X, Y, distance="manhattan")))
    # print("Hausdorff distance test: {0}".format(hausdorff_distance(X, Y, distance="euclidean")))
    # print("Hausdorff distance test: {0}".format(hausdorff_distance(X, Y, distance="chebyshev")))
    # print("Hausdorff distance test: {0}".format(hausdorff_distance(X, Y, distance="cosine")))
    mean_hausdorff = 0

    print('IoU: %.4f, F1: %.4f, accuracy: %.4f, recall: %.4f, SPE: %.4f, pre: %.4f, mean_hausdorff:  %.4f' % (avg_meter.avg, F1, acc, recall, spe, pre, mean_hausdorff))


    torch.cuda.empty_cache()

    # for c in range(config['num_classes']):
    #     os.makedirs(os.path.join(args.test_path, args.model_name, str(c)), exist_ok=True)
    # with torch.no_grad():
    #     for input, target, meta in tqdm(val_loader, total=len(val_loader)):
    #         input = input.cuda()
    #         target = target.cuda()
    #
    #         # compute output
    #         if config['deep_supervision']:
    #             output = model(input)[-1]
    #         else:
    #             output = model(input)
    #
    #         iou = iou_score(output, target)
    #         avg_meter.update(iou, input.size(0))
    #
    #         output = torch.sigmoid(output).cpu().numpy()
    #
    #         for i in range(len(output)):
    #             for c in range(config['num_classes']):
    #                 save_path = args.test_path+args.model_name +'/' + str(c) + '/'+ meta['img_id'][i].split('/')[-1][:-4] + '.jpg'
    #                 predict = (output[i, c] * 255).astype('uint8')
    #                 ret, binary = cv2.threshold(predict,0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #                 cv2.imwrite(save_path,binary )
    #
    # print('IoU: %.4f' % avg_meter.avg)
    #
    # torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
