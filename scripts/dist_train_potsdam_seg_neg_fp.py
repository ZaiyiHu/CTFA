import argparse
import datetime
import logging
import os
import random
import sys

sys.path.append(".")
sys.path.append(r'/home/zaiyihu/CodeSpace/CTFA-main')
sys.path.append(r'/home/zaiyihu/CodeSpace/CTFA-main/datasets')
sys.path.append(r'/home/zaiyihu/CodeSpace/CTFA-main/model')
sys.path.append(r'/home/zaiyihu/CodeSpace/CTFA-main/utils')
sys.path.append(r'/home/hzy/CTFA-main/CTFA-main')
sys.path.append(r'/home/hzy/CTFA-main/CTFA-main/datasets')
sys.path.append(r'/home/hzy/CTFA-main/CTFA-main/model')
sys.path.append(r'/home/hzy/CTFA-main/CTFA-main/utils')
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import potsdam
from model.losses import get_masked_ptc_loss, get_seg_loss, CTCLoss_neg, DenseEnergyLoss, get_energy_loss, JointLoss
from model.model_seg_neg_fp import network
from torch import autograd
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from model.PAR import PAR
from utils import evaluate, imutils, optimizer
# from utils.camutils import (cam_to_label, cams_to_affinity_label, ignore_img_box,
#                             multi_scale_cam, multi_scale_cam_with_aff_mat,
#                             propagte_aff_cam_with_bkg, refine_cams_with_bkg_v2,
#                             refine_cams_with_cls_label)
from utils.camutils_ori import cam_to_label, cam_to_roi_mask2, multi_scale_cam2, label_to_aff_mask, refine_cams_with_bkg_v2, crop_from_roi_neg
from utils.pyutils import AverageMeter, cal_eta, format_tabs, setup_logger
torch.hub.set_dir("./pretrained")
parser = argparse.ArgumentParser()

parser.add_argument("--backbone", default='deit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")
parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")

parser.add_argument("--data_folder", default='/data1/zaiyihu/Datasets/potsdam_IRRG_wiB_512_256_dl', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='/home/zaiyihu/CodeSpace/CTFA-main/datasets/potsdam', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=6, type=int, help="number of classes")
parser.add_argument("--crop_size", default=320, type=int, help="crop_size in training")
parser.add_argument("--local_crop_size", default=96, type=int, help="crop_size for local view")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")

parser.add_argument("--work_dir", default="work_dir_potsdam_wseg", type=str, help="work_dir_potsdam_wseg")

parser.add_argument("--train_set", default="train", type=str, help="training split")
parser.add_argument("--val_set", default="val", type=str, help="validation split")
parser.add_argument("--test_set", default="test", type=str, help="testing split")
parser.add_argument("--spg", default=8, type=int, help="samples_per_gpu")
parser.add_argument("--scales", default=(0.5, 2), help="random rescale in training")

parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")
parser.add_argument("--lr", default=4e-5, type=float, help="learning rate")
parser.add_argument("--warmup_lr", default=1e-6, type=float, help="warmup_lr")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
parser.add_argument("--betas", default=(0.9, 0.999), help="betas for Adam")
parser.add_argument("--power", default=0.9, type=float, help="poweer factor for poly scheduler")

parser.add_argument("--max_iters", default=20000, type=int, help="max training iters")
parser.add_argument("--log_iters", default=200, type=int, help=" logging iters")
parser.add_argument("--eval_iters", default=2000, type=int, help="validation iters")
parser.add_argument("--warmup_iters", default=1500, type=int, help="warmup_iters")

parser.add_argument("--high_thre", default=0.7, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.25, type=float, help="low_bkg_score")
parser.add_argument("--bkg_thre", default=0.6, type=float, help="bkg_score")
parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5), help="multi_scales for cam")

parser.add_argument("--w_ptc", default=0.3, type=float, help="w_ptc")
parser.add_argument("--w_ctc", default=0.5, type=float, help="w_ctc")
parser.add_argument("--w_seg", default=0.1, type=float, help="w_seg")
parser.add_argument("--w_reg", default=0.05, type=float, help="w_reg")
parser.add_argument("--w_joi", default=0.3, type=float, help="w_joi")
parser.add_argument("--temp", default=0.5, type=float, help="temp")
parser.add_argument("--momentum", default=0.9, type=float, help="temp")
parser.add_argument("--aux_layer", default=-3, type=int, help="aux_layer")

parser.add_argument("--seed", default=0, type=int, help="fix random seed")
parser.add_argument("--save_ckpt", default="1",action="store_true", help="save_ckpt")

parser.add_argument("--local_rank", default=0, type=int, help="local_rank")
parser.add_argument("--num_workers", default=10, type=int, help="num_workers")
parser.add_argument('--backend', default='nccl')
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


os.environ['MASTER_ADDR'] = '127.0.0.1'

os.environ['MASTER_PORT'] = '32301'

dist.init_process_group(backend='nccl', init_method='env://', rank = 0, world_size = 1)

def get_down_size(ori_shape=(512, 512), stride=16):
    h, w = ori_shape
    _h = h // stride + 1 - ((h % stride) == 0)
    _w = w // stride + 1 - ((w % stride) == 0)
    return _h, _w
def get_mask_by_radius(h=20, w=20, radius=8):
    hw = h * w
    # _hw = (h + max(dilations)) * (w + max(dilations))
    mask = np.zeros((hw, hw))
    for i in range(hw):
        _h = i // w
        _w = i % w

        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius + 1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius + 1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1
                mask[_i2, i] = 1

    return mask

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def validate(model=None, data_loader=None, args=None):

    preds, gts, cams, cams_aux = [], [], [], []
    model.eval()
    avg_meter = AverageMeter()
    count = 0
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            name, inputs, labels, cls_label = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            inputs  = F.interpolate(inputs, size=[args.crop_size, args.crop_size], mode='bilinear', align_corners=False)

            cls, segs, _, _ = model(inputs,)

            cls_pred = (cls>0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            _cams, _cams_aux = multi_scale_cam2(model, inputs, args.cam_scales)
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_to_label(resized_cam, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)

            resized_cam_aux = F.interpolate(_cams_aux, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label_aux = cam_to_label(resized_cam_aux, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)

            cls_pred = (cls > 0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
            cams_aux += list(cam_label_aux.cpu().numpy().astype(np.int16))
            #
            # valid_label = torch.nonzero(cls_label[0])[:, 0]
            # out_cam = torch.squeeze(resized_cam)[valid_label]
            # np.save(os.path.join(cfg.work_dir.pred_dir, name[0]+'.npy'), {"keys":valid_label.cpu().numpy(), "cam":out_cam.cpu().numpy()})

    cls_score = avg_meter.pop('cls_score')
    seg_score = evaluate.scores(gts, preds,6)
    cam_score = evaluate.scores(gts, cams,6)
    cam_aux_score = evaluate.scores(gts, cams_aux,6)
    model.train()

    tab_results, mIoU = format_tabs([cam_score, cam_aux_score, seg_score], name_list=["CAM", "aux_CAM", "Seg_Pred"],
                                    cat_list=potsdam.class_list)

    return cls_score, tab_results, mIoU

def train(args=None):

    torch.cuda.set_device(args.local_rank)
    # dist.init_process_group(backend=args.backend, )
    logging.info("Total gpus: %d, samples per gpu: %d..."%(dist.get_world_size(), args.spg))
    mIoU = 0.0001
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    train_dataset = potsdam.potsdamClsDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.train_set,
        stage='train',
        aug=True,
        # resize_range=cfg.dataset.resize_range,
        rescale_range=args.scales,
        crop_size=args.crop_size,
        img_fliplr=True,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    val_dataset = potsdam.potsdamSegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.test_set,
        stage='test',
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.spg,
        #shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        sampler=train_sampler,
        prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=False,
                            drop_last=False)

    device = torch.device(args.local_rank)

    model = network(
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        init_momentum=args.momentum,
        aux_layer=args.aux_layer
    )
    param_groups = model.get_param_groups()
    model.to(device)

    # cfg.optimizer.learning_rate *= 2
    optim = getattr(optimizer, args.optimizer)(
        params=[
            {
                "params": param_groups[0],
                "lr": args.lr,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[1],
                "lr": args.lr,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[2],
                "lr": args.lr * 8,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[3],
                "lr": args.lr * 8,
                "weight_decay": args.wt_decay,
            },
        ],
        lr=args.lr,
        weight_decay=args.wt_decay,
        betas=args.betas,
        warmup_iter=args.warmup_iters,
        max_iter=args.max_iters,
        warmup_ratio=args.warmup_lr,
        power=args.power)

    logging.info('\nOptimizer: \n%s' % optim)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    train_sampler.set_epoch(np.random.randint(args.max_iters))
    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()


    loss_layer = DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)
    ncrops = 10
    CTC_loss = CTCLoss_neg(ncrops=ncrops, temp=args.temp).cuda()
    JOINT_loss = JointLoss().cuda()
    par = PAR(num_iter=10, dilations=[1,2,4,8,12,24]).cuda()

    for n_iter in range(args.max_iters):

        try:
            img_name, inputs, cls_label, img_box, crops = next(train_loader_iter)

        except:
            train_sampler.set_epoch(np.random.randint(args.max_iters))
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_label, img_box, crops = next(train_loader_iter)

        inputs = inputs.to(device, non_blocking=True)
        b, c, h, w = inputs.shape
        inputs_denorm = imutils.denormalize_img2(inputs.clone())
        cls_label = cls_label.to(device, non_blocking=True)

        # get local crops from uncertain regions
        cams, cams_aux = multi_scale_cam2(model, inputs=inputs, scales=args.cam_scales)
        roi_mask = cam_to_roi_mask2(cams_aux.detach(), cls_label=cls_label, low_thre=args.low_thre, hig_thre=args.high_thre)

        local_crops, flags = crop_from_roi_neg(images=crops[2], roi_mask=roi_mask, crop_num=ncrops-2, crop_size=args.local_crop_size)
        roi_crops = crops[:2] + local_crops
        # with autograd.detect_anomaly():
        cls, segs, foreground_seg, fmap, cls_aux, out_t, out_s = model(inputs, crops=roi_crops, n_iter=n_iter)
        cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
        cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)

        # ctc_loss
        ctc_loss = CTC_loss(out_s, out_t, flags)

        # seg_loss & reg_loss
        valid_cam, _ = cam_to_label(cams.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_cam, cls_labels=cls_label,  high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box, )
        segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        foreground_seg = F.interpolate(foreground_seg, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        seg_loss = get_seg_loss(segs, refined_pseudo_label.type(torch.long), ignore_index=args.ignore_index)

        reg_loss = get_energy_loss(img=inputs, logit=segs, label=refined_pseudo_label, img_box=img_box, loss_layer=loss_layer)

        # ptc loss
        resized_cams_aux = F.interpolate(cams_aux, size=fmap.shape[2:], mode="bilinear", align_corners=False)
        _, pseudo_label_aux = cam_to_label(resized_cams_aux.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        aff_mask = label_to_aff_mask(pseudo_label_aux)
        ptc_loss = get_masked_ptc_loss(fmap, aff_mask)
        # ptc_loss = get_ptc_loss(fmap, low_fmap)

        # joint loss
        joint_loss = JOINT_loss(foreground_seg,segs,refined_pseudo_label.type(torch.long))

        # warmup
        if n_iter <= 2000:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + args.w_ctc * ctc_loss  + 0.0 * seg_loss + 0.0 * reg_loss + 0.0 * joint_loss
        else:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + args.w_ctc * ctc_loss  + args.w_seg * seg_loss + args.w_reg * reg_loss + args.w_joi * joint_loss

        cls_pred = (cls > 0).type(torch.int16)
        cls_score = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])

        avg_meter.add({
            'cls_loss': cls_loss.item(),
            'ptc_loss': ptc_loss.item(),
            'ctc_loss': ctc_loss.item(),
            'cls_loss_aux': cls_loss_aux.item(),
            'seg_loss': seg_loss.item(),
            'joint_loss': joint_loss.item(),
            'cls_score': cls_score.item(),

        })

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if (n_iter + 1) % args.log_iters == 0:

            delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
            cur_lr = optim.param_groups[0]['lr']

            if args.local_rank == 0:
                logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, cls_loss_aux: %.4f, ptc_loss: %.4f, ctc_loss: %.4f, seg_loss: %.4f..., joint_loss: %.4f..." % (n_iter + 1, delta, eta, cur_lr, avg_meter.pop('cls_loss'), avg_meter.pop('cls_loss_aux'), avg_meter.pop('ptc_loss'), avg_meter.pop('ctc_loss'), avg_meter.pop('seg_loss'),avg_meter.pop('joint_loss')))

        if (n_iter + 1) % args.eval_iters == 0:

            if args.local_rank == 0:
                logging.info('Validating...')
            val_cls_score, tab_results, mIoU_result = validate(model=model, data_loader=val_loader, args=args)
            if args.save_ckpt and (n_iter + 1) >= 6000 and mIoU_result[2] > mIoU:
                mIoU = mIoU_result[2]
                ckpt_name = os.path.join(args.ckpt_dir,
                                         "Best mIoU: {}, model: {} model_iter_%d.pth".format(mIoU, "TCFA_potsdam") % (
                                                 n_iter + 1))
                torch.save(model.state_dict(), ckpt_name)
            if args.local_rank == 0:
                logging.info("val cls score: %.6f" % (val_cls_score))
                logging.info("\n" + tab_results)

    return True


if __name__ == "__main__":

    args = parser.parse_args()
    timestamp = "{0:%Y-%m-%d-%H-%M-%S-%f}".format(datetime.datetime.now())
    args.work_dir = os.path.join(args.work_dir, timestamp)
    args.ckpt_dir = os.path.join(args.work_dir, "checkpoints")
    args.pred_dir = os.path.join(args.work_dir, "predictions")

    if args.local_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.pred_dir, exist_ok=True)

        setup_logger(filename=os.path.join(args.work_dir, 'train.log'))
        logging.info('Pytorch version: %s' % torch.__version__)
        logging.info("GPU type: %s"%(torch.cuda.get_device_name(0)))
        logging.info('\nargs: %s' % args)

    ## fix random seed
    setup_seed(args.seed)
    train(args=args)
