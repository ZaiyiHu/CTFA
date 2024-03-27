import argparse
import datetime
import logging
import os
import random
import sys

sys.path.append("../scripts")
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
from datasets import voc as voc
from datasets import iSAID
from thop import profile
from model.losses import get_masked_ptc_loss, get_seg_loss, CTCLoss_neg, DenseEnergyLoss, get_energy_loss, JointLoss
from model.SLRNet.slrnet import Net as network
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

parser.add_argument("--backbone", default='resnet38', type=str, help="vit_base_patch16_224")
parser.add_argument("--pretrained", default='/home/zaiyihu/CodeSpace/CTFA-main/scripts/pretrained/ilsvrc-cls_rna-a1_cls1000_ep-0001.pth', type=str, help="vit_base_patch16_224")
parser.add_argument("--scale_factor", default=0.5, type=str, help="vit_base_patch16_224")
parser.add_argument("--pamr_iter", default=10, type=str, help="vit_base_patch16_224")
parser.add_argument("--cutoff_top", default=0.6, type=str, help="vit_base_patch16_224")
parser.add_argument("--cutoff_low", default=0.2, type=str, help="vit_base_patch16_224")
parser.add_argument("--pamr_dilations", default=[1, 2, 4, 8, 12, 24], type=str, help="vit_base_patch16_224")
parser.add_argument("--temperature", default=1.0, type=str, help="vit_base_patch16_224")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")
# parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")

parser.add_argument("--data_folder", default='/data1/zaiyihu/Datasets/iSAID_patches_512/sampled_process', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='/home/zaiyihu/CodeSpace/CTFA-main/datasets/iSAID', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=16, type=int, help="number of classes")
parser.add_argument("--crop_size", default=320, type=int, help="crop_size in training")
parser.add_argument("--local_crop_size", default=96, type=int, help="crop_size for local view")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")

parser.add_argument("--work_dir", default="work_dir_iSAID_wseg", type=str, help="work_dir_iSAID_wseg")

parser.add_argument("--train_set", default="train", type=str, help="training split")
parser.add_argument("--val_set", default="val", type=str, help="validation split")
parser.add_argument("--spg", default=8, type=int, help="samples_per_gpu")
parser.add_argument("--scales", default=(0.5, 2), help="random rescale in training")

parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")
parser.add_argument("--lr", default=6e-5, type=float, help="learning rate")
parser.add_argument("--warmup_lr", default=1e-6, type=float, help="warmup_lr")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
parser.add_argument("--betas", default=(0.9, 0.999), help="betas for Adam")
parser.add_argument("--power", default=0.9, type=float, help="poweer factor for poly scheduler")

parser.add_argument("--max_iters", default=21250, type=int, help="max training iters")
parser.add_argument("--log_iters", default=200, type=int, help=" logging iters")
parser.add_argument("--eval_iters", default=2000, type=int, help="validation iters")
parser.add_argument("--warmup_iters", default=1500, type=int, help="warmup_iters")

parser.add_argument("--high_thre", default=0.7, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.25, type=float, help="low_bkg_score")
parser.add_argument("--bkg_thre", default=0.5, type=float, help="bkg_score")
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
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


os.environ['MASTER_ADDR'] = '127.0.0.1'

os.environ['MASTER_PORT'] = '32521'

dist.init_process_group(backend='nccl', init_method='env://', rank = 0, world_size = 1)
def balanced_mask_loss_ce(mask, pseudo_gt, gt_labels, ignore_index=255):
    """Class-balanced CE loss
    - cancel loss if only one class in pseudo_gt
    - weight loss equally between classes
    """

    mask = F.interpolate(mask, size=pseudo_gt.size()[-2:], mode="bilinear", align_corners=True)

    # indices of the max classes
    mask_gt = torch.argmax(pseudo_gt, 1)

    # for each pixel there should be at least one 1
    # otherwise, ignore
    ignore_mask = pseudo_gt.sum(1) < 1.
    mask_gt[ignore_mask] = ignore_index

    # class weight balances the loss w.r.t. number of pixels
    # because we are equally interested in all classes
    bs, c, h, w = pseudo_gt.size()
    num_pixels_per_class = pseudo_gt.view(bs, c, -1).sum(-1)
    num_pixels_total = num_pixels_per_class.sum(-1, keepdim=True)
    class_weight = (num_pixels_total - num_pixels_per_class) / (1 + num_pixels_total)
    class_weight = (pseudo_gt * class_weight[:, :, None, None]).sum(1).view(bs, -1)

    # BCE loss
    loss = F.cross_entropy(mask, mask_gt, ignore_index=ignore_index, reduction="none")
    loss = loss.view(bs, -1)

    # we will have the loss only for batch indices
    # which have all classes in pseudo mask
    gt_num_labels = gt_labels.sum(-1).type_as(loss) + 1  # + BG
    ps_num_labels = (num_pixels_per_class > 0).type_as(loss).sum(-1)
    batch_weight = (gt_num_labels == ps_num_labels).type_as(loss)

    loss = batch_weight * (class_weight * loss).mean(-1)
    return loss
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

            cls, segs = model(inputs,)

            cls_pred = (cls>0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})


            cls_pred = (cls > 0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})
            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

            #
            # valid_label = torch.nonzero(cls_label[0])[:, 0]
            # out_cam = torch.squeeze(resized_cam)[valid_label]
            # np.save(os.path.join(cfg.work_dir.pred_dir, name[0]+'.npy'), {"keys":valid_label.cpu().numpy(), "cam":out_cam.cpu().numpy()})

    cls_score = avg_meter.pop('cls_score')
    seg_score = evaluate.scores(gts, preds,16)

    model.train()

    tab_results, mIoU = format_tabs([seg_score], name_list=["Seg_Pred"], cat_list=iSAID.class_list)

    return cls_score, tab_results, mIoU

def train(args=None):

    torch.cuda.set_device(args.local_rank)
    # dist.init_process_group(backend=args.backend, )
    logging.info("Total gpus: %d, samples per gpu: %d..."%(dist.get_world_size(), args.spg))
    mIoU = 0.0001
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    train_dataset = iSAID.iSAIDClsDataset(
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

    val_dataset = iSAID.iSAIDSegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.val_set,
        stage='val',
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

    )
    lr_multi = [1.0, 2.0, 10.0, 20.0]
    param_groups = model.parameter_groups()
    param_groups = [
        {'params': param_groups[0], 'lr': lr_multi[0] * 0.001, 'weight_decay': 0.0005},
        {'params': param_groups[1], 'lr': lr_multi[1] * 0.001, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': lr_multi[2] * 0.001, 'weight_decay': 0.0005},
        {'params': param_groups[3], 'lr': lr_multi[3] * 0.001, 'weight_decay': 0},
    ]
    model.to(device)
    # Create a dummy input with the same shape as your actual input
    dummy_input = torch.randn(4, 3, 512, 512).to(device)
    # Initialize CUDA events for measuring time
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warm-up the GPU
    for _ in range(10):
        _ = model(dummy_input)

    # Record start time
    start_event.record()

    # Forward pass for inference
    with torch.no_grad():
        _ = model(dummy_input)

    # Record end time
    end_event.record()

    # Wait for GPU to finish
    torch.cuda.synchronize()

    # Calculate inference time
    inference_time_ms = start_event.elapsed_time(end_event)
    print(f"Inference Time: {inference_time_ms} ms")
    # Use THOP (THink OPtimization) library to profile the model
    flops, params = profile(model, inputs=(dummy_input,))
    # Convert params to millions (M)
    params_in_million = params / 1e6

    # Convert flops to GFLOPs
    flops_in_gigaflops = flops / 1e9

    # Print the converted values
    print(f"Number of Parameters: {params_in_million} M")
    print(f"GFLOPs: {flops_in_gigaflops} GFLOPs")
    # optimizer
    optim = torch.optim.SGD(param_groups, 0.001, momentum=0.9,
                                     weight_decay=0.0005, nesterov=True)

    logging.info('\nOptimizer: \n%s' % optim)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    train_sampler.set_epoch(np.random.randint(args.max_iters))
    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()


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
        x_aug,x2_aug,x_aux,cls,cls_affine,cls_aux,pseudo_gt, reg_loss1,  reg_loss2 = model(inputs, inputs_denorm, cls_label)
        # ----------seg_loss-------------
        seg_loss1 = balanced_mask_loss_ce(x_aug, pseudo_gt, cls_label)
        seg_loss2 = balanced_mask_loss_ce(x2_aug, pseudo_gt, cls_label)
        seg_loss_aux = balanced_mask_loss_ce(x_aux, pseudo_gt, cls_label)
        seg_loss = seg_loss1 + seg_loss2 + 0.4 * seg_loss_aux
        # classification loss
        cls_loss1 = F.multilabel_soft_margin_loss(cls, cls_label)
        cls_loss2 = F.multilabel_soft_margin_loss(cls_affine, cls_label)
        cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)
        cls_loss = cls_loss1 + cls_loss2 + 0.4 * cls_loss_aux

        cls_pred = (cls > 0).type(torch.int16)
        cls_score = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
        cls_loss = cls_loss.mean()
        seg_loss = seg_loss.mean()
        reg_loss1 = reg_loss1.mean()
        reg_loss2 = reg_loss2.mean()
        if n_iter <= 10000:
            loss = 1.0 * cls_loss + 4.0 * reg_loss1 + 4.0 * reg_loss2
        else:
            loss = 1.0 * cls_loss + 4.0 * reg_loss1 + 4.0 * reg_loss2 + 1.0 * seg_loss
        avg_meter.add({
            'cls_loss': cls_loss.item(),
            'reg_loss1': reg_loss1.item(),
            'reg_loss2': reg_loss2.item(),
            'seg_loss': seg_loss.item(),
            'loss': loss.item(),
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
                logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, reg_loss1: %.4f, reg_loss2: %.4f, seg_loss: %.4f, loss: %.4f..., cls_score: %.4f..." % (n_iter + 1, delta, eta, cur_lr, avg_meter.pop('cls_loss'), avg_meter.pop('reg_loss1'), avg_meter.pop('reg_loss2'), avg_meter.pop('seg_loss'),avg_meter.pop('loss'),avg_meter.pop('cls_score')))

        if (n_iter + 1) % args.eval_iters == 0:
            if args.local_rank == 0:
                logging.info('Validating...')
            val_cls_score, tab_results, mIoU_result = validate(model=model, data_loader=val_loader, args=args)
            if args.save_ckpt and (n_iter + 1) >= 10000 and mIoU_result > mIoU:
                mIoU = mIoU_result
                ckpt_name = os.path.join(args.ckpt_dir, "Best mIoU: {}, model: {} model_iter_%d.pth".format(mIoU,"SLRNet") % (n_iter + 1))
                torch.save(model.state_dict(), ckpt_name)
            if args.local_rank == 0:
                logging.info("val cls score: %.6f" % (val_cls_score))
                logging.info("\n"+tab_results)
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
