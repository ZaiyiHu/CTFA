import torch
import torch.nn as nn
import torch.nn.functional as F
from .single_stage import SingleStageNet, balanced_mask_loss_ce, focal_loss, pseudo_gtmask, RELU_INPLACE
from model.SLRNet.mods.transforms import resize_as, resize_to, random_hflip


def affine_transform(x, scale_factor=0.5, flip=False):
    B, _, H, W = x.shape
    out = resize_to(x, size=(int(H * scale_factor), int(W * scale_factor)))
    if flip:
        out, flip_p = random_hflip(out, return_p=True)

    def inverse_transform(y, target_size):
        if flip:
            y = random_hflip(y, p=flip_p, return_p=False)
        y = resize_to(y, size=target_size)
        return y
    return out, inverse_transform


def l2norm(inp, dim):
    '''
    Normlize the inp tensor with l2-norm.
    '''
    return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))


def max_onehot(x):
    x_max = torch.max(x, dim=1, keepdim=True)[0]
    x_ = x.clone()
    x_[x != x_max] = 0.0
    return x_


class DictionaryInitialization(nn.Module):
    def __init__(self):
        super(DictionaryInitialization, self).__init__()

    def forward(self, x, p):
        b, c, h, w = x.size()
        k = p.size(1)
        x = x.view(b, c, -1)
        # here scale using 1
        p = p.view(b, k, -1)                        # (b, c, n)
        p = p.softmax(dim=1)                       # (b, k, n)
        p = p / (p.sum(dim=-1, keepdim=True) + 1e-6)
        dict_ = x.matmul(p.permute(0, 2, 1))      # (b, c, n)(b, n, k)->(b, c, k)
        return dict_


class FactorizationReconstruction(nn.Module):
    def __init__(self, c, num_iters=1, scale=1.0, eps=1e-6):
        super().__init__()
        self.num_iters = num_iters

        self.transform1 = nn.Linear(c, c)
        self.transform2 = nn.Linear(c, c)

        self.scale = scale
        self.eps = eps

    def forward(self, x, inits):
        idn = x
        x = self.transform1(x)
        B, N, C = x.shape
        dictionary = self.transform1(inits)
        # x = l2norm(x, dim=-1)
        dictionary = l2norm(dictionary, dim=-1)
        coding = None
        for i in range(self.num_iters):
            dots = torch.einsum('bid,bjd->bij', dictionary, x) * self.scale  # (B, K, D) (B, N, D) -> (B, K, N)
            coding = dots.softmax(dim=1)
            attn = coding + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            dictionary = l2norm(torch.einsum('bjd,bij->bid', x, attn), dim=-1)  # (B, N, D) (B, K, N) -> (B, K, D)
        # reconstruction
        x = torch.einsum('bij,bid->bjd', coding, dictionary)  # (B, K, N)(B, K, D) -> (B, N, D)
        x = F.relu(x)

        x = self.transform2(x)
        x = idn + x
        x = F.relu(x)
        return x, coding, dictionary


class Net(SingleStageNet):
    def __init__(self, *args, scale_factor=0.3, use_flip=False, num_slots=64, num_iters=1, temperature=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_factor = scale_factor
        self.use_flip = use_flip
        self.num_slots = num_slots
        self.factorization_reconstruction = FactorizationReconstruction(256, num_iters=num_iters, scale=1.0 / temperature)
        self.initialize_dictionary = DictionaryInitialization()
        self.aux_head = nn.Sequential(nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
                                      self.norm_layer(256),
                                      nn.ReLU(RELU_INPLACE),
                                      nn.Dropout(0.5),
                                      nn.Conv2d(256, self.num_classes - 1, 1, stride=1))

        self.newly_added.append(self.factorization_reconstruction)
        self.newly_added.append(self.initialize_dictionary)
        self.newly_added.append(self.aux_head)
        self.init_weights()

    def backbone_forward(self, x):
        x = self.forward_backbone(x)
        x = self.aspp(x)
        x2_x = self.fc8_skip(self._backbone['conv3'])
        x_up = F.interpolate(x, size=x2_x.shape[-2:], mode='bilinear', align_corners=True)
        x = self.fc8_x(torch.cat([x_up, x2_x], dim=1))
        return x, self._backbone['conv3']

    def segmentation_forward(self, x, x_s):
        x2 = self.shallow_mask(x_s, x)
        x = self.sg(x, x2, alpha_rate=0.3)
        x = self.last_conv(x)
        bg = torch.ones_like(x[:, :1])
        x = torch.cat([bg, x], dim=1)
        masks = F.softmax(x, dim=1)
        return x, masks

    def aux_forward(self, x):
        x = self.aux_head(x)
        bg = 1 - torch.max(x, dim=1, keepdim=True)[0]
        x = torch.cat([bg, x], dim=1)
        masks = F.softmax(x, dim=1)
        return x, masks

    def classifier_forward(self, x, masks):
        """
        Pooling & focal loss
        :param x:
        :param masks:
        :return:
        """
        bs, c, h, w = x.size()
        features = x.view(bs, c, -1)
        masks_ = masks.view(bs, c, -1)
        cls_1 = (features * masks_).sum(-1) / (1.0 + masks_.sum(-1))
        # focal penalty loss
        cls_2 = focal_loss(masks_.mean(-1), p=3, c=0.01)
        cls = cls_1[:, 1:] + cls_2[:, 1:]
        return cls
    def parameter_groups(self):
        assert len(list(self.parameters())) == len(list(self.backbone.parameters())) \
               + len(list(self.newly_added.parameters())), 'param list error'
        groups = ([], [], [], [])
        for name, p in self.named_parameters():
            if name.startswith('backbone.') and name.endswith('.weight'):
                groups[0].append(p)
            elif name.startswith('backbone.') and name.endswith('.bias'):
                groups[1].append(p)
            elif name.endswith('.weight'):
                groups[2].append(p)
            elif name.endswith('.bias'):
                groups[3].append(p)
            elif name.endswith('.gamma') or name.endswith('mu') or name.endswith('sigma'):
                groups[2].append(p)
            else:
                groups[2].append(p)
                # logging.warning(f'=>Not support parameter: {name}')
        assert len(list(self.parameters())) == sum([len(g) for g in groups])
        return groups
    def forward(self, x, x_raw=None, labels=None, single_scale=False):
        test_mode = x_raw is None and labels is None
        if not isinstance(x, torch.Tensor):
            x, x2 = x
        else:
            x2 = None
        B, _, H, W = x.shape
        if test_mode:
            x, x_s = self.backbone_forward(x)
            x_aux, mask_aux = self.aux_forward(x)
            dict_init = self.initialize_dictionary(x, x_aux)
            x_, _, _ = self.factorization_reconstruction.forward(x.view(B, x.size(1), -1).permute(0, 2, 1), dict_init.permute(0, 2, 1))
            x = x_.permute(0, 2, 1).view_as(x)
            x, masks = self.segmentation_forward(x, x_s)
            cls = self.classifier_forward(x, masks)
            masks = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=True)
            return cls, masks

        # affine transform
        if x2 is None:
            x2, inv_transform = affine_transform(x, scale_factor=self.scale_factor, flip=self.use_flip)
        else:
            x2, inv_transform = affine_transform(x2, scale_factor=self.scale_factor, flip=self.use_flip)
        # base forward
        x, x_s = self.backbone_forward(x)
        x2, x2_s = self.backbone_forward(x2)

        # aux forward
        x_aux, mask_aux = self.aux_forward(x)
        dict_init = self.initialize_dictionary(x, x_aux.detach()).detach()
        dict_init = dict_init.permute(0, 2, 1)

        # cross-view MF
        C = x.size(1)
        x_multi_view = torch.cat([x.view(B, C, -1), x2.view(B, C, -1)], dim=-1)
        # x_multi_view: (B,N,C), multi_view_coding: (B,K,N), dictionary: (1,K,C)
        x_multi_view, multi_view_coding, dictionary = self.factorization_reconstruction.forward(x_multi_view.permute(0, 2, 1), dict_init)
        x_multi_view = x_multi_view.permute(0, 2, 1)  # (B, C, N1+N2)
        x_aug, x2_aug = torch.split(x_multi_view, (x.size(-1) * x.size(-2), x2.size(-1) * x2.size(-2)), dim=-1)
        x_aug = x_aug.reshape_as(x)
        x2_aug = x2_aug.reshape_as(x2)
        coding, coding2 = torch.split(multi_view_coding,
                                      (x.size(-1) * x.size(-2), x2.size(-1) * x2.size(-2)), dim=-1)
        coding = coding.view(coding.size(0), coding.size(1), x.size(2), x.size(3))
        coding2 = coding2.view(coding.size(0), coding.size(1), x2.size(2), x2.size(3))

        # seg forward (x2)
        x_aug, mask_aug = self.segmentation_forward(x_aug, x_s)
        x2_aug, mask2_aug = self.segmentation_forward(x2_aug, x2_s)

        # cls forward (x2)
        cls = self.classifier_forward(x_aug, mask_aug)
        cls_affine = self.classifier_forward(x2_aug, mask2_aug)

        # inverse transform
        target_size = x.shape[-2:]
        coding2 = inv_transform(coding2, target_size=target_size)
        x2_aug = inv_transform(x2_aug, target_size=target_size)
        mask2_aug = inv_transform(mask2_aug, target_size=target_size)

        # multi-scale test
        if test_mode:
            return (cls + cls_affine) / 2.0,\
                   F.interpolate((mask_aug + mask2_aug) / 2.0, size=(H, W),
                                 mode='bilinear', align_corners=True)

        # aux forward
        cls_aux = self.classifier_forward(x_aux, mask_aux)
        # Ensemble & Refine ! NO NEED gradients
        with torch.no_grad():
            masks = (mask_aug.detach() + mask2_aug.detach()) / 2.0
            masks_refined = self.run_pamr(x_raw, masks)
            # rescale & clean invalid categories
            masks = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=True)
            masks_refined = F.interpolate(masks_refined, size=(H, W), mode='bilinear', align_corners=True)
            masks[:, 1:] *= labels[:, :, None, None].type_as(masks)
            masks_refined[:, 1:] *= labels[:, :, None, None].type_as(masks_refined)
            pseudo_gt = pseudo_gtmask(masks_refined).detach()

        # loss for masks
        # bg is ignored
        reg_loss = torch.abs(mask_aug[:, 1:, :, :] - mask2_aug[:, 1:, :, :])
        reg_loss *= labels[:, :, None, None].type_as(reg_loss)
        reg_loss = torch.mean(reg_loss)

        code_reg_loss = torch.abs(max_onehot(coding) - max_onehot(coding2))
        code_reg_loss = torch.mean(code_reg_loss)

        return x_aug,x2_aug,x_aux,cls,cls_affine,cls_aux,pseudo_gt, reg_loss.unsqueeze(0), code_reg_loss.unsqueeze(0)
        ######################## Compute losses #############################


        # # Ensemble & Refine ! NO NEED gradients
        # with torch.no_grad():
        #     # masks = (mask.detach() + mask_affine.detach() + mask_aug.detach() + mask_affine_aug.detach()) / 4.0
        #     masks = (mask_aug.detach() + mask2_aug.detach()) / 2.0
        #     masks_refined = self.run_pamr(x_raw, masks)
        #     # rescale & clean invalid categories
        #     masks = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=True)
        #     masks_refined = F.interpolate(masks_refined, size=(H, W), mode='bilinear', align_corners=True)
        #     masks[:, 1:] *= labels[:, :, None, None].type_as(masks)
        #     masks_refined[:, 1:] *= labels[:, :, None, None].type_as(masks_refined)
        #     pseudo_gt = pseudo_gtmask(masks_refined).detach()
        # seg_loss1 = balanced_mask_loss_ce(x_aug, pseudo_gt, labels)
        # seg_loss2 = balanced_mask_loss_ce(x2_aug, pseudo_gt, labels)
        # seg_loss_aux = balanced_mask_loss_ce(x_aux, pseudo_gt, labels)
        #
        # seg_loss = seg_loss1 + seg_loss2 + 0.4 * seg_loss_aux
        #
        # # classification loss
        # cls_loss1 = F.multilabel_soft_margin_loss(cls, labels)
        # cls_loss2 = F.multilabel_soft_margin_loss(cls_affine, labels)
        # cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, labels)
        # cls_loss = cls_loss1 + cls_loss2 + 0.4 * cls_loss_aux
        #
        # return cls_loss.unsqueeze(0), seg_loss, reg_loss.unsqueeze(0), code_reg_loss.unsqueeze(0),\
        #        {'cam': masks, 'dec': masks_refined, 'pseudo': pseudo_gt}, dictionary

