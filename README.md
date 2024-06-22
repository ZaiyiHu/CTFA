## Contrastive Tokens and Label Activation for Remote Sensing Weakly Supervised Semantic Segmentation

This is the official repository of TGRS 2024 paper: *Contrastive Tokens and Label Activation for Remote Sensing Weakly Supervised Semantic Segmentation*.


<div align="center">

<br>
  <img width="100%" alt="Framework of CTFA" src="./docs/imgs/Framework of CTFA.jpg">
</div>

## Abstract 

In recent years, there has been remarkable progress in Weakly Supervised Semantic Segmentation (WSSS), with Vision Transformer (ViT) architectures emerging as a natural fit for such tasks due to their inherent ability to leverage global attention for comprehensive object information perception. However, directly applying ViT to WSSS tasks can introduce challenges. The characteristics of ViT can lead to an over-smoothing problem, particularly in dense scenes of remote sensing images, significantly compromising the effectiveness of Class Activation Maps (CAM) and posing challenges for segmentation. Moreover, existing methods often adopt multi-stage strategies, adding complexity and reducing training efficiency.

To overcome these challenges, a comprehensive framework
CTFA (Contrastive Token and Foreground Activation) based on
the ViT architecture for WSSS of remote sensing images is
presented. Our proposed method includes a Contrastive Token
Learning Module (CTLM), incorporating both patch-wise and
class-wise token learning to enhance model performance. In
patch-wise learning, we leverage the semantic diversity preserved
in intermediate layers of ViT and derive a relation matrix from
these layers and employ it to supervise the final output tokens,
thereby improving the quality of CAM. In class-wise learning,
we ensure the consistency of representation between global and
local tokens, revealing more entire object regions. Additionally, by
activating foreground features in the generated pseudo label using
a dual-branch decoder, we further promote the improvement of
CAM generation. Our approach demonstrates outstanding results
across three well-established datasets, providing a more efficient
and streamlined solution for WSSS.

## Bugs to Fix
At present, the code repository seems to have the following issues that need to be addressed:
#### 1) There is an error in the CRF code.
① For the potsdam dataset, simply modify line 117 in infer_seg_potsdam.py to: labelname=os. path. coin (labels_path, name+". png");
② For the iSAID dataset, I should not have uploaded unprocessed labels through the link on Baidu Cloud. You can use a color mapping table to convert RGB images into category images, please see the attachment. Subsequently, in the def_job (i) of the CRF processing, add label=imageio. imread (labelname) followed by label=convertecolor_to_class (label) conversion;
③ For the deepglobal dataset, this issue should not exist. However, it should be noted that during crf processing, all datasets should change the num_classes in the 21 rows def scores() in evaluate.py accordingly (16,6,7). Alternatively, you can directly modify the parameters provided.
#### 2) When training the Deepglobal model, the background was not used. Therefore, the following modifications need to be made:
① Modify the model section: model_seg_ceg_fp. py: line 71-73: The classifier output dimension does not need to be further reduced because there is no background left.
self.classifier = nn.Conv2d(in_channels=self.in_channels[-1],  out_channels=self.num_classes, kernel_size=1,
bias=False, )
self.aux_classifier = nn.Conv2d(in_channels=self.in_channels[-1],  out_channels=self.num_classes,
kernel_size=1, bias=False, )
Line 170-171: Similarly, when outputting classifications, modifications are also needed:
cls_x4 = cls_x4.view(-1, self.num_classes)
cls_aux = cls_aux.view(-1, self.num_classes)
② Modify the cam section: camutils'ori. py: line 13: No longer need to add+=1 to the pseudo label, otherwise it will be out of bounds.
# _pseudo_label += 1
Line 377, refine_camb_with_bkg_v2(): Background related information is no longer needed.
# For deepglobe dataset training,  since no explicit backgroudn category is given, cls_labels no longer needs to cat with bkg_cls. So does the cams.
# For instance,  you can change the code like this:
'''
b, _, h, w = images.shape
_images = F.interpolate(images, size=[h // down_scale, w // down_scale], mode="bilinear",  align_corners=False)
refined_label = torch.ones(size=(b,  h, w)) * ignore_index
refined_label = refined_label.to(cams.device)
refined_label_h = refined_label.clone()
refined_label_l = refined_label.clone()
cams_with_bkg_h = cams
_cams_with_bkg_h = F.interpolate(cams_with_bkg_h, size=[h // down_scale, w // down_scale], mode="bilinear",
align_corners=False)  # .softmax(dim=1)
cams_with_bkg_l = cams
_cams_with_bkg_l = F.interpolate(cams_with_bkg_l, size=[h // down_scale, w // down_scale], mode="bilinear",
align_corners=False)
'''

## Data Preparations
<details>
<summary>
iSAID dataset
</summary>

#### 1. Data Download

You may download the iSAID dataset from their official webiste https://captain-whu.github.io/iSAID/dataset.html.


#### 2. Data Preprocessing
After downloading, you may craft your own dataset. Please refer to datasets/iSAID/make_data.py.

</details>

<details>

<summary>
ISPRS Potsdam dataset
</summary>

#### 1. Data Download
Datasets for ISPRS Potsdam are widely accessible on the Internet. You may find the original content on: https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx. 

#### 2. Data Preprocessing
You may refer to the datasets/potsdam/potsdam_clip_dataset.py provided by [OME](https://github.com/NJU-LHRS/OME). Great thanks for their brilliant works.
</details>

<details>

<summary>
Deepglobe Land Cover Classification Dataset
</summary>

#### 1. Data Download
You may find the original content on:http://deepglobe.org/challenge.html. 

#### 2. Data Preprocessing
Please refer to datasets/deepglobe/deepglobe_clip_dataset.py.

</details>

We also provide the BaiduNetDiskDownload link for processed dataset at [Here](https://pan.baidu.com/s/1hkCg8uX2oGpNYORL2KfZug). Code: CTFA



## Create environment
We provide our requirements file for building the environemnt. Note that extra packages may be downloaded.
``` bash 
## Download Dependencies.
pip install -r requirements.txt 
```

### Build Reg Loss

To use the regularized loss, download and compile the python extension, see [Here](https://github.com/meng-tang/rloss/tree/master/pytorch#build-python-extension-module).

### Train
Please refer to the scripts folder, where all scripts are clared by their name. You can also modify them to distributed training, which cost more GPUs. A simple startup like this:
```bash
## for iSAID
python dist_train_iSAID_seg_neg_fp.py
## for potsdam
python dist_train_postdam_seg_neg_fp.py
## for deepglobe
python dist_train_deepglobe_seg_neg_fp.py
```
For deepglobe dataset, due to the lack of clear background partitioning in this dataset, some modifications may be necessary. Please check the annotations in camutils_ori.py.

You should remember to change the data path to your own and make sure all setting are matched.

I will try my best to reorganize the code to minimize issues. Apologize for any inconvenience caused by the code issues and thank you for your understanding.

### Evalution
To evaluation:
```bash
## for iSAID
python infer_seg_iSAID.py
...
```


## Acknowledgement

Our work is built on the codebase of [ToCo](https://github.com/rulixiang/ToCo) and [Factseg](https://github.com/Junjue-Wang/FactSeg). We sincerely thank for their exceptional work.
