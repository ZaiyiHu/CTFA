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
