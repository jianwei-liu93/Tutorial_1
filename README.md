<style>
r { color: Red }
o { color: Orange }
g { color: Green }
</style>

## KAIST Geometric AI Lab - Internship Week 1
Instructor: [Minhyuk Sung](https://mhsung.github.io/) (mhsung@kaist.ac.kr)  
Session Host: Hyunjin Kim (Lead), Eunji Hong, Juil Koo (Support)
Last Updated: Jun 13, 2023.

If you have any question, feel free to ask via Slack DM.

## Introduction
[PointNet](https://arxiv.org/abs/1612.00593) is a fundamental yet powerful neural network processing point cloud data. In the following programming assignments, we will learn how to use PointNet for different tasks including _classification_, _auto-encoding_, and _segmentation_. 


## Introduction to 3D Data
<b>Due: Jul 3 (Mon) </b>

The goal of the first assignment is to implement **PointNet classification, segmentation, and auto-encoding**. Since we aim to make you familiar with implementing neural network models and loss, we provide skeleton codes and what you have to do is just fill in the <r>**TODO**</r> parts of the codes. Before implementing codes, please read the [PointNet](https://arxiv.org/abs/1612.00593) paper and codes careful and check how codes flow and also how to implement codes using Pytorch.

## Project Structure

```
├── model.py              <- PointNet models implementation. <TODO>
│ 
├── dataloaders 
│   ├── modelnet.py         <- Dataloader of ModelNet40 dataset.
│   └── shapenet_partseg.py <- Dataloader of ShapeNet Part Annotation dataset. 
│
├── utils
│   ├── metrics.py          <- Easy-to-use code to compute metrics.
│   ├── misc.py             <- Point cloud normalization ft. and code to save rendered point clouds. 
│   └── model_checkpoint.py <- Automatically save model checkpoints during training.
│
├── train_cls.py          <- Run classification. <TODO>
├── train_ae.py           <- Run auto-encoding. <TODO>
├── train_seg.py          <- Run part segmentation. <TODO>
├── visualization.ipynb   <- Simple point cloud visualization example code.
│
├── data                  <- Project data.
│   ├── modelnet40_ply_hdf5_2048     <- ModelNet40   
│   └── shapenet_part_seg_hdf5_data  <- ShapeNet Part Annotation
│
└── checkpoints           <- Directory storing checkpoints. 
    ├── classification
    │    └── mm-dd_HH-MM-SS/epoch=16-val_acc=88.6.ckpt
    ├── auto_encoding
    └── segmentation
```

## Classification
**_PA1: You will get the perfect score if you achieve test accuracy over 85%._**

You can train and test a model on ModelNet40 classification by running `train_cls.py`.
It will automatically download ModelNet40 dataset on `data` directory.
```
python train_cls.py [--epochs 100] [--batch_size 32] [--lr 1e-3] [--save]
```
If you pass `--save` option, only the topk model checkpoints will be saved in `checkpoints` directory. 

On ModelNet40 test set:

|                                | Overall Acc |
| ------------------------------ | ----------- |
| Paper                          | 89.2 %      |
| Ours (w/o feature trans.)      | 88.6 %      |
| Ours (w/ feature trans.)       | 87.7 %      | 

## Auto-Encoding
**_PA1: You will get the perfect score if you achieve chamfer distance lower than 0.005 on the test set._**

To train auto-encoder, run:
```
python train_ae.py
```
On ModelNet40 test set:
|        | Chamfer Dist. |
| ------ | ------------- |
| Ours   | 0.0043        |

## Part Segmentation
### TODOs
```
- model.py
- train_seg.py
```
**_PA2: You will get the perfect score if you achieve test mIoU over 80%._**

In `model.py`, you should implement PointNet part segmentation architecture on your own. We marked where you should fill the code in as `# TODO: Implement this`.

After implementing the network, call the network in `train_seg.py`. You should be able to see progress bars:

![image](https://user-images.githubusercontent.com/37788686/158202971-159e4dc3-199a-4cf2-9b12-c01059a06a4c.png)

ShapeNet part dataset will automatically be downloaded on `data` directory when `train_seg.py` is first executed.

We provide the code to measure instance mIoU in `utils/metrics.py`.

You can refer to other existing codes on the internet, but **DO NOT COPY** other students'. We will use a **plagiarism checker**.

On ShapeNet Part test set:
|        | ins. mIoU |
| ------ | --------- |
| Paper  | 83.7 %    |
| Ours   | 83.6 %    | 
## FYI
We provide some util codes to help you. Feel free to use them if you need.
- utils/model_checkpoint.py: tracks model's metric and automatically save only topk checkpoints.
- visualization.ipynb: simple visulization code on jupyter lab.

## Reference
- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)
<!-- 
- `Classification`
<img width="647" alt="스크린샷 2022-03-14 오후 8 25 29" src="https://user-images.githubusercontent.com/37788686/158196370-c0239f51-1974-4934-bcfb-1582a0da61d4.png">

- `Auto-Encoder`
<img width="811" alt="스크린샷 2022-03-14 오후 8 24 20" src="https://user-images.githubusercontent.com/37788686/158196598-618ec2f8-bd47-4d3e-a24f-da7073b24631.png">

### What to hand in
In a single document, write your name and student ID, add screenshots of the outputs (one for each task; thus two) like the ones above, and submit the document as a **PDF** file to KLMS.

If you want, you can implement the PointNet classification network and the autoencoder yourself! (e.g., using TensorFlow). In this case, submit your code (along with the snapshot document) as a **ZIP** file.

If you implement chamfer distance yourself, take the mean across both batches and points.  
_I.e., given P1 [B,N,3] and P2 [B,N,3], Loss = ChamferDistance(P1, P2) / (B\*N), where B and N are the batch size and the number of points, respectively._

### Grading
_You'll get a zero score if you implement yourself but don't submit your code._ Otherwise,

- 100% score:
  - If you submit on time, and
  - the document includes everything (your name, student ID, and two snapshots; one for classification and the other for auto-encoding),
  - and your results are on par with ours:

| Classification  | Auto-Encoding      |
| --------------  | -------------      |
| test acc >= 85% | test loss <= 0.005 |

- 50% score:
  - If you submit on time, but
    - either one of the snapshots is missing, OR
    - one of the results is worse than the threshold above.

- 0% score: Otherwise.


## Programming Assignment 2 (Segmentation)
<b>Due: Mar 30 (Wed) 23:59 KST</b>

The goal of the second assignment is to implement PointNet segmentation network based on our code.
Our code includes script automatically downloading the [ShapeNet](https://shapenet.org/) dataset, a dataloader, main training/test loops, and the evaluation code computing mIoU. **Implement the PointNet segmentation network in the `PointNetPartSeg` class of the `model.py` file. You can even use the code in the Internet to fill in this part!**

**Check out Figure 2 (not Figure 9) in the [paper](https://arxiv.org/abs/1612.00593)) for the details of the architecture, and feel free to leave out T-Nets**

Run `train_seg.py` after filling in the `PointNetPartSeg` class of the `model.py` file. Then, the code will show the test mIoU and save some images (`segmentation_samples.png`) showing the part segmentation results:

<img width="660" alt="스크린샷 2022-03-14 오후 10 37 41" src="https://user-images.githubusercontent.com/37788686/158196815-da63ec47-04b2-468a-9247-1dae80dc612e.png">
  
<img width="424" alt="스크린샷 2022-03-15 오전 12 03 56" src="https://user-images.githubusercontent.com/37788686/158200389-a2299163-8b60-4462-bc1a-491e87355b0f.png">

### What to hand in
Submit the followings to KLMS:
- A single **PDF** file including your name, student ID, a screenshot for mIoU, and the output image visualizing some segmentation results (like the ones above).
- A **ZIP** file including ALL the code (not just the part you implemented but all).

You can also implement the entire pipeline yourself if you want.

### Grading
_You'll get a zero score if you don't submit your code._ Otherwise,

- 100% score:
  - If you submit on time, and
  - the document includes everything (your name, student ID, and a mIoU snapshot, a segmentation visualization image),
  - and your result is on par with ours:

| Segmentation |
| -----------  |
| mIoU >= 80%  |

- 50% score:
  - If you submit on time, but
    - either one of the results (a mIoU snapshot or a segmentation visualization image) is missing, OR
    - the result is worse than the threshold above.

- 0% score: Otherwise. -->
