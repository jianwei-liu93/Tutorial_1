## KAIST Geometric AI Lab - Internship Week 1
Instructor: [Minhyuk Sung](https://mhsung.github.io/) (mhsung@kaist.ac.kr)
Session Host: **Hyunjin Kim** (Lead), **Eunji Hong** (Support)
Last Updated: Jun 13, 2023.

If you have any question, feel free to ask us via Slack DM.

## Introduction to 3D Data
<b>Due: Jul 3 (Mon) </b>

[PointNet](https://arxiv.org/abs/1612.00593) is a fundamental yet powerful neural network processing point cloud data. In the first tutorial, we will learn how to use PointNet for different tasks including _classification_, _auto-encoding_, and _segmentation_ by implementing them. Since we aim to make you familiar with implementing neural network models and losses using Pytorch, we provide skeleton codes and what you have to do is just fill in the **TODO** parts of the codes. Before implementing codes, please read the [PointNet](https://arxiv.org/abs/1612.00593) paper and the provided codes careful and check how codes flow. Also, we recommend you to read how to implement codes using Pytorch ([Tutorial link](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html)).

## Project Structure
Below shows the overall structure of this repository. Bascially, in this tutorial, what you have to do is implementing models and losses by filling in the **TODO** parts of below 4 files.
 ### TODOs
```
- model.py
- train_cls.py
- train_ae.py
- train_seg.py
```

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

## Task 0. Global feature extraction
![image](https://github.com/KorMachine/Internship_Week1/assets/43216580/d02a0a66-4f82-4e15-9245-9d8003a48c5e)

PointNet takes 3D point clouds(# points, 3) as inputs and extracts a 1024-sized global feature latent vector, which contains the geometric information of the input point clouds. This global feature vector will be used in the downstream tasks; point cloud classification, segmentation, and auto-encoding. In this part, you implement PointNetFeat model that only results out the global feature vector so that you can utilize this model for implementing the remaining 3 tasks.

### To-dos
```
- model.py
```
- Fill in the blank in model.py > PointNetFeat class

※ When implementing PointNetFeat, you can utilize STDkd we give you in model.py code. 


## Task 1. Point cloud classification
**_Success condition: You will get the perfect score if you achieve test accuracy over 85%._**
![image](https://github.com/KorMachine/Internship_Week1/assets/43216580/d3471b20-5325-47ab-a68e-97a6774dcc69)

In point cloud classification tasks, PointNet inputs point clouds (# points, 3) and generates a 1024-sized global feature latent vector, which is then reduced to the number of categories (k) through multi-layer perceptrons, forming logits for each category. 

### To-dos
```
- model.py
- train_cls.py
```
- Fill in the blank in [model.py](http://model.py) > PointNetCls
- Fill in the blank in train_cls.py > step and train_step
- Train the model by the following command.

```bash
python train_cls.py [--epochs 100] [--batch_size 32] [--lr 1e-3] [--save]
```

If you pass `--save` option, only the topk model checkpoints will be saved in `checkpoints` directory. 

On ModelNet40 test set:

|                                | Overall Acc |
| ------------------------------ | ----------- |
| Paper                          | 89.2 %      |
| Ours (w/o feature trans.)      | 88.6 %      |
| Ours (w/ feature trans.)       | 87.7 %      | 


## Task 2. Point cloud part Segmentation
**_Success condition: You will get the perfect score if you achieve test mIoU over 80%._**
![image](https://github.com/KorMachine/Internship_Week1/assets/43216580/af71824a-bb94-48af-a8bd-2a81e1ce6e56)

For segmentation tasks, PointNet concatenates the second transformed feature with the global latent vector to form a point-wise feature tensor, which is then passed through an MLP to produce logits for m part labels.

### To-dos
```
- model.py
- train_seg.py
```


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

## Task 3. Point cloud Auto-Encoding
**_success condition: You will get the perfect score if you achieve chamfer distance lower than 0.005 on the test set._**
![image](https://github.com/KorMachine/Internship_Week1/assets/43216580/eb932953-3ecb-4b8c-9129-03b15494bcb5)

The PointNet Auto-encoder comprises an encoder that inputs point clouds and produces a 1024-sized global feature latent vector, and an MLP decoder that expands this latent vector incrementally until it reaches N*3. This tensor is reshaped into (N, 3), representing N points in 3D coordinates.

### To-dos
```
- model.py
- train_seg.py
```
- Fill in the blank in [model.py](http://model.py) > `PointNetAutoEncoder`
- Fill in the blank in train_ae.py > `step` and `train_step`
- Train the model by the following command.

```
python train_ae.py
```

On ModelNet40 test set:
|        | Chamfer Dist. |
| ------ | ------------- |
| Ours   | 0.0043        |


## Reference
- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)
