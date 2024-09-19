# LiDAR-based Person Re-identification
Code and data for LiDAR-based Person Re-identification.

### [Paper](https://arxiv.org/abs/2312.03033) | [Project Page](https://github.com/GWxuan/ReID3D) | [Dataset](https://cloud.tsinghua.edu.cn/d/cdcdab829e184a698b63/)
🔥 **News**: The dataset LReID is now available!

## Introduction
Camera-based person re-identification (ReID) systems have been widely applied in the field of public security. However, cameras often lack the perception of 3D morphological information of human and are susceptible to various limitations, such as inadequate illumination, complex background, and personal privacy. In this paper, we propose a LiDAR-based ReID framework, ReID3D, that utilizes pre-training strategy to retrieve features of 3D body shape and introduces Graph-based Complementary Enhancement Encoder for extracting comprehensive features. Due to the lack of LiDAR datasets, we build LReID, the first LiDAR-based person ReID dataset, which is collected in several outdoor scenes with variations in natural conditions. Additionally, we introduce LReID-sync, a simulated pedestrian dataset designed for pre-training encoders with tasks of point cloud completion and shape parameter learning. Extensive experiments on LReID show that ReID3D achieves exceptional performance with a rank-1 accuracy of 94.0, highlighting the significant potential of LiDAR in addressing person ReID tasks. To the best of our knowledge, we are the first to propose a solution for LiDAR-based ReID.

<img src="./fig/intro.jpg" width = 60%>

## Data
<img src="./fig/dataset.jpg" width = 60%>

## Data Acquisition Scenes
<img src="./fig/scene.jpg" width = 60%>

## LReID
LReID adopts a similar file naming convention to camera-based ReID datasets, such as 0001C1T0000F000.bin, which encodes information about the person ID, camera ID, tracklet number, and frame number in the file name.

## Method
Pre-training:

<img src="./fig/method1.jpg" width = 50%>

ReID network:

<img src="./fig/method2.jpg" width = 60%>      <img src="./fig/method3.jpg" width = 30%>

## Getting Started
Installation:
```
pip install -r requirements.txt
```
Prepare LReID:
```
bash run.sh
```
Training:
```
cd reidnet
bash train.sh
```
## Citation

If you find this project helpful, please consider citing the following paper:
```
@article{guo2023lidarbased,
      title={LiDAR-based Person Re-identification}, 
      author={Wenxuan Guo and Zhiyu Pan and Yingping Liang and Ziheng Xi and Zhi Chen Zhong and Jianjiang Feng and Jie Zhou},
      journal={arXiv preprint arXiv:2308.00628},
      year={2023}
}
```
