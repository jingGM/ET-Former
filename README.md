<div align="center">   
  
# ET-Former: Efficient Triplane Deformable Attention for 3D Semantic Scene Completion From Monocular Camera
</div>

![](./assets/teaser.gif "")

[//]: # (> **ET-Former: Efficient Triplane Deformable Attention for 3D Semantic Scene Completion From Monocular Camera**.)

> [Jing Liang](https://jingliangc.github.io/), [He Yin](https://scholar.google.com/citations?hl=en&user=hKMVC8IAAAAJ), [Xuewei Qi](https://scholar.google.com/citations?hl=en&user=pOA6uKMAAAAJ&view_op=list_works&sortby=pubdate), [Jong Jin Park](https://scholar.google.com/citations?user=W-W1ew4AAAAJ), [Min Sun](https://scholar.google.com/citations?user=1Rf6sGcAAAAJ), [Min Sun](https://scholar.google.com/citations?user=1Rf6sGcAAAAJ), [Rajasimman Madhivanan](https://www.amazon.science/author/rajasimman-madhivanan), [Dinesh Manocha](https://scholar.google.com/citations?user=X08l_4IAAAAJ)


>  [[PDF]](https://arxiv.org/abs/2410.11019) [[Project]](https://github.com/jingGM/ET-Former.git) [[Intro Video]](https://youtu.be/DcXVHMpL4oQ?si=Ey5jeSMXcAdiKcAU) 


## News
- [2025/02]: We submitted the paper to IROS 2025;
</br>


## Abstract
We introduce ET-Former, a novel end-to-end algorithm for semantic scene completion using a single monocular camera. Our approach generates a semantic occupancy map from single RGB observation while simultaneously providing uncertainty estimates for semantic predictions. By designing a triplane-based deformable attention mechanism, our approach improves geometric understanding of the scene than other SOTA approaches and reduces noise in semantic predictions. Additionally, through the use of a Conditional Variational AutoEncoder (CVAE), we estimate the uncertainties of these predictions. The generated semantic and uncertainty maps will help formulate navigation strategies that facilitate safe and permissible decision making in the future. Evaluated on the Semantic-KITTI dataset, ET-Former achieves the highest Intersection over Union (IoU) and mean IoU (mIoU) scores while maintaining the lowest GPU memory usage, surpassing state-of-the-art (SOTA) methods. It improves the SOTA scores of IoU from 44.71 to 51.49 and mIoU from 15.04 to 16.30 on SeamnticKITTI test, with a notably low training memory consumption of 10.9 GB.

## Method

|                                                                                                                                                                                                                                                                                                                                                                              ![space-1.jpg](assets/architecture.png)                                                                                                                                                                                                                                                                                                                                                                              | 
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:| 
| ***Figure 1. Overall Architecture of ET-Former**. We present a two-stage pipeline for processing mono-cam images and generate both a semantic occupancy map m_s and its corresponding uncertainty map m_u. In stage 1, we introduce a novel triplane-based deformable attention model to generate the occupancy queries m_o from the given mono-cam images, which reduces high-dimensional 3D feature processing to 2D computations. In stage 2, we employ the efficient triplane-based deformable attention mechanism to generate the semantic map, with the inferred voxels from stage 1 as input and conditioned on the RGB image. To estimate the uncertainty in the semantic map, we incorporate a CVAE method, and quantify the uncertainty using the variance of the CVAE latent samples.* |

## Getting Started
- The code will come soon.

## Dataset

- [x] SemanticKITTI

[//]: # (## Bibtex)

[//]: # (If this work is helpful for your research, please cite the following BibTeX entry.)

[//]: # ()
[//]: # (```)

[//]: # (@InProceedings{li2023voxformer,)

[//]: # (      title={VoxFormer: Sparse Voxel Transformer for Camera-based 3D Semantic Scene Completion}, )

[//]: # (      author={Li, Yiming and Yu, Zhiding and Choy, Christopher and Xiao, Chaowei and Alvarez, Jose M and Fidler, Sanja and Feng, Chen and Anandkumar, Anima},)

[//]: # (      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition &#40;CVPR&#41;},)

[//]: # (      year={2023})

[//]: # (})

[//]: # (```)

## License

[//]: # (Copyright © 2022-2023, NVIDIA Corporation and Affiliates. All rights reserved.)

[//]: # ()
[//]: # (This work is made available under the Nvidia Source Code License-NC. Click [here]&#40;https://github.com/NVlabs/VoxFormer/blob/main/LICENSE&#41; to view a copy of this license.)

[//]: # ()
[//]: # (The pre-trained models are shared under [CC-BY-NC-SA-4.0]&#40;https://creativecommons.org/licenses/by-nc-sa/4.0/&#41;. If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.)

[//]: # ()
[//]: # (For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing]&#40;https://www.nvidia.com/en-us/research/inquiries/&#41;.)

## Acknowledgement

Many thanks to these excellent open source projects:
- [MonoScene](https://github.com/astra-vision/MonoScene)
- [VoxFormer](https://github.com/NVlabs/VoxFormer)
