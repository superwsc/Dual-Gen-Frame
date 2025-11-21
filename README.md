# A Dual-Generalization Low-Light Enhancement Framework for Capsule Endoscopy Image Restoration and Segmentation

**TMI 2025**

**Shuocheng Wang, Jiaming Liu, Ruoxi Zhu, Chengkang Huang, Minge Jing, and Yibo Fan**

**State Key Laboratory of Integrated Chips and Systems, Fudan University**



## Abstract

In recent years, deep learning technology has automated the diagnosis of gastrointestinal (GI) tract disease, enabling doctor-machine collaborative diagnosis. However, the images captured by wireless capsule endoscopy (WCE) easily suffer from varying brightness levels of low-light degradation due to the complex structure of GI tract and the limitations of the light source, which impacts both human and machine diagnostic accuracy. Moreover, images may contain varying degrees of structural and semantic details even under a similar brightness level, which still can compromise segmentation accuracy. To address these issues, we propose a dual-generalization framework for low-light WCE images. Our framework includes an Image Guidance and Laplacian Fusion Module (IGLFM), a Brightness Level Generalization Module (BLGM) and a Wavelet Segmentation Generalization Module (WSGM). IGLFM and BLGM can restore low-light images across different brightness levels and WSGM can enhance segmentation accuracy by generalizing the varying degrees of details across images. With BLGM and WSGM, our framework enables two aspects of generalization: generalization to input images with different brightness levels and generalization to images with varying detail levels. Extensive experiments demonstrate that our method achieves significant performance under varying brightness levels and improvements in segmentation accuracy, surpassing the existing state-of-the-art (SOTA) method with gains of 4.70 dB / 0.022 (PSNR/SSIM) on Kvasir-Capsule dataset and 1.61 dB / 0.018 on RLE dataset. WSGM consistently improves segmentation accuracy across six popular networks, achieving up to +4.7% mIoU and +5.3% Dice improvements on RLE dataset. Our code will be available at https://github.com/superwsc/Dual-Gen-Frame.

## Inference


