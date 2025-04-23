# A Dual-Generalization Low-Light Enhancement Framework for Capsule Endoscopy Image Restoration and Segmentation

**Shuocheng Wang, Jiaming Liu, Ruoxi Zhu, Chengkang Huang, Minge Jing, and Yibo Fan**

**State Key Laboratory of Integrated Chips and Systems, Fudan University**



## Abstract

In recent years, deep learning technology has automated the diagnosis of gastrointestinal (GI) tract disease, enabling doctor-machine collaborative diagnosis. However, the images captured by wireless capsule endoscopy (WCE) easily suffer from insufficient illumination due to the complex structure of GI tract and the limitations of light source, which impacts both human and machine diagnostic accuracy. Moreover, the image features required for human visual inspection and machine-based diagnosis are different, making the images obtained by general low-light enhancement algorithms often unsuitable for machine-based tasks such as image segmentation. To address these issues, we propose a dual-generalization framework for low-light WCE images. Our framework includes an Image Guidance and Laplacian Fusion Module (IGLFM), a Brightness Level Generalization Module (BLGM) and a Wavelet Segmentation Generalization Module (WSGM). IGLFM and BLGM can restore low-light images across different brightness levels and WSGM is designed to enhance segmentation performance. With BLGM and WSGM, our framework enables two aspects of generalizations: generalization to input images with different luminance levels, and generalization to different downstream diagnostic tasks (human perception and image segmentation). Extensive experiments demonstrate that our method achieves significant improvements in both visual quality and segmentation performance, surpassing the existing state-of-the-art (SOTA) method by 4.62 dB in PSNR and 0.022 in SSIM on Kvasir-Capsule dataset. Our code will be available at https://github.com/superwsc/Dual-Gen-Frame.
