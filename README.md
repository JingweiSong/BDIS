# Bayesian dense inverse searching algorithm for real-time stereo matching in minimally invasive surgery #

This is the CPU level real-time stereo software for minimally invasive surgery.

 
  
## Compiling ##

The program was only tested under a 64-bit Linux distribution.
SSE instructions from built-in X86 functions for GNU GCC were used.


```
mkdir build
cd build
cmake ../
make -j
```

The code depends on Eigen3 and OpenCV.
      

## Usage ##
1. ./run_bash_*.sh generates the disparity.      
2. Run Main.m in the matlab folder for visualization.      
      

Citation:      
@InProceedings{10.1007/978-3-031-16449-1_32,
author="Song, Jingwei
and Zhu, Qiuchen
and Lin, Jianyu
and Ghaffari, Maani",
editor="Wang, Linwei
and Dou, Qi
and Fletcher, P. Thomas
and Speidel, Stefanie
and Li, Shuo",
title="Bayesian Dense Inverse Searching Algorithm for Real-Time Stereo Matching in Minimally Invasive Surgery",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2022",
year="2022",
publisher="Springer Nature Switzerland",
address="Cham",
pages="333--344",
abstract="This paper reports a CPU-level real-time stereo matching method for surgical images (10 Hz on {\$}{\$}640 {\backslash}times 480{\$}{\$}640{\texttimes}480image with a single core of i5-9400). The proposed method is built on the fast LK algorithm, which estimates the disparity of the stereo images patch-wisely and in a coarse-to-fine manner. We propose a Bayesian framework to evaluate the probability of the optimized patch disparity at different scales. Moreover, we introduce a spatial Gaussian mixed probability distribution to address the pixel-wise probability within the patch. In-vivo and synthetic experiments show that our method can handle ambiguities resulted from the textureless surfaces and the photometric inconsistency caused by the non-Lambertian reflectance. Our Bayesian method correctly balances the probability of the patch for stereo images at different scales. Experiments indicate that the estimated depth has similar accuracy and fewer outliers than the baseline methods in the surgical scenario with real-time performance. The code and data set are available at https://github.com/JingweiSong/BDIS.git.",
isbn="978-3-031-16449-1"
}     


## LICENCE CONDITIONS ##

This work is released under GPLv3 license. For commercial purposes, please contact the authors: jingweisong@yahoo.com











