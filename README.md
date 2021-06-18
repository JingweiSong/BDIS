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
Jingwei Song, Qiuchen Zhu, Jianyu Lin, Maani Ghaffari "Bayesian dense inverse searching algorithm for real-time stereo matching in minimally invasive surgery." arXiv preprint arXiv:2106.07136. https://arxiv.org/abs/2106.07136      
@article{song2021bayesian,      
  title={Bayesian dense inverse searching algorithm for real-time stereo matching in minimally invasive surgery},      
  author={Song, Jingwei and Zhu, Qiuchen and Lin, Jianyu and Ghaffari, Maani},      
  journal={arXiv preprint arXiv:2106.07136},      
  year={2021}      
}      


## LICENCE CONDITIONS ##

This work is released under GPLv3 license. For commercial purposes, please contact the authors: jingweisong@yahoo.com











