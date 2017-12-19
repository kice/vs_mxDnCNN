Description
===========

A [MXNet](http://mxnet.incubator.apache.org/) implement of the paper "[Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](http://www4.comp.polyu.edu.hk/~cslzhang/paper/DnCNN.pdf)" for VapourSynth

Usage
=====

Require mxnet version 1.0 (You can donwload this plugin from [Here](https://github.com/kice/vs_mxDnCNN/releases))

    noise = mx.mxDNCNN(clip, [int patch_w=clip.width, int patch_h=clip.height, int param=88, int ctx=1, int dev_id=0])
    
    res = std.core.MakeDiff(clip, noise)

* clip: Clip to process. Only planar format is YUV with float sample type of 32 bit depth is supported.

* patch_w: The horizontal block size for dividing the image during processing. Smaller value results in lower VRAM usage, while larger value may not necessarily give faster speed. The optimal value may vary according to different graphics card and image size. **NOT SUPPORT NOW**
> **NOT SUPPORT NOW** Right now the plugin will use the entire image without spliting to process. Due the fact MXNet does not take a lot of GPU memory, it will take around 2G GPU memory for 1440x960 processing.

* patch_h: The same as `patch_w` but for vertical. **NOT SUPPORT NOW**

* param: Specifies which pararm to initialize the model. Currently this plugin comes with two param data.
    * 88  = a stronger denoiser [default]
    * 125 = a weaker denoiser

* ctx: Specifies which type of device to use. 
    * 1 = CPU
    * 2 = GPU
    > If GPU was chosen, cuDNN will be used by defalut.

* dev_id: Which device to use. Starting with 0.

Compilation
===========

Requires [`MXNet C predict API`](https://github.com/apache/incubator-mxnet/tree/master/include/mxnet).

It may require the lasest version of MXNet (>= 1.0.0). You can get the lasest verion DLL by install MXNet using `pip install mxnet-cu90` and copy the required dll from Python install directory.