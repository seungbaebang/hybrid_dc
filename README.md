# Infinite Vector Graphics c++

## Dependency:
[libigl](https://github.com/libigl/libigl)

## Installation

First, compile static library `libigl`:

    cd libigl
    mkdir build
    cd build
    cmake .. -DLIBIGL_USE_STATIC_LIBRARY=ON
    make -j8

Then, compile our code with

    mkdir build
    cd build
    cmake ..
    make -j8


## Usage:
We provide four excutable commands `bem_brute` for standard BEM with brute force computation,`hybrid_brute` for our hybrid method with brute computation, and `hybrid_fmm` for Hybrid method + FMM, and `hybrid_fmm_adap` for hybrid method + FMM + Adaptive Subdivision.

For `bem_brute`, `hybrid_brute`, and `hybrid_fmm`, run `command input_file.txt` to run example. You can find input data for diffusion curve in `data` folder.
For example, try running follwing command:

    hybrid_fmm ../data/cherry.txt

This should generate the output `cheryy.png` file in `result` folder:

The resolution of the image will be determined as the same size as the pre-defined image size. You can customize the resolution by specifying resolution of width and height as following:

    hybrid_fmm ../data/cherry.txt 1024(resolution of width) 1024(resolution of height)

or you can simply specify a scaling value that will be multiplied o its pre-defined image size. For example,

    hybrid_fmm ../data/cherry.txt 2

will output 1024x1024 resolution of image which is twice size of pre-defined image size of 512x512

For `hybrid_fmm_adap`, run `hybrid_fmm_adap input_file.json` to run example with zoom-in. 
For example, try running follwing command:

    hybrid_fmm_adap ../data/cherry.json

This should generate the sequence of `.png` files in `result` folder:

## Note:
only confirmed to work on `Mac`.
