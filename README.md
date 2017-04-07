## Team

## Hardware

| Field                | Value           |
| ---                  | ---             |
| Architecture:        | x86_64          |
| CPU op-mode(s):      | 32-bit, 64-bit  |
| Byte Order:          | Little Endian   |
| CPU(s):              | 24              |
| On-line CPU(s) list: | 0-23            |
| Thread(s) per core:  | 1               |
| Core(s) per socket:  | 6               |
| Socket(s):           | 4               |
| NUMA node(s):        | 4               |
| Vendor ID:           | GenuineIntel    |
| CPU family:          | 6               |
| Model:               | 46              |
| Stepping:            | 6               |
| CPU MHz:             | 2660.155        |
| BogoMIPS:            | 5319.22         |
| Virtualization:      | VT-x            |
| L1d cache:           | 32K             |
| L1i cache:           | 32K             |
| L2 cache:            | 256K            |
| L3 cache:            | 18432K          |
| NUMA node0 CPU(s):   | 0,4,8,12,16,20  |
| NUMA node1 CPU(s):   | 1,5,9,13,17,21  |
| NUMA node2 CPU(s):   | 2,6,10,14,18,22 |
| NUMA node3 CPU(s):   | 3,7,11,15,19,23 |

## OS

Linux compute 2.6.32-504.16.2.el6.x86_64 #1 SMP Wed Apr 22 06:48:29 UTC 2015 x86_64 x86_64 x86_64 GNU/Linux

## Dependencies

* Everything is run within a docker container pulled from dockerhub. `docker pull kaggle/python`. Dockerfile found at 'https://github.com/Kaggle/docker-python'.

* 3D Convolution Network built on NEON (v1.8.2)

* Github for 3D unet https://github.com/anlthms/dsb-2017

* Download Resnet 50 layers at http://data.dmlc.ml/mxnet/models/imagenet-11k-place365-ch/


## How to use our model

Make sure you are in the same directory as train.py and predict.py before running

```
./train.py 
./python predict.py
```

If you have error running the above code try 
```
chmod +x train.py
chmod +x predict.py
```

then the above code
