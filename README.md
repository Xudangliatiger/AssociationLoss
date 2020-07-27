# Association Loss for Visual Object Detection

This project hosts the code for implementing the PSE Loss for object detection, as presented in our paper:

    Association Loss for Visual Object Detection;
    Dongli Xu, Jian Guan, Pengming Feng, and Wenwu Wang;
    arXiv preprint arXiv:.

The full paper is available at: 

##
## Updates
### 30 July 2019
   - This edition of PSE loss has been implemented via FCOS 0.1

## Required hardware
We use 1 2080Ti GPU.

## Installation

Please fallow this [INSTALL.md](INSTALL.md) for installation instructions.
You may also want to see the original [README.md](MASKRCNN_README.md) of maskrcnn-benchmark.

## A quick demo
Once the installation is done, you can follow the below steps to run a quick demo.
    
    # assume that you are under the root directory of this project,
    # and you have activated your virtual environment if needed.
    wget https://cloudstor.aarnet.edu.au/plus/s/dDeDPBLEAt19Xrl/download -O FCOS_R_50_FPN_1x.pth
    python demo/fcos_demo.py


## Inference
The inference command line on coco minival split:

    python tools/test_net.py \
        --config-file configs/fcos/fcos_R_50_FPN_1x.yaml \
        MODEL.WEIGHT models/FCOS_R_50_FPN_1x.pth \
        TEST.IMS_PER_BATCH 4    

Please note that:
1) If your model's name is different, please replace `models/FCOS_R_50_FPN_1x.pth` with your own.
2) If you enounter out-of-memory error, please try to reduce `TEST.IMS_PER_BATCH` to 1.
3) If you want to evaluate a different model, please change `--config-file` to its config file (in [configs/fcos](configs/fcos)) and `MODEL.WEIGHT` to its weights file.


## Training

The following command line will train FCOS_R_50_FPN_1x on 8 GPUs with Synchronous Stochastic Gradient Descent (SGD):

    python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port=$((RANDOM + 10000)) \
        tools/train_net.py \
        --skip-test \
        --config-file configs/fcos/fcos_R_50_FPN_1x.yaml \
        DATALOADER.NUM_WORKERS 2 \
        OUTPUT_DIR training_dir/fcos_R_50_FPN_1x
        
Note that:
1) If you want to use fewer GPUs, please change `--nproc_per_node` to the number of GPUs. No other settings need to be changed. The total batch size does not depends on `nproc_per_node`. If you want to change the total batch size, please change `SOLVER.IMS_PER_BATCH` in [configs/fcos/fcos_R_50_FPN_1x.yaml](configs/fcos/fcos_R_50_FPN_1x.yaml).
2) The models will be saved into `OUTPUT_DIR`.
3) If you want to train FCOS with other backbones, please change `--config-file`.
4) The link of ImageNet pre-training X-101-64x4d in the code is invalid. Please download the model [here](https://cloudstor.aarnet.edu.au/plus/s/k3ys35075jmU1RP/download).
5) If you want to train FCOS on your own dataset, please follow this instruction [#54](https://github.com/tianzhi0549/FCOS/issues/54#issuecomment-497558687).
## Contributing to the project

Any pull requests or issues are welcome.

## Citations
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.
```
@article{xu2020assloss,
  title   =  {Association Loss for Object Detection},
  author  =  {Xu, Dongli and Guan, Jian and Feng, Pengming and Wenwu, Wang},
  journal =  {arXiv preprint arXiv:},
  year    =  {2020}
}
```


## License

For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact the authors. 
