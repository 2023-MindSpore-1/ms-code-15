# BCS-NET-MindSpore_TIM2022
Runmin Cong, Haowei Yang, Qiuping Jiang, Wei Gao, Haisheng Li, Cong Wang, Yao Zhao, BCS-Net: Boundary, Context and Semantic for Automatic COVID-19 Lung Infection Segmentation from CT Images, IEEE Transactions on Instrumentation and Measurement, 2022.

# Results of BCS-NET-MindSpore:
* Results:
  - We provide the resutls of our BCS-NET-MindSpore on COVID-19 CT segmentation dataset, and COVID-19 CT lung and infection segmentation dataset. 
    ```
    Baidu Cloud: https://pan.baidu.com/s/19_e7WLgmpYSF5l6DBOr9dQ   Password: u6bh 
    ```

# MindSpore Code of BCS-NET:
* MindSpore-GPU 1.9.0 implementation of BCS-NET
* Pretrained model:
  - We provide our code. If you want to use our code for training, please configure your dataset. and we are used the pretrained backbone ` res2net50_ascend_v190_imagenet2012_research_cv_top1acc78_top5acc94.ckpt` on mindspore.If you test our model, please download the pretrained model, unzip it, and put the checkpoint `model_BCS.pth` to `checkpoints/save_weights/` folder  and put the pretrained backbone `res2net50_ascend_v190_imagenet2012_research_cv_top1acc78_top5acc94.ckpt` to `checkpoints` folder.
  - Pretrained model download:
    ```
    Baidu Cloud: https://pan.baidu.com/s/19_e7WLgmpYSF5l6DBOr9dQ   Password: u6bh 
    ```


## Requirements

* python 3.8
* mindspore-gpu 1.9.0
* GPU cuda 11.1
```
conda install mindspore-gpu=1.9.0 cudatoolkit=11.1 -c mindspore -c conda-forge

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir  opencv-python
```


## Data Preprocessing
* Please download the test data, and put the data to `Dataset/` folder.
* COVID-19 CT segmentation dataset: https://medicalsegmentation.com/COVID19/,  COVID-19 CT lung and infection segmentation dataset: https://zenodo.org/record/3757476
* NII file processing tool and edge generation tool in `Dataset/` folder. 
* test datasets:
```
Baidu Cloud: https://pan.baidu.com/s/19_e7WLgmpYSF5l6DBOr9dQ   Password: u6bh 
```

## Test
```
python test.py
```

* You can find the results in the `'Results/'` folder.

## Document description
* trainsingleloss.py:  Calculate the gradient after summing the multiple losses returned by the network, and print the overall loss
* train.py:  Inheritance override nn.TrainOneStepCell as MultiLossTrainOneStepCell, which outputs 5 layers of loss to calculate gradient and prints 5 loops separately

# If you use our BCS-NET, please cite our paper:

    @article{BCS-NET,
        title={ BCS-Net: Boundary, Context and Semantic for Automatic COVID-19 Lung Infection Segmentation from CT Images},
        author={Runmin Cong, Haowei Yang, Qiuping Jiang, Wei Gao, Haisheng Li, Cong Wang, Yao Zhao},
        journal={IEEE TIM.},
        publisher={IEEE}
    }

# Contact Us:
If you have any questions, please contact Runmin Cong (rmcong@bjtu.edu.cn) or Haowei Yang (hwyang@bjtu.edu.cn).
