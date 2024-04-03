## Introduction

This repository contains Ruijie Tao's unofficial reimplementation of the standard [ECAPA-TDNN](https://arxiv.org/pdf/2005.07143.pdf).

This repository is modified based on [voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer) and [TaoRuijie/ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN).

In this project, we use CN-Celeb2 Dataset to train ECAPA-TDNN model.

## Best Performance in this project

| Dataset  | CN-Celeb2 |
|----------|-----------|
| EER      | 2.99%     |
| Accuracy | 79.60%    |


***
## Dependencies

Create the environment:
```
conda create -n ECAPA python=3.7
conda activate ECAPA
pip install -r requirements.txt
```


## Data preparation


Dataset for training: 

1) CN-Celeb2 training set;
```
wget https://us.openslr.org/resources/82/cn-celeb2_v2.tar.gzaa
wget https://us.openslr.org/resources/82/cn-celeb2_v2.tar.gzab
wget https://us.openslr.org/resources/82/cn-celeb2_v2.tar.gzac
cat cn-celeb2_v2.tar.gzaa cn-celeb2_v2.tar.gzab cn-celeb2_v2.tar.gzac > cn-celeb2_v2.tar.gz
tar -xzvf cn-celeb2_v2.tar.gz
```

2) MUSAN dataset;
```
wget https://us.openslr.org/resources/17/musan.tar.gz
tar -xzvf musan.tar.gz
```

3) RIR dataset.
```
wget https://us.openslr.org/resources/28/rirs_noises.zip
unzip rirs_noises.zip
```


## Training

We provide data/train_files.txt and data/test_pairs.txt. You can use these lists to train and evaluate.

Or you can use `GenerateList()` in `tools.py` to generate the train_list.txt and test_pairs.txt randomly.

Then you can change the data path in the `train.py`. Train ECAPA-TDNN model end-to-end by using:

```
python train.py --save_path exps/[your_exp_name]
```

Every `test_step` epoches, system will be evaluated and print the EER. 

The result will be saved in `exps/[your_exp_name]/score.txt`. The model will saved in `exps/[your_exp_name]/model`

In my case, I trained 80 epoches in one 3090 GPU. Each epoch takes 90 mins.


### Reference

Original ECAPA-TDNN paper
```
@inproceedings{desplanques2020ecapa,
  title={{ECAPA-TDNN: Emphasized Channel Attention, propagation and aggregation in TDNN based speaker verification}},
  author={Desplanques, Brecht and Thienpondt, Jenthe and Demuynck, Kris},
  booktitle={Interspeech 2020},
  pages={3830--3834},
  year={2020}
}
```

Ruijie Tao's reimplement report
```
@article{das2021hlt,
  title={HLT-NUS SUBMISSION FOR 2020 NIST Conversational Telephone Speech SRE},
  author={Das, Rohan Kumar and Tao, Ruijie and Li, Haizhou},
  journal={arXiv preprint arXiv:2111.06671},
  year={2021}
}
```

VoxCeleb_trainer paper
```
@inproceedings{chung2020in,
  title={In defence of metric learning for speaker recognition},
  author={Chung, Joon Son and Huh, Jaesung and Mun, Seongkyu and Lee, Minjae and Heo, Hee Soo and Choe, Soyeon and Ham, Chiheon and Jung, Sunghwan and Lee, Bong-Jin and Han, Icksang},
  booktitle={Interspeech},
  year={2020}
}
```

### Acknowledge

We study many useful projects in our codeing process, which includes:

[clovaai/voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer).

[lawlict/ECAPA-TDNN](https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py).

[speechbrain/speechbrain](https://github.com/speechbrain/speechbrain/blob/96077e9a1afff89d3f5ff47cab4bca0202770e4f/speechbrain/lobes/models/ECAPA_TDNN.py)

[ranchlai/speaker-verification](https://github.com/ranchlai/speaker-verification)

Thanks for these authors to open source their code!
