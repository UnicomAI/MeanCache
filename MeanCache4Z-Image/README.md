# MeanCache-Z-Image

[MeanCache](https://arxiv.org/abs/2601.19961) now supports accelerated inference for [Z-Image](https://github.com/Tongyi-MAI/Z-Image), providing three optional acceleration paths that effectively balance image quality and generation speed.

## 📊 Inference Latency
#### Comparisons on a Single H800 GPU


| Z-Image-base | MeanCache(B=25) | MeanCache(B=20) | MeanCache(B=15) | MeanCache(B=13) |
|:-------:|:-----------:|:-------------:|:-----------:|:-----------:|
| 18.07 s | 9.15 s | 7.36 s | 5.58 s | 4.85 s |
|<img src="https://github.com/user-attachments/assets/6f3b7858-f0e7-41f5-86e2-239a8e281215" width="117" height="208" style="object-fit: cover; display: block;"> | <img src="https://github.com/user-attachments/assets/85a5211d-358a-462f-a669-cce31f3660ce" width="117" height="208" style="object-fit: cover; display: block;"> | <img src="https://github.com/user-attachments/assets/146f4070-2a44-4635-b9b9-257e3e157c26" width="117" height="208" style="object-fit: cover; display: block;"> | <img src="https://github.com/user-attachments/assets/51581088-60ee-40ec-ac26-48621c3ab0a7" width="117" height="208" style="object-fit: cover; display: block;"> | <img src="https://github.com/user-attachments/assets/da1a649f-93a6-43de-bd6e-b8386bd0467e" width="117" height="208" style="object-fit: cover; display: block;"> |

## 🛠️ Installation & Usage

Please refer to the original [Z-Image]([Z-Image](https://github.com/Tongyi-MAI/Z-Image) ) project for base installation instructions.


```bash

# Baseline: Run vanilla Z-Image (No Acceleration)
python MC_zimage.py

# MeanCache: Accelerated Inference
python MC_zimage.py.py --cache 25
python MC_zimage.py.py --cache 20
python MC_zimage.py.py --cache 15
python MC_zimage.py.py --cache 13
```


## 📖 Citation
If you find **MeanCache** useful in your research or applications, please consider giving us a star ⭐ and citing it by the following BibTeX entry:



```bibtex
@inproceedings{gao2025meancache,
  title     = {MeanCache: From Instantaneous to Average Velocity for Accelerating Flow Matching Inference},
  author    = {Huanlin Gao and Ping Chen and Fuyuan Shi and Ruijia Wu and Yantao Li and Qiang Hui and Yuren You and Ting Lu and Chao Tan and Shaoan Zhao and Zhaoxiang Liu and Fang Zhao and Kai Wang and Shiguo Lian},
  journal   = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2601.19961}
}
```