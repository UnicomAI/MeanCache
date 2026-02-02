# MeanCache4Qwen-Image

[MeanCache](https://github.com/UnicomAI/LeMiCa) already supports accelerated inference for [Qwen-Image](https://github.com/QwenLM/Qwen-Image) and provides three optional acceleration paths based on the balance between quality and speed.


## 📊 Performance & Latency 


### 🚀 MeanCache vs. LeMiCa: Speedup Comparison



**Baseline Latency (Original Qwen-Image-2512): 32.8s**

| Constraint | Method | Latency | Speedup | Time Reduction |
|:---:|:---|:---:|:---:|:---:|
| **$B=25$** | LeMiCa | 18.83 s | 1.74x | - |
| | **MeanCache** | **17.13 s** | **1.91x** | **9.0%** |
| **$B=17$** | LeMiCa | 14.35 s | 2.29x | - |
| | **MeanCache** | **11.67 s** | **2.81x** | **18.7%** |
| **$B=10$** | LeMiCa | 10.41 s | 3.15x | - |
| | **MeanCache** | **6.95 s** | **4.72x** | **33.2%** |

> **Note:** Under identical caching constraints ($B$), **MeanCache** consistently delivers superior latency reduction compared to LeMiCa. Especially in high-acceleration settings (e.g., $B=10$), MeanCache achieves an additional **33.2% time reduction** beyond LeMiCa's performance.




### Qwen-Image-2512

| Method | Qwen-Image-2512 | MeanCache (B=25) | MeanCache (B=17) | MeanCache (B=10) |
|:-------:|:-------:|:-----------:|:-------------:|:-----------:|
| **Latency** | 32.8 s | **17.13 s** | **11.67 s** | **6.95 s** |
| **T2I** | <img width="160" alt="Qwen-Image-2512" src="https://github.com/user-attachments/assets/9ce355df-c745-47fc-8d85-e91fa32dd071" /> | <img width="160" alt="Meancache_b25" src="https://github.com/user-attachments/assets/11130a7a-f6fe-41e7-b7d0-5de9590930c1" /> | <img width="160" alt="Meancache_b17" src="https://github.com/user-attachments/assets/ef5fd9ee-bf93-4634-a98b-0d51418b3cf4" /> | <img width="160" alt="Meancache_b10" src="https://github.com/user-attachments/assets/10296ab6-45dc-482b-a92c-445a553c9e52"/> |





### Qwen-Image

| Method | Qwen-Image | MeanCache (B=25) | MeanCache (B=17) | MeanCache (B=10) |
|:-------:|:-------:|:-----------:|:-------------:|:-----------:|
| **Latency** | 33.13 s | **17.04 s** | **11.63 s** | **6.92 s** |
| **T2I** | <img width="160" alt="Qwen-Image" src="https://github.com/user-attachments/assets/91e556cf-438f-43e5-a825-5c3a0980df17" /> | <img width="160" alt="Meancache_b25" src="https://github.com/user-attachments/assets/1a5aba9d-650c-4d4e-a588-c1fdd124aa30" /> | <img width="160" alt="Meancache_b17" src="https://github.com/user-attachments/assets/9c854138-078d-4c19-87f7-e7e5221fac43" /> | <img width="160" alt="Meancache_b10" src="https://github.com/user-attachments/assets/fd1cb552-ec3e-47fb-88ed-9a914e6b0a22"/> |

---

## 🛠️ Installation & Usage 

Please refer to [Qwen-Image](https://github.com/QwenLM/Qwen-Image) for basic environment setup.

MeanCache provides different acceleration modes by controlling the --cache-jvp parameter (smaller B values result in faster inference):

```shell
# Basic usage with different cache steps
python Mc_qwenimage.py
python Mc_qwenimage.py --cache-jvp 25
python Mc_qwenimage.py --cache-jvp 17
python Mc_qwenimage.py --cache-jvp 10

# For Qwen-Image-2512
# Ensure the model_name is set to "Qwen/Qwen-Image-2512" in the script
python Mc_qwenimage_2512.py --cache-jvp 25
python Mc_qwenimage_2512.py --cache-jvp 17
python Mc_qwenimage_2512.py --cache-jvp 10
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