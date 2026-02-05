# [ICLR 2026] MeanCache: From Instantaneous to Average Velocity for Accelerating Flow Matching Inference

<div class="is-size-5 publication-authors" align="center">
  <span class="author-block">
    <a href="https://github.com/joelulu" target="_blank">Huanlin Gao</a><sup>1,2</sup>,
  </span>
  <span class="author-block">
    <a href="https://scholar.google.com/citations?hl=zh-CN&view_op=list_works&user=gpNOW2UAAAAJ" target="_blank">Ping Chen</a><sup>1,2</sup>,
  </span>
  <span class="author-block">
    <a href="https://github.com/stone002" target="_blank">Fuyuan Shi</a><sup>1,2</sup>,
  </span>
  <span class="author-block">
    <a href="https://github.com/5RJ" target="_blank">Ruijia Wu</a><sup>2</sup>,
  </span>
  <span class="author-block">
    <a href="#" target="_blank">Yantao Li</a><sup>1,2,3</sup>,
  </span>
  <span class="author-block">
    <a href="https://github.com/kabutohui" target="_blank">Qiang Hui</a><sup>1,2</sup>,
  </span>
   <br>
  <span class="author-block">
    <a href="https://github.com/markyouyuren" target="_blank">Yuren You</a><sup>2</sup>,
  </span>
  <span class="author-block">
    <a href="#" target="_blank">Ting Lu</a><sup>1,2</sup>,
  </span>
  <span class="author-block">
    <a href="https://github.com/tanchaow" target="_blank">Chao Tan</a><sup>1,2</sup>,
  </span>
  <span class="author-block">
    <a href="https://github.com/zoumaguanxin" target="_blank">Shaoan Zhao</a><sup>1,2</sup>,
  </span>
  <span class="author-block">
    <a href="https://scholar.google.com/citations?hl=en&user=L4OXOs0AAAAJ" target="_blank">Zhaoxiang Liu</a><sup>1,2</sup>
  </span>
  <br>
  <span class="author-block">
    <a href="https://github.com/FangGet" target="_blank">Fang Zhao</a><sup>1,2*</sup>,
  </span>
  <span class="author-block">
    <a href="https://scholar.google.com/citations?user=CFUQLCAAAAAJ&hl=en" target="_blank">Kai Wang</a><sup>1,2</sup>,
  </span>
  <span class="author-block">
    <a href="https://scholar.google.com.hk/citations?user=kCC2oKwAAAAJ&hl=zh-CN&oi=ao" target="_blank">Shiguo Lian</a><sup>1,2*</sup>
  </span>
</div>

<div class="is-size-5 publication-authors" align="center">
  <span class="author-block"><sup>1</sup>Data Science & Artificial Intelligence Research Institute, China Unicom,</span>
   <br>
  <span class="author-block"><sup>2</sup>Unicom Data Intelligence, China Unicom,</span>
   <br>
  <span class="author-block"><sup>3</sup>National Key Laboratory for Novel Software Technology, Nanjing University</span>
</div>

<div class="is-size-5 publication-authors" align="center">
  (* Corresponding author.)
</div>

<h5 align="center">

<a href="https://unicomai.github.io/MeanCache/" target="_blank">
  <img src="https://img.shields.io/badge/Project-Website-blue.svg" alt="Project Page">
</a>
<a href="https://arxiv.org/abs/2601.19961" target="_blank">
  <img src="https://img.shields.io/badge/Paper-PDF-critical.svg?logo=adobeacrobatreader" alt="Paper">
</a>
</a>
<a href="./LICENSE" target="_blank">
  <img src="https://img.shields.io/badge/License-Apache%202.0-yellow.svg" alt="License">
</a>
<a href="https://github.com/UnicomAI/MeanCache/stargazers" target="_blank">
  <img src="https://img.shields.io/github/stars/UnicomAI/MeanCache.svg?style=social" alt="GitHub Stars">
</a>

</h5>


### 🎬 Demo Video
https://github.com/user-attachments/assets/6deadbcf-0a7f-4ecc-96fa-645ca86bba7f



## Introduction

In Flow Matching inference, existing caching methods primarily rely on reusing Instantaneous Velocity or its feature-level proxies. However, we observe that instantaneous velocity often exhibits sharp fluctuations across timesteps. This leads to severe trajectory deviations and cumulative errors, especially as the cache interval increases.
Inspired by MeanFlow, we propose MeanCache. Compared to unstable instantaneous velocity, Average Velocity is significantly smoother and more robust over time. By shifting the caching perspective from a single "point" to an "interval," MeanCache effectively mitigates trajectory drift under high acceleration ratios.


## Latest News
- [2026/02/05] 🚀 Support [**ComfyUI-MeanCache-Z**](https://github.com/facok/comfyui-meancache-z) with dynamic strategy, thanks [**@facok**](https://github.com/facok)!
- [2025/02/04]  Support [**Z-Image**](https://github.com/UnicomAI/MeanCache/tree/main/MeanCache4Z-Image) and released the **MeanCache vs. LeMiCa** comparative study.
- [2025/02/02]  Support [**Qwen-Image**](https://github.com/UnicomAI/MeanCache/tree/main/MeanCache4Qwen-Image) and Inference Code Released !  



##  MeanCache vs. LeMiCa 

This benchmark evaluates the performance of **MeanCache** against **LeMiCa** using the **Qwen-Image-2512** model as the base.

### 🚀 Efficiency 
**Baseline Latency (Original Qwen-Image-2512): 32.8s**
| Constraint | Method | Latency | Speedup | Time Reduction |
|:---:|:---|:---:|:---:|:---:|
| **$B=25$** | LeMiCa | 18.83 s | 1.74x | - |
| | **MeanCache** | **17.13 s** | **1.91x** | **9.0%** |
| **$B=17$** | LeMiCa | 14.35 s | 2.29x | - |
| | **MeanCache** | **11.67 s** | **2.81x** | **18.7%** |
| **$B=10$** | LeMiCa | 10.41 s | 3.15x | - |
| | **MeanCache** | **6.95 s** | **4.72x** | **33.2%** |


### 🎨 Quality 

| Constraint | Method | PSNR (↑) | SSIM (↑) | LPIPS (↓) |
|:---:|:---|:---:|:---:|:---:|
| **$B=25$** | LeMiCa | 29.20 | **0.945** | 0.065 |
| | **MeanCache** | **29.46** | 0.944 | **0.057** |
| **$B=17$** | LeMiCa | 24.31 | 0.835 | 0.176 |
| | **MeanCache** | **26.49** | **0.907** | **0.104** |
| **$B=10$** | LeMiCa | 17.80 | 0.637 | 0.368 |
| | **MeanCache** | **19.44** | **0.767** | **0.237** |


##  Demo

### Z-Image 

| Z-Image-base | MeanCache(B=25) | MeanCache(B=20) | MeanCache(B=15) | MeanCache(B=13) |
|:-------:|:-----------:|:-------------:|:-----------:|:-----------:|
| 18.07 s | 9.15 s | 7.36 s | 5.58 s | 4.85 s |
|<img src="https://github.com/user-attachments/assets/6f3b7858-f0e7-41f5-86e2-239a8e281215" width="117" height="208" style="object-fit: cover; display: block;"> | <img src="https://github.com/user-attachments/assets/85a5211d-358a-462f-a669-cce31f3660ce" width="117" height="208" style="object-fit: cover; display: block;"> | <img src="https://github.com/user-attachments/assets/146f4070-2a44-4635-b9b9-257e3e157c26" width="117" height="208" style="object-fit: cover; display: block;"> | <img src="https://github.com/user-attachments/assets/51581088-60ee-40ec-ac26-48621c3ab0a7" width="117" height="208" style="object-fit: cover; display: block;"> | <img src="https://github.com/user-attachments/assets/da1a649f-93a6-43de-bd6e-b8386bd0467e" width="117" height="208" style="object-fit: cover; display: block;"> |

### Qwen-Image-2512

| Method   | Qwen-Image-2512 | MeanCache(B=25) | MeanCache(B=17) | MeanCache(B=10) |
|:-------:|:-------:|:-----------:|:-------------:|:-----------:|
| **Latency** | 32.8 s  | 17.13 s      | 11.67 s        | 6.95 s      |
| **T2I** | <img width="160" alt="Qwen-Image-2512" src="https://github.com/user-attachments/assets/9ce355df-c745-47fc-8d85-e91fa32dd071" /> | <img width="160" alt="Meancache_b25" src="https://github.com/user-attachments/assets/11130a7a-f6fe-41e7-b7d0-5de9590930c1"  /> | <img width="160" alt="Meancache_b17" src="https://github.com/user-attachments/assets/ef5fd9ee-bf93-4634-a98b-0d51418b3cf4" /> | <img width="160" alt="Meancache_b10" src="https://github.com/user-attachments/assets/10296ab6-45dc-482b-a92c-445a553c9e52"/> |

### Qwen-Image

| Method   | Qwen-Image | MeanCache(B=25) | MeanCache(B=17) | MeanCache(B=10) |
|:-------:|:-------:|:-----------:|:-------------:|:-----------:|
| **Latency** | 33.13 s  | 17.04 s      | 11.63 s        | 6.92 s      |
| **T2I** | <img width="160" alt="Qwen-Image" src="https://github.com/user-attachments/assets/91e556cf-438f-43e5-a825-5c3a0980df17" /> | <img width="160" alt="Meancache_b25" src="https://github.com/user-attachments/assets/1a5aba9d-650c-4d4e-a588-c1fdd124aa30"  /> | <img width="160" alt="Meancache_b17" src="https://github.com/user-attachments/assets/9c854138-078d-4c19-87f7-e7e5221fac43" /> | <img width="160" alt="Meancache_b10" src="https://github.com/user-attachments/assets/fd1cb552-ec3e-47fb-88ed-9a914e6b0a22"/> |





## License
The majority of this project is released under the **Apache 2.0 license** as found in the [LICENSE](./LICENSE) file.



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

