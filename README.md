<div align="center">

<h1> Efficient Long-duration Talking Video Synthesis with Linear Diffusion Transformer under Multimodal Guidance </h1>

####  <p align="center"> [Haojie Zhang](https://zhang-haojie.github.io/), [Zhihao Liang](https://lzhnb.github.io/), Ruibo Fu, Bingyan Liu, Zhengqi Wen, </p>
####  <p align="center"> Xuefei Liu, Jianhua Tao, Yaling Liang</p>

<a href='https://zhang-haojie.github.io/project-pages/letstalk.html'><img src='https://img.shields.io/badge/Project-Page-green'></a> 
<a href='https://arxiv.org/abs/2411.16748'><img src='https://img.shields.io/badge/Technique-Report-red'></a> 

</div>


## 🚀 Introduction
**TL;DR:** We propose LetsTalk, a diffusion transformer for audio-driven portrait animation. By leveraging DC-VAE and linear attention, LetsTalk enables efficient multimodal fusion and consistent portrait generation, while memory bank and noise-regularized training further improve the quality and stability of long-duration videos.

<div align="center">
<img width="800" alt="image" src="assets/teaser.webp?raw=true">
</div>

**Abstract:** Long-duration talking video synthesis faces enduring challenges in achieving high video quality, portrait and temporal consistency, and computational efficiency. As video length increases, issues such as visual degradation, identity inconsistency, temporal incoherence, and error accumulation become increasingly problematic, severely affecting the realism and reliability of the results.
To address these challenges, we present LetsTalk, a diffusion transformer framework equipped with multimodal guidance and a novel memory bank mechanism, explicitly maintaining contextual continuity and enabling robust, high-quality, and efficient generation of long-duration talking videos. In particular, LetsTalk introduces a noise-regularized memory bank to alleviate error accumulation and sampling artifacts during extended video generation. To further improve efficiency and spatiotemporal consistency, LetsTalk employs a deep compression autoencoder and a spatiotemporal-aware transformer with linear attention for effective multimodal fusion. We systematically analyze three fusion schemes and show that combining deep (Symbiotic Fusion) for portrait features and shallow (Direct Fusion) for audio achieves superior visual realism and precise speech-driven motion, while preserving diversity of movements. Extensive experiments demonstrate that LetsTalk establishes new state-of-the-art in generation quality, producing temporally coherent and realistic talking videos with enhanced diversity and liveliness, and maintains remarkable efficiency with 8x fewer parameters than previous approaches.


## 🎁 Overview

<div align="center">
<img width="800" alt="image" src="assets/Pipeline.webp?raw=true">
</div>

Overview of our LetsTalk framework for robust long-duration talking head video generation. Our system combines a deep compression autoencoder to reduce spatial redundancy while preserving temporal features, and transformer blocks with intertwined temporal and spatial attention to effectively capture both intra-frame details and long-range dependencies. 
Portrait and audio embeddings are extracted; Symbiotic Fusion integrates the portrait embedding, and Direct Fusion incorporates the audio embedding, providing effective multimodal guidance for video synthesis. Portrait embeddings are repeated along the temporal axis for consistent conditioning across frames. 
To further support long-sequence generation, a memory bank module is introduced to maintain temporal consistency, while a dedicated noise-regularized training strategy helps align the memory bank usage between training and inference stages, ensuring stable and high-fidelity generation.

<div align="center">
<img width="800" alt="image" src="assets/schemes.png?raw=true">
</div>

Illustration of three multimodal fusion schemes, our transformer backbone is formed by the left-side blocks.

(a) **Direct Fusion**. Directly feeding condition into each block's cross-attention module;

(b) **Siamese Fusion**. Maintaining a similar transformer and feeding the condition into it, extracting the corresponding features to guide the features in the backbone;

(c) **Symbiotic Fusion**. Concatenating modality with the input at the beginning, then feeding it into the backbone, achieving fusion via the inherent self-attention mechanisms.

<!-- 
## 📆 TODO
- [ ] Release code (coming soon!!!) -->

<!-- ## Visualization

### English

### Chinese

### Singing

### AI-generated Portraits -->



## 🎫 Citation
If you find this project useful in your research, please consider the citation:

```BibTeX
@article{zhang2024efficient,
  title={Efficient Long-duration Talking Video Synthesis with Linear Diffusion Transformer under Multimodal Guidance},
  author={Zhang, Haojie and Liang, Zhihao and Fu, Ruibo and Liu, Bingyan and Wen, Zhengqi and Liu, Xuefei and Tao, Jianhua and Liang, Yaling},
  journal={arXiv preprint arXiv:2411.16748},
  year={2024}
}
```
