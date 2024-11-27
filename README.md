<div align="center">

<h1> Latent Diffusion Transformer for Talking Video Synthesis </h1>

<a href='https://zhang-haojie.github.io/project-pages/letstalk.html'><img src='https://img.shields.io/badge/Project-Page-green'></a> 
<a href='https://arxiv.org/abs/2411.16748'><img src='https://img.shields.io/badge/Technique-Report-red'></a> 

</div>


## 🚀 Introduction
**TL;DR:** We present LetsTalk, an innotative Diffusion Transformer with tailored fusion schemes for audio-driven portrait animation, achieving excellent portrait consistency and liveliness in the generated animations.

<div align="center">
<img width="800" alt="image" src="assets/teaser.png?raw=true">
</div>

**Abstract:** Portrait image animation using audio has rapidly advanced, enabling the creation of increasingly realistic and expressive animated faces. The challenges of this multimodality-guided video generation task involve fusing various modalities while ensuring consistency in timing and portrait. We further seek to produce vivid talking heads. 
To address these challenges, we present **LetsTalk** (**L**at**E**nt Diffusion **T**ran**S**former for **Talk**ing Video Synthesis), a diffusion transformer that incorporates modular temporal and spatial attention mechanisms to merge multimodality and enhance spatial-temporal consistency. 
To handle multimodal conditions, we first summarize three fusion schemes, ranging from shallow to deep fusion compactness, and thoroughly explore their impact and applicability. Then we propose a suitable solution according to the modality differences of image, audio, and video generation. 
For portrait, we utilize a deep fusion scheme (Symbiotic Fusion) to ensure portrait consistency. For audio, we implement a shallow fusion scheme (Direct Fusion) to achieve audio-animation alignment while preserving diversity. Our extensive experiments demonstrate that our approach generates temporally coherent and realistic videos with enhanced diversity and liveliness.


## 🎁 Overview

<div align="center">
<img width="800" alt="image" src="assets/pipeline.png?raw=true">
</div>

The overview of our method (a) and the illustration of our designed transformer block (b). For better illustration, we omit the timestep encoder and Layer Norm in (b). LetsTalk integrates transformer blocks equipped with both temporal and spatial attention modules, designed to capture intra-frame spatial details and establish temporal correspondence across time steps. After obtaining portrait and audio embeddings, Symbiotic Fusion is used for fusing the portrait embedding and Direct Fusion is for fusing the audio embedding. Notably, we repeat the portrait embedding along the frame axis to make it have the same shape as the noise embedding.

<div align="center">
<img width="800" alt="image" src="assets/schemes.png?raw=true">
</div>

Illustration of three multimodal fusion schemes, our transformer backbone is formed by the left-side blocks.

(a) **Direct Fusion**. Directly feeding condition into each block's cross-attention module;

(b) **Siamese Fusion**. Maintaining a similar transformer and feeding the condition into it, extracting the corresponding features to guide the features in the backbone;

(c) **Symbiotic Fusion**. Concatenating modality with the input at the beginning, then feeding it into the backbone, achieving fusion via the inherent self-attention mechanisms.


## 📆 TODO
- [ ] Release code (coming soon!!!)

<!-- ## Visualization

### English

### Chinese

### Singing

### AI-generated Portraits -->



## 🎫 Citation
If you find this project useful in your research, please consider cite:

```BibTeX
@misc{zhang2024letstalklatentdiffusiontransformer,
      title={LetsTalk: Latent Diffusion Transformer for Talking Video Synthesis}, 
      author={Haojie Zhang and Zhihao Liang and Ruibo Fu and Zhengqi Wen and Xuefei Liu and Chenxing Li and Jianhua Tao and Yaling Liang},
      year={2024},
      eprint={2411.16748},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.16748}, 
}
```