<div align="center">

<h1>Multimodal Diffusion Transformer with Memory Bank for Scalable Long-Duration Talking Video Generation</h1>

#### <p align="center">[Haojie Zhang](https://zhang-haojie.github.io/), [Zhihao Liang](https://lzhnb.github.io/), Ruibo Fu, Bingyan Liu, Zhengqi Wen,</p>
#### <p align="center">Xuefei Liu, Jianhua Tao, Yaling Liang</p>

<a href="https://zhang-haojie.github.io/project-pages/letstalk.html"><img src="https://img.shields.io/badge/Project-Page-green"></a>
<a href="https://arxiv.org/abs/2411.16748"><img src="https://img.shields.io/badge/arXiv-2411.16748-red"></a>
<img src="https://img.shields.io/badge/IEEE%20TMM-Accepted-blue">

</div>

## News

- **Accepted by IEEE Transactions on Multimedia (TMM).** The final publication metadata will be updated once the paper is formally available online.
- The README has been updated to reflect the accepted version of the paper, including the revised title, method description, and latest experimental results.

## Introduction

**TL;DR:** We propose **LetsTalk**, a diffusion transformer framework for audio-driven portrait animation. LetsTalk combines a deep compression autoencoder, an efficient spatiotemporal transformer, asymmetric multimodal fusion, and a noise-regularized memory bank to generate realistic, temporally coherent, and scalable long-duration talking videos.

<div align="center">
<img width="800" alt="LetsTalk teaser" src="assets/teaser_tmm.png?raw=true">
</div>

**Abstract:** Long-duration talking video synthesis faces enduring challenges in achieving high video quality, portrait consistency, temporal coherence, and computational efficiency. As video length increases, visual degradation, portrait drift, temporal artifacts, and error accumulation become increasingly problematic. To address these challenges, we present **LetsTalk**, a diffusion transformer framework equipped with multimodal guidance and a novel memory bank mechanism. LetsTalk explicitly maintains contextual continuity and enables robust, high-quality, and efficient generation of long-duration talking videos. In particular, LetsTalk introduces a noise-regularized memory bank to alleviate error accumulation and sampling artifacts during extended video generation. To further improve efficiency and spatiotemporal modeling, LetsTalk employs a deep compression autoencoder and a spatiotemporal-aware transformer with linear attention for effective multimodal fusion. We systematically analyze three fusion schemes and show that combining **Symbiotic Fusion** for portrait features and **Direct Fusion** for audio achieves superior visual realism and precise speech-driven motion while preserving movement diversity. Extensive experiments demonstrate that LetsTalk establishes new state-of-the-art generation quality, produces temporally coherent and realistic talking videos with enhanced diversity and liveliness, and maintains remarkable efficiency with **8x fewer parameters** than previous approaches.

## Overview

<div align="center">
<img width="800" alt="LetsTalk framework overview" src="assets/pipeline_tmm.png?raw=true">
</div>

LetsTalk generates talking videos from a single reference portrait and driving audio. During training, the reference portrait, driving video clip, and memory clip are compressed by a 32x deep compression autoencoder, while the driving audio is encoded into synchronized audio tokens. The denoising backbone is built from stacked diffusion transformer blocks with temporal attention, audio attention, spatial linear attention, and Mix-FFN.

Portrait and audio guidance are injected with an asymmetric fusion strategy:

- **Symbiotic Fusion for portrait features:** visual reference tokens are concatenated with video latent tokens and jointly modeled through shared self-attention, improving identity and portrait consistency without a separate heavy reference branch.
- **Direct Fusion for audio features:** audio tokens are injected through cross-attention, providing speech-driven motion guidance while preserving visual spatial integrity and motion diversity.
- **Noise-regularized memory bank:** historical clean features are stored and reused during long-duration inference. During training, the memory bank is perturbed with controlled diffusion noise and patch-drop augmentation to reduce the training-inference gap and suppress accumulated artifacts.

<div align="center">
<img width="800" alt="Multimodal fusion schemes" src="assets/fusion_schemes_tmm.png?raw=true">
</div>

## Long-Duration Generation

<div align="center">
<img width="800" alt="Long-duration comparison" src="assets/long_duration_tmm.png?raw=true">
</div>

The memory bank allows LetsTalk to maintain contextual continuity across generated clips. In long-duration experiments beyond 4 minutes, LetsTalk maintains stable identity and visual quality while reducing temporal drift and artifact accumulation compared with previous long-duration talking-head methods.

## Experimental Highlights

LetsTalk is evaluated on identity-disjoint splits of HDTF and CelebV-HQ, totaling 24,794 clips after data refinement. The accepted version reports two model variants:

- **LetsTalk-B:** 0.3B parameters, 27 ms/frame latency, and competitive HDTF quality with substantially fewer parameters.
- **LetsTalk-L:** 0.7B parameters, 40 ms/frame latency, and state-of-the-art quality on HDTF and CelebV-HQ.

Key reported results include:

- **HDTF:** LetsTalk-L achieves FID 31.761, FVD 219.119, Sync-C 4.675, and E-FID 1.962.
- **CelebV-HQ:** LetsTalk-L achieves FID 7.610, FVD 33.175, Sync-C 4.385, and Sync-D 8.350.
- **Efficiency:** LetsTalk-B reaches comparable quality to larger diffusion-based baselines with about 8x fewer parameters than Hallo.

## Citation

If you find this project useful in your research, please consider citing:

```BibTeX
@article{zhang2026letstalk,
  title={Multimodal Diffusion Transformer with Memory Bank for Scalable Long-Duration Talking Video Generation},
  author={Zhang, Haojie and Liang, Zhihao and Fu, Ruibo and Liu, Bingyan and Wen, Zhengqi and Liu, Xuefei and Tao, Jianhua and Liang, Yaling},
  journal={IEEE Transactions on Multimedia},
  year={2026},
  note={Accepted, to appear}
}
```

The arXiv version is also available:

```BibTeX
@article{zhang2024letstalk,
  title={Efficient Long-duration Talking Video Synthesis with Linear Diffusion Transformer under Multimodal Guidance},
  author={Zhang, Haojie and Liang, Zhihao and Fu, Ruibo and Liu, Bingyan and Wen, Zhengqi and Liu, Xuefei and Tao, Jianhua and Liang, Yaling},
  journal={arXiv preprint arXiv:2411.16748},
  year={2024}
}
```

## License

This project is released under the Apache License 2.0.
