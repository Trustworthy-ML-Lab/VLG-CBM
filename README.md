# VLG-CBM: Training Concept Bottleneck Models with Vision-Language Guidance

This is the official repository for our paper _VLG-CBM: Training Concept Bottleneck Models with Vision-Language Guidance_.

- **VLG-CBM** provides a novel method to train Concept Bottleneck Models(CBMs) with guidance from both vision and language domain.
- **VLG-CBM** provides concise and accurate concept attribution for the decision made by the model. The following figure compares decision explanation of VLG-CBM with existing methods by listing top-five contributions for their decisions.

<p align="center">
  <img src="assets/decisions.png" width="90%" alt="Decision Explanation">
</p>

## Training pipeline

Following diagram provides an overview of our training pipeline:
<p align="center">
  <img src="assets/VLG-CBM.png" width="90%" alt="VLG-CBM Overview">
</p>

<!-- reduce image size to 90% -->

## Results

**Accuracy(NEC=5):**

| Method/Dataset                             | CIFAR10     | CIFAR100    | CUB200      | Places365   | ImageNet    |
| ------------------------------------------ | ----------- | ----------- | ----------- | ----------- | ----------- |
| Random                                     | 67.55\%     | 29.52\%     | 68.91\%     | 17.57\%     | 41.49\%     |
| [LF-CBM](https://arxiv.org/abs/2304.06129) | 84.05\%     | 56.52\%     | 53.51\%     | 37.65\%     | 60.30\%     |
| [LM4CV](https://arxiv.org/abs/2308.03685)  | 53.72\%     | 14.64\%     | N/A         | N/A         | N/A         |
| [LaBo](https://arxiv.org/abs/2211.11158)   | 78.69\%     | 44.82\%     | N/A         | N/A         | N/A         |
| VLG-CBM **(Ours)**                         | **88.55\%** | **65.73\%** | **75.79\%** | **41.92\%** | **73.15\%** |

(The results for LM4CV and LaBo is not reported for datasets that use non-CLIP backbone, as they only support CLIP image encoder as the backbone.)

**Note**: Our code will be released soon, please stay tuned.

## Sources

- CUB dataset: <https://www.vision.caltech.edu/datasets/cub_200_2011/>

- Sparse final layer training: <https://github.com/MadryLab/glm_saga>

- Explanation bar plots adapted from: <https://github.com/slundberg/shap>

- CLIP: <https://github.com/openai/CLIP>

- Label-free CBM: <https://github.com/Trustworthy-ML-Lab/Label-free-CBM>

- Grounding DINO: <https://github.com/IDEA-Research/GroundingDINO>

## Cite this work

```txt
@article{srivastava&yan2024vlgcbm,
  title={VLG-CBM: Training Concept Bottleneck Models with Vision-Language Guidance},
  author={Srivastava, Divyansh and Yan, Ge and Weng, Tsui-Wei},
  journal={arXiv preprint},
  year={2024}
}
```
