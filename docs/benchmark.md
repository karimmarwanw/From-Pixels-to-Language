# Benchmark: our CUB-200-2011 classifiers vs published results

Written 1 Nov 2026 (Karim). Every published number below was read from the source itself (table or
sentence given in the "Where" column), not from memory or a summary site. Venues and titles were
checked on 2 Nov 2026 against the publisher pages (CVF open access for ECCV, the BMVA archive for
BMVC) or, for CIN, the arXiv record ("AAAI 2020").

## Our results

Official CUB-200-2011 split: 5,994 train images (5,095 train + 899 val, our stratified split) and
5,794 test images. Each model was run on the test set once: the final ResNet50 on 1 Nov, the
scratch baseline on 2 Nov. Nothing was tuned or selected after either run.

| Model | Pretraining | Input | Training | Val top-1 | **Test top-1** | Test top-5 | Test macro F1 |
|---|---|---|---|---|---|---|---|
| ResNet50, stage 2 final (`grid-10`), W&B `ln4n82r0` | ImageNet-1k (torchvision `IMAGENET1K_V2`) | 224×224 | 10 epochs head only, then 20 epochs fine-tuning (AdamW) | 81.76% | **82.84%** | 95.34% | 82.73% |
| Scratch CNN (VGG-style, 4.8M params), W&B `snt7by8w` | none | 224×224 | 40 epochs | 43.38% | **44.77%** | 75.27% | 44.21% |

The scratch baseline was run on test (2 Nov) so that it is compared with the published from-scratch
results on the same split, as the project spec asks ("same split, same metrics", Section 5.11).
Both models score about 1 point higher on test than on val. That is within the noise of the
899-image val set (one standard error is about ±1.3 points at 82% and ±1.7 points at 44%).

## Published results

All rows use only the class labels for training (no bounding boxes, no part annotations).

| Source | Model | Pretraining | Input | Top-1 | Where |
|---|---|---|---|---|---|
| Gao et al., *Channel Interaction Networks for Fine-Grained Image Categorization*, AAAI 2020 ([arXiv:2003.05235](https://arxiv.org/abs/2003.05235)) | **ResNet-50, plain fine-tuning** | ImageNet | 448×448 | **84.9%** | Table 1 |
| same | ResNet-101, plain fine-tuning | ImageNet | 448×448 | 85.4% | Table 1 |
| same | VGG-19, plain fine-tuning | ImageNet | 448×448 | 80.2% | Table 1 |
| same | CIN (ResNet-50) | ImageNet | 448×448 | 87.5% | Table 2 |
| same | CIN (ResNet-101) | ImageNet | 448×448 | 88.1% | Table 2 |
| Yang et al., *Learning to Navigate for Fine-grained Classification* (NTS-Net), ECCV 2018, pp. 420–435 ([arXiv:1809.00287](https://arxiv.org/abs/1809.00287)) | **ResNet-50**, "implemented in" Li et al., ICCV 2017 | ImageNet | 448×448 | **84.5%** | Table 2 and Sec. 4.3 |
| same | NTS-Net (ResNet-50, K = 4) | ImageNet | 448×448 | 87.5% | Table 2 |
| same, citing Lin et al., ICCV 2015 | Bilinear-CNN (VGG) | ImageNet | (not stated in NTS-Net) | 84.1% | NTS-Net Table 2 (second-hand) |
| MMPretrain / MMClassification model zoo ([ResNet page](https://mmpretrain.readthedocs.io/en/mmcls-0.x/papers/resnet.html), open-source model `resnet50_8xb8_cub`) | ResNet-50 | **ImageNet-21k** ("mill") | 448×448 | 88.45% | "CUB-200-2011" table; split not stated on the page |
| Yu, Wang, Wu, *Mixup Without Hesitation*, 2021 ([arXiv:2101.04342](https://arxiv.org/abs/2101.04342)) | ResNet-18, fine-tuned, 150 / 300 epochs | ImageNet (PyTorch weights) | 224×224 crops | 76.35% / 77.08% | Table 7, "Baseline" rows |
| same | **ResNet-18 from scratch**, 175 / 350 epochs | none | 224×224 crops | **62.86% / 64.31%** | Table 7, "Baseline" rows |
| Branson, Van Horn, Perona, Belongie, *Improved Bird Species Recognition Using Pose Normalized Deep Convolutional Nets*, BMVC 2014 ([doi:10.5244/C.28.87](http://dx.doi.org/10.5244/C.28.87); the arXiv version, [1406.2952](https://arxiv.org/abs/1406.2952), is titled *Bird Species Categorization Using ...* and has the same passage and numbers) | CNN with ~60M parameters, whole image | ImageNet vs none | not stated | 57.0% vs **10.9%** | Sec. 5, "ImageNet pre-training is essential" |

On the split: the Mixup Without Hesitation paper states the official split (5,994 train / 5,794
test images). CIN and NTS-Net report on "CUB-200-2011", the standard benchmark, but the passages we
read do not restate the counts.

## How our results compare, and why

**Transfer model: 82.8% vs 84.5–84.9% for a plain ResNet-50 in the literature.** We are 1.7–2.1
points below the published plain ResNet-50 baselines. The likely reasons, largest first:

1. **Input resolution.** Every published ResNet-50 row uses 448×448 images: 4 times as many pixels
   as our 224×224. In fine-grained recognition the deciding details are small (eye ring, wing bars,
   beak shape, the streaks that separate a Waterthrush from an Ovenbird), and at 224 px they cover
   only a few pixels. We used 224 because it is our project default and it fits our compute:
   ~110 s per epoch on a laptop GPU, and ~4 times that at 448.
2. **Training length and recipe.** CIN trains for 100 epochs with SGD ("Implementation Details";
   the paper states this for its training in general). We train 10 epochs (head only) plus 20 epochs
   (everything), with hyperparameters tuned on val.
3. **Pretraining data.** The open-source MMPretrain model reaches 88.45% partly because it starts
   from ImageNet-21k, a much larger pretraining set with many more classes, instead of ImageNet-1k.
4. **Fewer training images.** We keep 899 of the 5,994 official training images for validation and
   train on the other 5,095 (85%). The published models are trained on the official training set; we
   did not retrain on train + val, so the size of this effect is not measured. With about 30 images
   per species, 15% fewer images probably costs a little accuracy.

**The +3 points of the specialized methods** (NTS-Net 87.5%, CIN 87.5–88.1%) come from extra
architecture built for fine-grained recognition: finding and zooming into informative regions
(NTS-Net), or modelling interactions between feature channels (CIN). Our model is a plain ResNet50
with a linear head, so the plain baselines are the fair comparison.

**Scratch model: 44.8% vs 62.9–64.3% for ResNet-18 from scratch (both on test).** The published from-scratch
ResNet-18 trains 4–9 times longer (175–350 epochs vs our 40). It also uses residual connections and
2.3 times as many parameters (about 11M vs 4.8M). Our training curves show that the scratch CNN's
val accuracy had levelled off by epoch 35, so a much longer schedule, more augmentation, or a residual
architecture would be needed to close the gap.

**Pretraining is the biggest single factor.** Pretraining adds +38 points for us (44.8% → 82.8% on
test; 43.4% → 81.8% on val), +13 points in the ResNet-18 experiments (64.3% → 77.1%), and +46 points in Branson et al.
(10.9% → 57.0%). With about 30 training images per species, a network trained from scratch cannot
learn general visual features and the fine distinctions between birds at the same time.

**A ceiling to keep in mind.** CIN notes that CUB-200-2011 has label noise of about 4.4% (they cite
Van Horn et al.), so no model can be measured as perfectly accurate on this test set.

## What would close the gap (not done, for the report's "future work")

- Fine-tune at 448×448 (about 4 times slower per epoch).
- Train stage 2 longer, using the tuning result that the backbone lr matters most.
- Start from a stronger backbone or larger pretraining (ImageNet-21k).
