# Scene Classification with Transfer Learning using ResNet-18
### Project 2: Deep Learning Applications — Week 5 Individual Lab
**Course:** Introduction to Applied AI | BS 8th Semester | Department of Electronics  
**Author:** Muhammad Huzaifa  
**Instructor:** Dr. Faiz Ahmad  
**Platform:** Kaggle (NVIDIA Tesla T4 GPU) | **Framework:** PyTorch & Torchvision  

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-0.15+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/vision/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Hardware](https://img.shields.io/badge/Accelerator-NVIDIA%20T4%20GPU-76B900?logo=nvidia&logoColor=white)](https://www.nvidia.com)
[![Dataset](https://img.shields.io/badge/Dataset-Intel%20Image%20Classification-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

---

## 📌 Executive Summary (Required README Section)

```markdown
## Project 2, Week 5: Scene Classification with Transfer Learning
- Dataset: Intel Image Classification, 6 classes, 11,929 train / 2,105 val / 3,000 test
- Model: ResNet-18 pretrained on ImageNet
- Feature extraction (3,078 trainable parameters): val accuracy 90.1%
- Fine-tuning layer4 (about 8.4M trainable): val accuracy 93.2%
- Test accuracy (used once): 93.4%; weakest class: glacier (recall 87.0%), confused with mountain
- Most instructive error: A snow-covered rocky ridge labeled as mountain was predicted as glacier with high confidence due to geological co-occurrence.
- Next: my own domain in Week 6: Automated Crop Foliage Disease Diagnosis from Field Photography using PlantVillage dataset.
```

---

## 📑 Table of Contents
1. [Project Overview & Learning Objectives](#1-project-overview--learning-objectives)
2. [Dataset Architecture & Exploratory Data Analysis (Part 1)](#2-dataset-architecture--exploratory-data-analysis-part-1)
3. [Mathematical Foundations of Convolution & Parameter Counts (Part 2)](#3-mathematical-foundations-of-convolution--parameter-counts-part-2)
4. [Data Pipelines, Augmentations & Split Strategy (Part 3)](#4-data-pipelines-augmentations--split-strategy-part-3)
5. [Transfer Learning Phase I: Feature Extraction (Part 4)](#5-transfer-learning-phase-i-feature-extraction-part-4)
6. [Transfer Learning Phase II: Fine-Tuning & Learning Curves (Part 5)](#6-transfer-learning-phase-ii-fine-tuning--learning-curves-part-5)
7. [Single Test-Set Evaluation & In-Depth Error Analysis (Part 6)](#7-single-test-set-evaluation--in-depth-error-analysis-part-6)
8. [Check Your Understanding (Theoretical & Mathematical Solutions)](#8-check-your-understanding-theoretical--mathematical-solutions)
9. [Week 6 Project Proposal](#9-week-6-project-proposal)
10. [LinkedIn Post](#10-linkedin-post)
11. [Repository Setup & How to Run](#11-repository-setup--how-to-run)

---

## 1. Project Overview & Learning Objectives

This project builds an end-to-end deep learning computer vision pipeline for multi-class natural scene classification across 6 environmental categories: **buildings, forest, glacier, mountain, sea, and street**.

### Key Technical Competencies
* **Image Tensor Operations:** Deconstructing raw images into multidimensional numerical arrays ($H \times W \times C$) and reproducing 2D discrete convolutions by hand in PyTorch.
* **Parameter Scaling Proofs:** Quantifying parameter efficiency between 2D convolutions ($1,792$ parameters) and dense fully connected projections ($34,560,512$ parameters).
* **Robust Data Pipelines:** Designing split-specific data loaders with data augmentation exclusively on training images while preserving clean, deterministic pipelines for validation and testing.
* **Two-Stage Transfer Learning:**
  1. *Feature Extraction:* Training only 3,078 parameters (0.0275% of ResNet-18) while keeping the convolutional feature extractor completely frozen.
  2. *Differential Fine-Tuning:* Unfreezing `layer4` (8,396,806 parameters) with differential learning rates ($10^{-4}$ for pretrained weights, $10^{-3}$ for classification head).
* **Strict Evaluation Discipline:** Preserving the 3,000-image test set for a single terminal evaluation run, diagnosing model performance via full precision/recall metrics, confusion matrices, logit inspection, and error galleries.

---

## 2. Dataset Architecture & Exploratory Data Analysis (Part 1)

The dataset used is the [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) collection curated by Puneet Bansal. It consists of 17,034 labeled natural scene images.

### Class Distribution and Partitions

| Scene Category | Training Partition (`seg_train`) | Test Partition (`seg_test`) | Total Samples |
| :--- | :---: | :---: | :---: |
| **buildings** | 2,191 | 437 | 2,628 |
| **forest** | 2,271 | 474 | 2,745 |
| **glacier** | 2,404 | 553 | 2,957 |
| **mountain** | 2,512 | 525 | 3,037 |
| **sea** | 2,274 | 510 | 2,784 |
| **street** | 2,382 | 501 | 2,883 |
| **Total** | **14,034** | **3,000** | **17,034** |

```
Device Verified: cuda (NVIDIA Tesla T4 GPU on Kaggle)
```

### Raw Image Tensor Specifications
* **Spatial Dimensions $(H, W, C)$:** `(150, 150, 3)` — height 150 px, width 150 px, 3 color channels (Red, Green, Blue).
* **Data Type:** `uint8` (8-bit unsigned integer, storing integer values from 0 to 255 per subpixel).
* **Intensity Dynamic Range:** `[0, 255]` (where 0 indicates minimum light intensity and 255 indicates maximum intensity).
* **Top-Left $3 \times 3$ Red Channel Matrix Sample:**
  $$\begin{bmatrix} 8 & 6 & 3 \\ 6 & 5 & 4 \\ 5 & 5 & 5 \end{bmatrix}$$

### Visual Inspection & Pre-Training Hypothesis
* **Most Similar Classes (Eye Inspection):** `glacier` and `mountain` share immense visual overlap (rugged rocky topography, high-altitude snowfields, ice ridges, and atmospheric sky gradients). A secondary visual pair is `buildings` and `street` (concrete walls, vertical perspective lines, and urban geometry).
* **Initial Hypothesis:** The model will produce its greatest rate of confusion between **`glacier`** and **`mountain`** due to geographical co-occurrence (glaciers physically exist on mountains).

---

## 3. Mathematical Foundations of Convolution & Parameter Counts (Part 2)

### 3.1 Hand-Calculated 2D Convolution Verification
A $5 \times 5$ image exhibiting a hard vertical intensity step between columns 2 and 3 was convolved with a $3 \times 3$ vertical edge-detecting kernel:

$$\mathbf{X} = \begin{bmatrix}
0 & 0 & 10 & 10 & 10 \\
0 & 0 & 10 & 10 & 10 \\
0 & 0 & 10 & 10 & 10 \\
0 & 0 & 10 & 10 & 10 \\
0 & 0 & 10 & 10 & 10
\end{bmatrix}, \quad
\mathbf{K} = \begin{bmatrix}
-1 & 0 & 1 \\
-1 & 0 & 1 \\
-1 & 0 & 1
\end{bmatrix}$$

Applying discrete valid cross-correlation:
$$(X * K)(i, j) = \sum_{m=-1}^{1} \sum_{n=-1}^{1} X(i+m, j+n) \cdot K(m, n)$$

Because columns 0 and 1 are 0 and columns 2, 3, 4 are 10:
* **Position $(0, 0)$:** Kernel overlaps columns 0, 1, 2 $\implies 3 \times (-1 \cdot 0 + 0 \cdot 0 + 1 \cdot 10) = 3 \times 10 = \mathbf{30}$.
* **Position $(0, 1)$:** Kernel overlaps columns 1, 2, 3 $\implies 3 \times (-1 \cdot 0 + 0 \cdot 10 + 1 \cdot 10) = 3 \times 10 = \mathbf{30}$.
* **Position $(0, 2)$:** Kernel overlaps columns 2, 3, 4 $\implies 3 \times (-1 \cdot 10 + 0 \cdot 10 + 1 \cdot 10) = 3 \times 0 = \mathbf{0}$.

The resulting PyTorch output verifies the exact theoretical tensor:
$$\mathbf{Y} = \begin{bmatrix}
30.0 & 30.0 & 0.0 \\
30.0 & 30.0 & 0.0 \\
30.0 & 30.0 & 0.0
\end{bmatrix}$$

---

### 3.2 Parameter Scaling Arithmetic: Conv2D vs. Dense Layer

#### 1. Convolutional Layer (`nn.Conv2d(3, 64, kernel_size=3, padding=1)`)
$$\text{Parameters} = (K_h \times K_w \times C_{\text{in}} + 1) \times C_{\text{out}}$$
* Kernel dimensions: $3 \times 3$
* Input channels ($C_{\text{in}}$): $3$
* Weights per filter: $3 \times 3 \times 3 = 27$
* Bias per filter: $1$
* Total weights + bias per filter: $28$
* Output channels ($C_{\text{out}}$): $64$
$$\text{Total Parameters}_{\text{Conv}} = 28 \times 64 = \mathbf{1,792}$$

#### 2. Dense Linear Layer (`nn.Linear(150 * 150 * 3, 512)`)
$$\text{Parameters} = (\text{in\_features} + 1) \times \text{out\_features}$$
* Input dimensionality: $150 \times 150 \times 3 = 67,500$
* Output dimensionality: $512$
$$\text{Total Parameters}_{\text{Dense}} = (67,500 + 1) \times 512 = 67,501 \times 512 = \mathbf{34,560,512}$$

#### Parameter Ratio Comparison
$$\text{Ratio} = \frac{34,560,512}{1,792} \approx \mathbf{19,286\times}$$

> **Key Takeaway:** A single standard dense hidden layer requires over **19,286 times** more parameters than the convolutional layer.

#### Why Weight Sharing Enables Spatial Generalization
Weight sharing enforces translational equivariance: the same set of localized filter coefficients is slid across the entire image grid. Consequently, visual features (e.g., sharp vertical edges, forest canopy textures, rock ridges) learned in one quadrant of the image are recognized anywhere else without allocating distinct parameters to individual pixel coordinates.

---

### 3.3 Output Spatial Dimension Formula

$$\text{Output Size } O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$

Given input tensor $(1, 3, 224, 224)$, kernel $K=3$, padding $P=1$:
1. **Stride $S = 1$:**
   $$O = \left\lfloor \frac{224 - 3 + 2(1)}{1} \right\rfloor + 1 = \frac{223}{1} + 1 = \mathbf{224} \implies (1, 64, 224, 224)$$
2. **Stride $S = 2$:**
   $$O = \left\lfloor \frac{224 - 3 + 2(1)}{2} \right\rfloor + 1 = \left\lfloor 111.5 \right\rfloor + 1 = 111 + 1 = \mathbf{112} \implies (1, 64, 112, 112)$$
3. **$2 \times 2$ Max Pooling (stride=2):**
   $$O = \left\lfloor \frac{224 - 2}{2} \right\rfloor + 1 = 111 + 1 = \mathbf{112} \implies (1, 3, 112, 112)$$

---

## 4. Data Pipelines, Augmentations & Split Strategy (Part 3)

### Split Architecture
The 14,034 images in `seg_train` were split into training and validation sets using a deterministic random permutation seed (`seed=42`), while `seg_test` was kept strictly isolated for final testing.

```
Total Dataset: 17,034 Images
├── seg_train (14,034 images)
│   ├── train_ds (85%): 11,929 images [Augmented View]
│   └── val_ds   (15%):  2,105 images [Clean View]
└── seg_test (3,000 images)
    └── test_ds       :  3,000 images [Clean View, Evaluated ONCE]
```

```
Split Verification: 11,929 train | 2,105 val | 3,000 test
```

### Transformation Pipelines

```python
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Training Pipeline (Stochastic Augmentations)
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# Validation & Test Pipeline (Deterministic Evaluation)
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])
```

### Critical Augmentation Analysis
* **Inadmissible Augmentation for Natural Scenes:** **Random Vertical Flip**
  * *Reasoning:* Environmental scenes are governed by gravity: the sky is always at the top and terrain/water/roads are at the bottom. Vertically inverting an image creates physically impossible visual distributions (e.g., an inverted ocean hanging in the sky), degrading the model's spatial reasoning.
* **Inadmissible Augmentation for Handwritten Digits (MNIST):** **Random Horizontal Flip**
  * *Reasoning:* Digits possess strict horizontal chirality. Horizontally mirroring digits like `3`, `5`, or `7` creates non-existent character glyphs, while mirroring `6` and `9` directly alters the true class label, injecting label corruption into the loss function.

---

## 5. Transfer Learning Phase I: Feature Extraction (Part 4)

### Model Setup
* **Pretrained Base:** ResNet-18 (trained on ImageNet-1k, ~1.28M natural photos).
* **Frozen Backbone:** All convolutional and batch normalization layers frozen:
  $$\forall p \in \text{model.parameters}(), \quad p\text{.requires\_grad} = \text{False}$$
* **New Linear Head:**
  $$\text{model.fc} = \text{nn.Linear}(\text{in\_features}=512, \text{out\_features}=6)$$

```
Trainable Parameters: 3,078 of 11,179,590 total parameters (0.0275%)
Optimizer: Adam (lr = 1e-3)
Loss Function: CrossEntropyLoss
```

### Feature Extraction Training Progression

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Epoch Duration |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Epoch 1** | 0.588 | 80.9% | 0.327 | 89.4% | 47s |
| **Epoch 2** | 0.326 | 89.0% | 0.290 | 89.9% | 35s |
| **Epoch 3** | 0.299 | 89.5% | 0.276 | **90.1%** | 36s |

$$\mathbf{\text{Best Validation Accuracy (Feature Extraction): } 90.1\% \text{ (0.901)}}$$

### Why 0.03% Parameters Achieve 90.1% Accuracy
Because ResNet-18 was pretrained on 1.28 million ImageNet images, its convolutional layers have already learned a hierarchical representation of vision:
1. Early layers extract low-level edges, oriented bars, and color gradients.
2. Intermediate layers assemble edges into textures, repeating patterns, and surfaces.
3. Deep layers synthesize complex category parts (canopy branches, water ripples, masonry facades).

The frozen backbone maps any $224 \times 224$ natural scene into a linearly separable 512-dimensional embedding space. The new classification head needs to learn only 6 linear hyperplanes ($512 \times 6$ weights $+ 6$ biases $= 3,078$ parameters) to separate the scene categories.

---

## 6. Transfer Learning Phase II: Fine-Tuning & Learning Curves (Part 5)

### Selective Unfreezing Strategy
Rather than unfreezing the entire network (which risks catastrophic forgetting in early layers), we unfreeze only `model.layer4` (the final residual block containing 2 BasicBlocks with 4 convolutional layers and downsampling projections).

```
Trainable Parameters: 8,396,806 of 11,179,590 (75.1% of network)
Pretrained Layer4 Learning Rate: 1e-4 (gentle adaptation)
New Head (fc) Learning Rate: 1e-3 (faster task alignment)
```

### Fine-Tuning Training Progression

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Epoch Duration |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Epoch 4** | 0.262 | 90.6% | 0.227 | 92.6% | 36s |
| **Epoch 5** | 0.170 | 93.8% | 0.204 | **93.2%** | 37s |
| **Epoch 6** | 0.124 | 95.5% | 0.218 | 92.8% | 36s |

$$\mathbf{\text{Best Validation Accuracy (Fine-Tuned): } 93.2\% \text{ (0.932)}}$$
$$\mathbf{\text{Net Gain over Feature Extraction: } +3.1\% \text{ absolute improvement}}$$

```
[Best model checkpoint loaded from Epoch 5 with validation loss = 0.204 and val accuracy = 0.932]
```

### Learning Curve Interpretation & Overfitting Diagnosis
* **Validation Accuracy Leap:** Unfreezing `layer4` caused an immediate jump from 90.1% to 92.6% in Epoch 4, peaking at **93.2%** in Epoch 5.
* **Overfitting Diagnostic:** There is **no sign of harmful overfitting**. Overfitting is characterized by training loss continuing to decrease while validation loss increases sharply. Here, validation loss dropped from 0.276 down to 0.204 in Epoch 5 and leveled off at 0.218 in Epoch 6. The `fit()` function saved the exact model weights from Epoch 5 (`best_acc = 0.932`).
* **Why Training Accuracy Can Be LOWER Than Validation Accuracy:**
  1. *Augmentation Gap:* Training images pass through random crops, horizontal flips, and color jitter, making training batches artificially difficult. Validation images are clean and unaugmented.
  2. *Evaluation Mode Normalization:* In validation (`model.eval()`), batch normalization uses running population statistics accumulated across the dataset, yielding more stable feature representations.

---

## 7. Single Test-Set Evaluation & In-Depth Error Analysis (Part 6)

The best saved checkpoint was evaluated **once** on the 3,000 unseen images in `test_ds`.

### Overall Performance

$$\mathbf{\text{Overall Test Accuracy: } 93.4\% \quad (2,802 \text{ / } 3,000 \text{ correct})}$$

### Comprehensive Classification Report

| Class | Precision | Recall | F1-Score | Support (N) |
| :--- | :---: | :---: | :---: | :---: |
| **buildings** | 0.947 | 0.902 | 0.924 | 437 |
| **forest** | **0.994** | **0.985** | **0.989** | 474 |
| **glacier** | 0.920 | **0.870** | 0.894 | 553 |
| **mountain** | 0.884 | 0.914 | 0.899 | 525 |
| **sea** | 0.945 | 0.982 | 0.963 | 510 |
| **street** | 0.923 | 0.954 | 0.938 | 501 |
| **Macro Average** | **0.935** | **0.935** | **0.935** | **3,000** |
| **Weighted Average** | **0.934** | **0.934** | **0.933** | **3,000** |

### Class Performance Breakdown
* **Strongest Category:** **`forest`** achieved near-perfect performance with **98.5% recall** and **99.4% precision** (F1 = 0.989). Its dense green canopy patterns, texture frequencies, and foliage hues are visually distinct from the other 5 categories.
* **Weakest Categories by Recall:**
  1. **`glacier`** (**87.0% recall**): The single lowest recall in the model. 
  2. **`buildings`** (**90.2% recall**): Often confused with urban street scenes.
  3. **`mountain`** (**91.4% recall**): Frequently confused with glaciers.

### Confusion Analysis: Hypothesis Validation
* **Most Confused Pair:** **`glacier` vs. `mountain`**
* **Verification:** The initial visual hypothesis made in Task 1.2 was **confirmed**. Alpine mountains and glaciers share snowfields, jagged rocky ridges, scree slopes, and identical high-altitude lighting conditions.

---

### Inside a Model Prediction (Task 6.3: Softmax Probabilities)
Evaluating test sample 0 (true class: `buildings`):

| Class | Output Logit ($z_i$) | Softmax Probability ($p_i$) |
| :--- | :---: | :---: |
| **buildings** | **+8.01** | **1.000 (100.0%)** |
| **forest** | -9.46 | 0.000 (0.0%) |
| **glacier** | -4.60 | 0.000 (0.0%) |
| **mountain** | -5.04 | 0.000 (0.0%) |
| **sea** | -4.74 | 0.000 (0.0%) |
| **street** | -3.48 | 0.000 (0.0%) |

The model demonstrates strong certainty: the positive logit (+8.01) creates a probability of $\approx 1.000$, while all other classes receive negative logits.

---

### Error Gallery Case Studies (Auditing Misclassifications)

1. **Error Case 1: True = `glacier`, Predicted = `mountain` (Confidence = 0.84)**
   * *Diagnostic:* **Genuinely Ambiguous Scene**. The photo shows a snow-capped rocky peak with an icy slope below it. The physical scene literally contains both a mountain and a glacier. Because both geological entities coexist in the same image, single-label classification cannot capture the full scene.
2. **Error Case 2: True = `street`, Predicted = `buildings` (Confidence = 0.79)**
   * *Diagnostic:* **Crop / Viewpoint Limitation**. The image was photographed looking upward at historic brick building facades from street level, but the asphalt road surface and sidewalk were cropped out of the lower frame. With only building walls and windows visible in the crop, the model categorized the scene as `buildings`.

---

## 8. Check Your Understanding (Theoretical & Mathematical Solutions)

### Question 1: Why does a network without nonlinear activations collapse to a single linear layer?
**Mathematical Proof:**  
Consider a multi-layer network where each layer $l$ performs an affine transformation without an activation function:
$$\mathbf{y}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1$$
Passing $\mathbf{y}_1$ into layer 2:
$$\mathbf{y}_2 = \mathbf{W}_2 \mathbf{y}_1 + \mathbf{b}_2 = \mathbf{W}_2 (\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (\mathbf{W}_2 \mathbf{W}_1)\mathbf{x} + (\mathbf{W}_2 \mathbf{b}_1 + \mathbf{b}_2)$$
By matrix multiplication associativity, we define:
$$\mathbf{W}_{\text{eff}} = \mathbf{W}_2 \mathbf{W}_1, \quad \mathbf{b}_{\text{eff}} = \mathbf{W}_2 \mathbf{b}_1 + \mathbf{b}_2$$
For an arbitrary depth of $L$ layers:
$$\mathbf{y}_L = \left( \prod_{l=L}^{1} \mathbf{W}_l \right) \mathbf{x} + \mathbf{b}_{\text{eff}} = \mathbf{W}_{\text{net}} \mathbf{x} + \mathbf{b}_{\text{net}}$$
Because the composition of linear transformations is strictly another linear transformation, no matter how many layers or parameters are added, the model can only learn linear decision boundaries (hyperplanes) and cannot solve non-linearly separable problems (such as XOR).

---

### Question 2: For logits $[3, 1, 0]$, what probability does softmax give the first class?
**Step-by-step Calculation:**  
Given $\mathbf{z} = [z_1, z_2, z_3] = [3, 1, 0]$:
1. Compute exponentials:
   $$e^{z_1} = e^3 \approx 20.085537$$
   $$e^{z_2} = e^1 \approx 2.718282$$
   $$e^{z_3} = e^0 = 1.000000$$
2. Sum the exponentials:
   $$\sum_{j=1}^{3} e^{z_j} = 20.085537 + 2.718282 + 1.000000 = 23.803819$$
3. Compute softmax probability $p_1$:
   $$p_1 = \frac{e^{z_1}}{\sum_{j=1}^{3} e^{z_j}} = \frac{20.085537}{23.803819} \approx \mathbf{0.84379} \quad (\mathbf{84.38\%})$$

---

### Question 3: How many parameters does a $3 \times 3$ convolution from 64 to 128 channels have?
**Step-by-step Calculation:**  
* Kernel dimensions: $K_h = 3, K_w = 3$
* Input channels ($C_{\text{in}}$): $64$
* Output channels ($C_{\text{out}}$): $128$
* Weights per filter: $K_h \times K_w \times C_{\text{in}} = 3 \times 3 \times 64 = 576$
* Bias per filter: $1$
* Total parameters per filter: $576 + 1 = 577$
* Total across all 128 filters:
  $$\text{Total Parameters} = (3 \times 3 \times 64 + 1) \times 128 = 577 \times 128 = \mathbf{73,856}$$
  *(Comprising 73,728 weight parameters and 128 bias parameters).*

---

### Question 4: An input of $56 \times 56$ goes through a $3 \times 3$ convolution with padding 1 and stride 2. What is the output size?
**Step-by-step Calculation:**  
Using the spatial output dimension formula:
$$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$
Given $W = 56$, $K = 3$, $P = 1$, $S = 2$:
$$O = \left\lfloor \frac{56 - 3 + 2(1)}{2} \right\rfloor + 1 = \left\lfloor \frac{56 - 3 + 2}{2} \right\rfloor + 1 = \left\lfloor \frac{55}{2} \right\rfloor + 1 = 27 + 1 = \mathbf{28}$$
**Output Spatial Dimensions:** $\mathbf{28 \times 28}$ feature map.

---

### Question 5: Why does a residual (skip) connection help gradients reach the early layers?
**Mathematical Proof:**  
In a standard feed-forward network, the output of a block is $\mathbf{y} = \mathcal{F}(\mathbf{x})$. By the chain rule, gradients backpropagate as:
$$\frac{\partial \mathcal{E}}{\partial \mathbf{x}} = \frac{\partial \mathcal{E}}{\partial \mathbf{y}} \frac{\partial \mathcal{F}(\mathbf{x})}{\partial \mathbf{x}}$$
Repeatedly multiplying by Jacobians whose singular values are less than 1 causes exponential gradient decay (vanishing gradients) as depth increases.

In a residual block with an identity shortcut:
$$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$
Differentiating with respect to the input $\mathbf{x}$:
$$\frac{\partial \mathcal{E}}{\partial \mathbf{x}} = \frac{\partial \mathcal{E}}{\partial \mathbf{y}} \left( \frac{\partial \mathcal{F}(\mathbf{x})}{\partial \mathbf{x}} + \mathbf{I} \right) = \frac{\partial \mathcal{E}}{\partial \mathbf{y}} \frac{\partial \mathcal{F}(\mathbf{x})}{\partial \mathbf{x}} + \frac{\partial \mathcal{E}}{\partial \mathbf{y}}$$
The additive identity matrix $\mathbf{I}$ creates an uninterrupted gradient highway. Even if $\frac{\partial \mathcal{F}(\mathbf{x})}{\partial \mathbf{x}}$ decays toward zero, the term $\frac{\partial \mathcal{E}}{\partial \mathbf{y}}$ flows directly back to early layers, preventing vanishing gradients regardless of network depth.

---

### Question 6: When would you train a CNN from scratch instead of using transfer learning?
Training from scratch is preferred in three scenarios:
1. **Extreme Domain Discrepancy:** When input images differ fundamentally from natural ImageNet photography, such as 3D medical volumetric scans (MRI, CT), Synthetic Aperture Radar (SAR), ultrasonic imaging, seismic waveforms, or astronomical radio interferometry.
2. **Massive In-Domain Datasets:** When the target task has hundreds of thousands or millions of labeled images. In this regime, the model can learn domain-optimal feature representations directly from data without inheriting ImageNet inductive biases or risking overfitting.
3. **Non-Standard Input Modalities:** When images have non-3-channel structures (e.g., 12-band multispectral satellite imagery, depth-only LiDAR grids) where altering the first convolutional layer partially invalidates pretrained ImageNet weights.

---

## 9. Week 6 Project Proposal

### Proposal Title: Autonomous Crop Foliage Disease Diagnosis from Field Photography
* **Problem Statement:** Fungal and bacterial crop foliage diseases account for an estimated 20% to 40% loss in global agricultural yields annually. Smallholder farmers often lack immediate access to expert agronomists, resulting in delayed pesticide application or improper chemical interventions that damage soils and reduce harvest output.
* **Target Beneficiaries:** Agricultural extension workers, small-scale farming cooperatives, and autonomous agricultural monitoring systems seeking real-time diagnostic tools.
* **Proposed Dataset:** The [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) (Open Access Academic License), containing over **54,300 labeled images** across **38 class categories** covering healthy and diseased leaves for 14 commercial crop species (including apple, tomato, potato, corn, and grape).
* **Deep Learning Methodology:** We will evaluate deep residual architectures (ResNet-34 / ConvNeXt-Tiny) via transfer learning, investigating the impact of domain-specific color jitter and leaf-scale geometric augmentations, followed by model quantization for edge deployment on low-cost field devices.

---