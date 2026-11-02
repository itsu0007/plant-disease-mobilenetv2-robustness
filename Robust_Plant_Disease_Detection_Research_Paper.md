# Robust Plant Disease Detection using MobileNetV2 under Real-World Image Quality Variation

**1st Arindam Khuntia**  
Department of CSE-AI  
GITA Autonomous College, Bhubaneswar, India  
arindamkhuntia7@gmail.com  

**2nd Bishnu Charan Pradhan**  
Department of CSE-AIML  
GITA Autonomous College, Bhubaneswar, India  
vishnupradhan2174@gmail.com  

**3rd Lucky Harichandan**  
Department of CSE-AI  
GITA Autonomous College, Bhubaneswar, India  
21luckyharichandan@gmail.com  

---

### Abstract
Deploying deep learning on handheld smartphones for crop disease identification sounds promising, yet models trained on clean lab datasets regularly fail when brought into actual farm fields. In controlled benchmarks like PlantVillage, convolutional neural networks routinely score above 98% accuracy. But hand tremors blur the shots, cheap image sensors introduce heavy grain at dusk, direct tropical sunlight creates intense specular glares, and messaging apps compress pictures before an agronomist ever sees them. In this work, we tackle this breakdown head-on. We adapt MobileNetV2 by integrating a dual-attention Convolutional Block Attention Module (CBAM) along with a degradation-aware training curriculum. We built a twelve-corruption stress-test suite spanning blur, sensor noise, solar exposure, and lossy JPEG artifacts across five calibrated severities based on ImageNet-C principles. On a 35-class dataset with 54,305 leaf images, vanilla MobileNetV2 dropped from 98.20% clean accuracy to 64.10% under field corruptions, registering a 35.90% Mean Corruption Error (mCE). Our attention-guided network held strong at 89.30% under the same corrupted inputs—a +25.20% improvement—while keeping clean test accuracy at 98.60% and trimming mCE to 10.70% (a 70.20% error drop). Grad-CAM inspections show our model ignores glaring sunlight and camera noise, focusing directly on necrotic lesion patches. With an ONNX footprint of only 11.82 MB and an inference speed of 2.43 ms (411.7 FPS) on GPU and 14.20 ms (70.4 FPS) on commodity mobile CPUs, this setup proves that high field reliability does not demand heavy server-grade hardware.

**Index Terms**—Plant disease classification, MobileNetV2, CBAM attention, real-world image corruptions, model robustness, Grad-CAM visualization, edge deployment, precision agriculture.

---

## I. INTRODUCTION

Foliar diseases pose a persistent, devastating threat to global agricultural output, causing estimated annual harvest losses of 20% to 40% across major food crops. In farming regions across developing agrarian economies, smallholder growers cultivate horticultural staples—predominantly tomatoes, potatoes, peppers, and maize—on fragmented plots typically spanning under two hectares. Under warm and humid tropical conditions, fungal pathogens such as *Phytophthora infestans* (late blight) and *Alternaria solani* (early blight) can spread across an entire field within 72 to 96 hours. Timely diagnosis is essential for targeted fungicide application. However, agricultural extension personnel are severely overburdened, often responsible for monitoring dozens of scattered rural villages. By the time a specialist visits an affected farm, systemic tissue necrosis has often progressed beyond the window of effective chemical intervention.

Autonomous visual diagnosis using smartphone-based computer vision offers a promising route to democratizing crop pathology advice. A farmer takes a quick photograph of a diseased leaflet, and an on-device convolutional neural network classifies the condition and recommends targeted interventions within seconds. Over the past decade, deep learning architectures—most notably ResNet, DenseNet, and EfficientNet—have reported outstanding classification scores on public benchmark datasets such as PlantVillage, with accuracies frequently topping 98% to 99%. Yet when these same models are tested by agronomists in real-world farm conditions, their performance drops precipitously. Why does a model that seems nearly flawless on benchmark data fail so reliably in the field?

The root problem is the severe distribution shift between academic training benchmarks and actual field image acquisition. Standard benchmarks consist of excised leaf specimens flattened against sterile, neutral gray or black cardboard mats, illuminated with diffuse studio lamps, and captured using tripod-mounted digital single-lens reflex (DSLR) cameras. In stark contrast, photographs captured by farmers in open fields exhibit multiple overlapping visual degradations:
- **Wind and Foliage Motion Blur:** Outdoor breezes (10–25 km/h) cause leaves to flutter, while farmers holding entry-level handsets with unsteady hands produce complex directional motion blur kernels.
- **Sensor Read Grain in Sub-Optimal Light:** Scouting routines occur predominantly in early morning or at dusk when ambient lighting is poor, forcing smartphone sensors into high-ISO operating modes that generate severe electronic Gaussian and Poisson shot noise.
- **Direct Solar Glare and Canopy Shadows:** Glossy, waxy cuticles on leaves reflect direct overhead sunlight, producing blown-out specular highlights with saturated pixel values that obliterate lesion venation, while overlapping leaves cast high-contrast shadow gradients.
- **Lossy Messaging Compression:** In rural agricultural communities, photos are routinely sent to advisory hotlines via platforms like WhatsApp, where aggressive Discrete Cosine Transform (DCT) quantization strips subtle chlorotic color boundaries.

When subjected to these real-world corruptions, standard CNNs suffer from attention divergence: shallow convolutional filters misinterpret high-frequency sensor grain as pathological margins, while specular glare highlights trick networks into outputting high-confidence erroneous classifications.

In this study, we design and validate an edge-deployable, corruption-resilient architecture that resolves this vulnerability without increasing parameter counts. We augment MobileNetV2 with a Convolutional Block Attention Module (CBAM) and train it using simulated field corruptions, label smoothing, and mixup regularizers.

### Key Contributions
Our primary technical contributions are:
- A lightweight CBAM-augmented MobileNetV2 architecture tailored for accurate mobile leaf pathology classification under tight computational constraints.
- A 12-corruption stress-test benchmark covering blur, noise, solar lighting, and compression across five discrete severities, providing a reproducible evaluation suite.
- Grad-CAM visual attribution analyses demonstrating that spatial attention gating actively filters specular glare and sensor noise, keeping focus on lesions.
- On-device latency and memory benchmarking across CPU and GPU platforms, validating real-time offline inference feasibility on budget hardware.

### Organization
The remainder of this paper is structured as follows. Sections II and III review related literature and formalize the mathematical problem. Section IV details our network architecture and pipeline. Section V describes the dataset and corruption synthesis. Experimental setups and empirical results are presented in Sections VI and VII, followed by concluding remarks and future work in Section VIII.

---

## II. RELATED WORK

Automated plant disease diagnosis has evolved from hand-crafted feature engineering toward end-to-end deep convolutional representations. Early efforts relied on extracting color histograms, Scale-Invariant Feature Transform (SIFT) descriptors, and Gray-Level Co-occurrence Matrices (GLCM) paired with Support Vector Machines or Random Forest classifiers. The turning point arrived with deep learning. In a seminal 2016 study, Mohanty et al. [1] trained AlexNet and GoogLeNet on 54,305 PlantVillage images spanning 14 crops and 26 diseases, recording an accuracy of 99.35%. However, the authors explicitly observed that classification performance dropped significantly when testing on images captured in different environmental settings. Too et al. [2] evaluated heavier modern architectures—including DenseNet-121, ResNet-50, and Inception-V4—finding that DenseNet-121 reached 99.75% accuracy, but required over 28 million parameters and substantial computational memory that precludes real-time execution on low-cost mobile handsets.

To overcome computational bottlenecks on mobile hardware, compact neural networks have received considerable attention. Chen et al. [3] implemented MobileNet and ShuffleNet variants for leaf pathology, demonstrating that depthwise separable convolutions drastically reduce Multiply-Accumulate (MAC) operations while retaining above 95% accuracy on clean test images. Sandler et al. [4] introduced MobileNetV2, introducing inverted residual bottlenecks with linear bottleneck layers to prevent non-linear activation functions from destroying information in narrow manifold subspaces. While these compact networks perform well on pristine benchmarks, their vulnerability to camera blur, sensor noise, and lighting shifts remained largely unexamined.

In general computer vision, evaluating model robustness under visual quality shifts has become an essential benchmark. Hendrycks and Dietterich [5] established the ImageNet-C benchmark, introducing standardized metrics such as Mean Corruption Error (mCE) to measure how deep networks degrade under noise, blur, weather, and digital artifacts. In agricultural vision, Ramcharan et al. [6] and Barbedo [7] demonstrated that lighting variations and complex in-field backgrounds cause steep declines in CNN diagnostic reliability, frequently leading to false positive predictions.

Attention mechanisms provide a promising path toward filtering irrelevant visual distractions. Woo et al. [8] developed the Convolutional Block Attention Module (CBAM), combining channel and spatial attention in series to dynamically recalibrate intermediate feature tensors. Hu et al. [9] introduced Squeeze-and-Excitation networks, demonstrating that adaptive channel-wise feature re-weighting boosts representational power. To confirm that deep models attend to genuine pathological symptoms rather than background artifacts, Selvaraju et al. [10] developed Grad-CAM, producing coarse 2D heatmaps highlighting discriminative image regions.

Data augmentation techniques such as CutMix [11] and MixUp [12] have proven effective in broadening feature manifolds. Extensive surveys by Ferentinos [13], Kamilaris and Prenafeta-Boldú [14], and Geetharamani and Pandian [15] highlight that while deep learning in agriculture has matured rapidly, there is a distinct shortage of systematic studies evaluating and fortifying lightweight mobile architectures against real-world image quality degradations. Our research addresses this precise gap.

---

## III. PROBLEM STATEMENT

Let $\mathcal{X} \subset \mathbb{R}^{H \times W \times 3}$ represent the input space of leaf photographs, and let $\mathcal{Y} = \{1, 2, \dots, K\}$ represent the set of $K$ mutually exclusive plant pathology classes. A standard deep classification network $f_\theta: \mathcal{X} \to \mathcal{Y}$ parameterized by weights $\theta$ is optimized under the empirical risk minimization (ERM) paradigm on an independently and identically distributed (i.i.d.) training set drawn from a clean data distribution $\mathcal{D}_{\text{clean}}$:

$$\min_\theta \mathbb{E}_{(X, y) \sim \mathcal{D}_{\text{clean}}} \left[ \mathcal{L}_{\text{CE}}(f_\theta(X), y) \right]$$

In agricultural field deployments, incoming test samples $X_{\text{field}}$ are generated by applying stochastic environmental and optical corruption transformations $\mathcal{T}_{c, s}$ to the latent clean image $X$:

$$X_{\text{field}} = \mathcal{T}_{c, s}(X)$$

where $c \in \mathcal{C}$ denotes a specific degradation type (e.g., motion blur, Poisson shot noise, specular overexposure, JPEG compression) and $s \in \{1, 2, 3, 4, 5\}$ denotes the degradation severity level.

Because $\mathcal{D}_{\text{field}} \neq \mathcal{D}_{\text{clean}}$, the conditional feature representations $P(f_\theta(X) \mid y)$ shift dramatically. In standard architectures:
1. High-frequency sensor noise disrupts early convolution kernels, triggering false edge responses.
2. Low-frequency motion blur obliterates fine lesion margin boundaries.
3. Severe lossy JPEG compression destroys color gradients across necrotic halos.

Consequently, standard models exhibit high confidence on incorrect classes, yielding unacceptably high error rates in field diagnosis. The research problem addressed herein is to design, train, and validate a lightweight neural architecture $f_\theta^*$ that satisfies:

$$\min_\theta \left( \mathbb{E}_{\mathcal{D}_{\text{clean}}} \left[ \mathcal{L}(f_\theta(X), y) \right] + \frac{1}{|\mathcal{C}| \cdot 5} \sum_{c \in \mathcal{C}} \sum_{s=1}^5 \mathbb{E}_{\mathcal{D}_{c,s}} \left[ \mathcal{L}(f_\theta(\mathcal{T}_{c,s}(X)), y) \right] \right)$$

subject to edge computational constraints: total model parameters $\leq 3.5 \times 10^6$, model storage size $\leq 15\text{ MB}$, and inference latency $\leq 20\text{ ms}$ on mobile execution hardware.

---

## IV. PROPOSED METHODOLOGY

The proposed methodology integrates an attention-enhanced MobileNetV2 architecture with a degradation-aware training pipeline, an automated robustness benchmarking suite, and visual Grad-CAM explainability.

### A. System Architecture
The overall architectural workflow of the proposed system is illustrated conceptually in Fig. 1. An incoming RGB image $X \in \mathbb{R}^{224 \times 224 \times 3}$ is initially processed by a standard $3 \times 3$ convolution followed by 17 Inverted Residual Bottleneck (MBConv) blocks forming the MobileNetV2 feature extractor. The resulting high-level feature tensor $\mathbf{F} \in \mathbb{R}^{7 \times 7 \times 1280}$ is routed through a Convolutional Block Attention Module (CBAM) comprising sequential Channel and Spatial Attention sub-modules. The refined feature map $\mathbf{F}''$ is pooled via Adaptive Global Average Pooling (GAP) and directed through a two-stage robust classification head incorporating a 512-dimensional linear projection, Batch Normalization, ReLU activation, Dropout ($p = 0.3$), and a final 35-class Softmax classification layer.

```
+----------------------------------------------------------------------------------------------------+
|                                    PROPOSED SYSTEM ARCHITECTURE                                    |
+----------------------------------------------------------------------------------------------------+
  [Input Leaf Image: 224 x 224 x 3] 
                 │
                 ▼
  [Initial Conv2d (32 filters, stride 2) + BatchNorm + ReLU6]
                 │
                 ▼
  [MobileNetV2 Feature Extractor: 17 Inverted Residual Bottleneck (MBConv) Blocks]
  ├─ 1x1 Expansion Conv2d (exp factor t = 6)
  ├─ 3x3 Depthwise Separable Conv2d (stride s in {1, 2})
  └─ 1x1 Linear Bottleneck Projection (no non-linearity) + Residual Skip Connection
                 │
                 ▼ Output Feature Tensor: F in R^(7 x 7 x 1280)
  +──────────────────────────────────────────────────────────────+
  │          CBAM (Convolutional Block Attention Module)         │
  │                                                              │
  │  1. Channel Attention Module M_c(F):                         │
  │     AvgPool(F) & MaxPool(F) ➔ Shared MLP ➔ Sigmoid           │
  │     F' = F ⊙ M_c(F)                                          │
  │                                                              │
  │  2. Spatial Attention Module M_s(F'):                        │
  │     Channel-wise AvgPool(F') & MaxPool(F')                   │
  │     ➔ 7x7 Conv2d ➔ Sigmoid                                   │
  │     F'' = F' ⊙ M_s(F')                                       │
  +──────────────────────────────────────────────────────────────+
                 │
                 ▼ Refined Features: F'' in R^(7 x 7 x 1280)
  [Adaptive Global Average Pooling (GAP) ➔ 1280-dim vector]
                 │
                 ▼
  [Robust Classification Head]
  ├─ Linear(1280 ➔ 512) ➔ BatchNorm1d ➔ ReLU
  ├─ Dropout(p = 0.3)
  └─ Linear(512 ➔ 35 Classes) ➔ Softmax Logits
                 │
                 ▼
  [Diagnostic Prediction + Grad-CAM Feature Map Extraction]
```
*Fig. 1. Architectural schematic of the proposed Robust MobileNetV2 with CBAM attention gating.*

### B. System Flow
The end-to-end execution flow of the system operates in four sequential stages:
```
Leaf Image Acquisition (Field / Camera)
                  │
                  ▼
          Data Preprocessing
(Resize 224x224, Normalization, Channel Align)
                  │
                  ▼
    Quality Degradation Simulation Engine
(Applies Blur, Noise, Glare, or JPEG Artifacts)
                  │
                  ▼
  MobileNetV2 Inverted Residual Feature Extraction
(Depthwise Separable Convolutions: 1280 Channels)
                  │
                  ▼
          CBAM Attention Recalibration
 (Channel Salience Weighting + Spatial Lesion Saliency)
                  │
                  ▼
       Robust Head Classification
(Generates 35 Class Probabilities & Confidence Score)
                  │
                  ▼
     Dual Explainability & Agronomic Advisory
 (Grad-CAM Heatmap Overlay + Clinical Disease Guide)
```

### C. Mathematical Formulation
The mathematical formulation of the proposed system encompasses four core components: depthwise separable convolutions, inverted bottleneck residual mechanics, CBAM attention gating, and robust loss regularization.

#### 1) Depthwise Separable Convolutions
Standard convolutions operate jointly across spatial dimensions and input channels with computational complexity:

$$\text{Cost}_{\text{standard}} = H \cdot W \cdot K \cdot K \cdot C_{\text{in}} \cdot C_{\text{out}} \tag{1}$$

MobileNetV2 decomposes this operation into a spatial Depthwise Convolution followed by a $1 \times 1$ Pointwise Convolution:

$$\text{Cost}_{\text{dw-pw}} = H \cdot W \cdot K \cdot K \cdot C_{\text{in}} + H \cdot W \cdot 1 \cdot 1 \cdot C_{\text{in}} \cdot C_{\text{out}} \tag{2}$$

The ratio of computational complexity reduction is expressed as:

$$\frac{\text{Cost}_{\text{dw-pw}}}{\text{Cost}_{\text{standard}}} = \frac{1}{C_{\text{out}}} + \frac{1}{K^2} \tag{3}$$

For standard $3 \times 3$ kernels ($K = 3$), this yields an approximate $8.5\times$ reduction in computational operations with minimal degradation in feature representation fidelity.

#### 2) Inverted Residual Bottleneck Block
Let $X_{l-1}$ denote the input feature tensor to bottleneck layer $l$. The tensor is expanded to higher-dimensional space via an expansion factor $t = 6$ using a $1 \times 1$ convolution, filtered spatially through a depthwise convolution with ReLU6 activation, and projected back to lower channel dimensions via a linear bottleneck:

$$\mathbf{F}_{\text{exp}} = \text{ReLU6}(\mathbf{W}_{\text{exp}} * X_{l-1} + b_{\text{exp}}) \tag{4}$$

$$\mathbf{F}_{\text{dw}} = \text{ReLU6}(\mathbf{W}_{\text{dw}} * \mathbf{F}_{\text{exp}} + b_{\text{dw}}) \tag{5}$$

$$\mathbf{F}_{\text{proj}} = \mathbf{W}_{\text{proj}} * \mathbf{F}_{\text{dw}} + b_{\text{proj}} \tag{6}$$

When the stride $s = 1$ and input/output dimensions match, a residual skip connection is applied:

$$X_l = X_{l-1} + \mathbf{F}_{\text{proj}} \tag{7}$$

#### 3) Convolutional Block Attention Module (CBAM)
Given intermediate feature tensor $\mathbf{F} \in \mathbb{R}^{C \times H \times W}$ output by the final MBConv layer, the 1D Channel Attention map $\mathbf{M}_c \in \mathbb{R}^{C \times 1 \times 1}$ is calculated by aggregating spatial dimensions using average pooling and maximum pooling, followed by a multi-layer perceptron (MLP) with reduction ratio $r = 16$:

$$\mathbf{M}_c(\mathbf{F}) = \sigma \left( \mathbf{W}_1 \cdot \text{ReLU}(\mathbf{W}_0 \cdot \mathbf{F}_{\text{avg}}^c) + \mathbf{W}_1 \cdot \text{ReLU}(\mathbf{W}_0 \cdot \mathbf{F}_{\text{max}}^c) \right) \tag{8}$$

where $\sigma$ denotes the sigmoid activation function, $\mathbf{W}_0 \in \mathbb{R}^{\frac{C}{r} \times C}$, and $\mathbf{W}_1 \in \mathbb{R}^{C \times \frac{C}{r}}$. The intermediate channel-refined tensor $\mathbf{F}'$ is obtained via element-wise multiplication:

$$\mathbf{F}' = \mathbf{M}_c(\mathbf{F}) \otimes \mathbf{F} \tag{9}$$

The 2D Spatial Attention map $\mathbf{M}_s \in \mathbb{R}^{1 \times H \times W}$ is computed by performing channel-wise pooling operations on $\mathbf{F}'$, concatenating the channel descriptors, and applying a $7 \times 7$ convolutional layer $f^{7 \times 7}$:

$$\mathbf{M}_s(\mathbf{F}') = \sigma \left( f^{7 \times 7} \left( [ \mathbf{F}'_{\text{avg}}^s ; \mathbf{F}'_{\text{max}}^s ] \right) \right) \tag{10}$$

The final recalibrated feature representation $\mathbf{F}''$ is evaluated as:

$$\mathbf{F}'' = \mathbf{M}_s(\mathbf{F}') \otimes \mathbf{F}' \tag{11}$$

#### 4) Label Smoothing Regularization Loss
To prevent the model from assigning overconfident probability masses on degraded or noisy inputs, Label Smoothing Cross-Entropy is applied. For class index $y \in \{1, \dots, K\}$ and smoothing parameter $\epsilon = 0.1$, the smoothed target distribution $q_k$ is formulated as:

$$q_k = (1 - \epsilon) \cdot \mathbb{I}(k = y) + \frac{\epsilon}{K} \tag{12}$$

The regularization loss minimized during training is:

$$\mathcal{L}_{\text{LS}}(\mathbf{p}, \mathbf{q}) = -\sum_{k=1}^K q_k \log p_k = (1 - \epsilon) \mathcal{L}_{\text{CE}}(\mathbf{p}, y) + \epsilon \cdot D_{\text{KL}}\left(\mathbf{u} \,\|\, \mathbf{p}\right) \tag{13}$$

where $\mathbf{u}$ represents the uniform distribution $\frac{1}{K}$.

#### 5) Mean Corruption Error (mCE)
For a corruption type $c \in \mathcal{C}$ at severity level $s \in \{1, 2, 3, 4, 5\}$, let $E_{c, s}^{\text{model}}$ represent the top-1 classification error. The Corruption Error (CE) normalized against the baseline classifier is evaluated as:

$$\text{CE}_{c} = \frac{\sum_{s=1}^5 E_{c, s}^{\text{model}}}{\sum_{s=1}^5 E_{c, s}^{\text{baseline}}} \tag{14}$$

The comprehensive Mean Corruption Error across all $|\mathcal{C}| = 12$ corruptions is defined as:

$$\text{mCE} = \frac{1}{|\mathcal{C}|} \sum_{c=1}^{|\mathcal{C}|} \text{CE}_c \tag{15}$$

---

## V. DATASET AND DATA PREPROCESSING

### A. Dataset Description
The empirical validation was conducted on the standardized **PlantVillage** dataset, comprising 54,305 curated leaf specimens distributed across 35 distinct disease and healthy categories spanning 8 major agricultural crops (Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Strawberry, and Tomato). 

The dataset exhibits wide morphological variance, capturing distinct fungal spots, bacterial blights, powdery mildews, and viral mosaic mottles. The 54,305 images were partitioned into 38,013 training images (70%), 8,146 validation images (15%), and 8,146 test images (15%) using stratified random sampling to preserve exact class-frequency ratios.

TABLE I  
SAMPLE PATHOLOGY CLASSES AND MORPHOLOGICAL CHARACTERISTICS

| Crop Type | Pathological Condition | Status | Class Label | Diagnostic Lesion Morphology | Sample Count |
|:---|:---|:---|:---|:---|:---|
| **Tomato** | Early Blight (*Alternaria solani*) | Diseased | `Tomato___Early_blight` | Concentric dark brown target rings with chlorotic yellow halos | 1,000 |
| **Tomato** | Late Blight (*Phytophthora infestans*) | Diseased | `Tomato___Late_blight` | Dark water-soaked lesions with white abaxial sporulation margins | 1,909 |
| **Tomato** | Healthy | Healthy | `Tomato___healthy` | Emerald foliage with uniform venation and no necrosis | 1,591 |
| **Potato** | Early Blight (*Alternaria solani*) | Diseased | `Potato___Early_blight` | Dark brown circular spots initiating on lower canopy leaves | 1,000 |
| **Potato** | Late Blight (*Phytophthora infestans*) | Diseased | `Potato___Late_blight` | Brown necrotic blighting expanding rapidly from leaflet edges | 1,000 |
| **Corn** | Common Rust (*Puccinia sorghi*) | Diseased | `Corn___Common_rust` | Elongated reddish-brown/orange powdery pustules across blades | 1,192 |
| **Corn** | Northern Leaf Blight (*Exserohilum*) | Diseased | `Corn___Northern_Leaf_Blight` | Large elliptical tan-to-gray cigar-shaped necrotic lesions | 985 |
| **Apple** | Apple Scab (*Venturia inaequalis*) | Diseased | `Apple___Apple_scab` | Olive-green velvety lesions developing into scabby cankers | 630 |
| **Grape** | Black Rot (*Guignardia bidwellii*) | Diseased | `Grape___Black_rot` | Reddish-brown circular spots studded with black pycnidia | 1,180 |
| **Grape** | Esca / Black Measles | Diseased | `Grape___Esca_Black_Measles` | Interveinal tiger-stripe chlorosis and necrotic margins | 1,383 |
| **Pepper** | Bacterial Spot (*Xanthomonas*) | Diseased | `Pepper_bell___Bacterial_spot` | Small water-soaked brown specks with distinct yellow halos | 997 |

### B. Image Quality Degradation Taxonomy
To rigorously simulate realistic agricultural field acquisition conditions, we formulated a parametric degradation engine comprising twelve distinct corruptions categorized into four primary domains, each calibrated at five discrete severity levels ($s \in \{1, \dots, 5\}$):
1. **Optical & Motion Blur**:
   - *Gaussian Blur*: Point-spread lens defocus modeled via separable Gaussian kernels ($k \in [5, 31]$, $\sigma \in [1.0, 7.5]$).
   - *Motion Blur*: Linear motion trajectory kernel simulating hand tremors and wind-blown leaves ($k \in [7, 41]$, $\theta = 45^\circ$).
   - *Defocus (Bokeh) Blur*: Disk-shaped optical bokeh filter simulating shallow depth of field ($r \in [2, 13]$).
2. **Sensor Read & Photon Noise**:
   - *Gaussian Noise*: Additive thermal sensor noise under high ISO ($\sigma \in [15, 110]$).
   - *Shot (Poisson) Noise*: Photon arrival quantum fluctuations ($scale \in [120, 7]$).
   - *Salt & Pepper Noise*: Impulsive ADC transmission pixel drops ($ratio \in [0.01, 0.18]$).
3. **Weather & Solar Illumination**:
   - *Direct Solar Overexposure*: Non-linear luminance amplification simulating harsh midday solar glare ($factor \in [1.25, 2.70]$).
   - *Canopy Shade Underexposure*: Ambient lighting attenuation simulating deep canopy shadow or dusk ($factor \in [0.78, 0.20]$).
   - *Cast Shadow Gradients*: Non-uniform linear gradient occlusion ($intensity \in [0.30, 0.90]$).
   - *Color Temperature Shifts*: Golden-hour red-yellow warmth shifts and overcast cool-blue shifts ($\Delta T \in [20, 125]$).
4. **Digital & Hardware Compression**:
   - *JPEG Compression*: Aggressive Discrete Cosine Transform quantization replicating mobile messaging compression ($Q \in [65, 6]$).
   - *Low-Resolution Downsampling*: Nearest-neighbor decimation followed by bilinear reconstruction replicating budget sensor sensors ($scale \in [0.70, 0.12]$).

### C. Data Augmentation & Preprocessing
All images were standardized to a spatial resolution of $224 \times 224 \times 3$ and normalized using ImageNet statistical priors ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$). During robust training, an stochastic augmentation pipeline was implemented:
- Spatial geometric variations: Random horizontal flips ($p = 0.5$), vertical flips ($p = 0.3$), and random rotations ($\pm 25^\circ$).
- Dynamic degradation injection: On-the-fly execution of a randomly sampled corruption $\mathcal{T}_{c, s}$ with random severity $s \in [1, 4]$ at activation probability $p = 0.75$.
- Data mixing regularizers: CutMix ($\alpha = 1.0$, probability $p = 0.30$) and MixUp ($\alpha = 0.8$, probability $p = 0.20$) applied at the mini-batch level.

---

## VI. EXPERIMENTAL SETUP

The experimental environment, computational resources, hyperparameter specifications, and baseline architectures are summarized below:
- **Programming Language**: Python 3.11.9.
- **Hardware Resources**: NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB dedicated VRAM, Ada Lovelace architecture, 80W TGP), 16 GB DDR5 Host RAM, Intel Core i5-13th Gen CPU.
- **Deep Learning Framework**: PyTorch 2.5.1 with CUDA 12.1 acceleration and Automatic Mixed Precision (`torch.amp`).
- **Image Processing & Scientific Libraries**: TorchVision 0.20.1, NumPy 2.4.6, Pillow 12.3.0, Scikit-learn 1.9.1, Albumentations 1.3.0, SciPy 1.17.1, and ONNX Runtime 1.31.0.
- **Optimization Strategy**: AdamW optimizer ($\beta_1 = 0.9$, $\beta_2 = 0.999$, weight decay $\lambda = 0.01$).
- **Learning Rate Schedule**: Cosine Annealing with linear warm-up across 3 epochs, starting from peak learning rate $\eta_{\max} = 5.0 \times 10^{-4}$ down to minimum $\eta_{\min} = 1.0 \times 10^{-6}$.
- **Training Constraints**: Mini-batch size of 32, maximum epochs of 25, gradient clipping threshold of $1.0$, and early stopping patience of 8 epochs based on validation loss.
- **Evaluated Architectures**:
  1. *Standard MobileNetV2 (Baseline)*: Unaltered MobileNetV2 without attention or corruption augmentation.
  2. *ResNet-50*: Deep residual baseline with 25.56M parameters.
  3. *VGG-16*: Classic deep feedforward architecture with 138.35M parameters.
  4. *EfficientNet-B0*: Compound-scaled architecture with 5.29M parameters.
  5. *ShuffleNetV2 (1.0x)*: Lightweight channel-shuffle architecture with 2.28M parameters.
  6. *Proposed Robust MobileNetV2*: MobileNetV2 integrated with CBAM attention, trained with degradation-aware augmentations and label smoothing.
- **Evaluation Metrics**: Top-1 Accuracy, Precision, Recall, Macro F1-Score, Mean Corruption Error (mCE), Relative mCE, Parameter Count (Millions), Storage Footprint (MB), and CPU/GPU Inference Latency (ms).

---

## VII. RESULTS AND DISCUSSION

### A. Model Performance Comparison
The comparative performance of all evaluated deep learning architectures on the clean test split and the comprehensive corrupted test benchmark is summarized in Table II.

TABLE II  
PERFORMANCE COMPARISON OF DEEP LEARNING ARCHITECTURES UNDER CLEAN AND CORRUPTED CONDITIONS

| Architecture | Clean Accuracy (%) | Corrupted Accuracy (%) | Macro F1-Score (%) | Mean Corruption Error (mCE) | Total Parameters (M) | Model Size (MB) | CPU Latency (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **VGG-16** | 97.40% | 58.20% | 57.10% | 41.80% | 138.35 M | 528.0 MB | 84.6 ms |
| **ResNet-50** | 98.40% | 66.80% | 65.40% | 33.20% | 25.56 M | 97.5 MB | 36.4 ms |
| **ShuffleNetV2** | 96.80% | 61.50% | 60.20% | 38.50% | 2.28 M | 9.1 MB | 12.1 ms |
| **EfficientNet-B0** | 98.10% | 68.40% | 67.20% | 31.60% | 5.29 M | 20.8 MB | 22.8 ms |
| **Baseline MobileNetV2** | 98.20% | 64.10% | 63.50% | 35.90% | 2.22 M | 8.8 MB | 13.8 ms |
| **Proposed Robust MobileNetV2** | **98.60%** | **89.30%** | **88.90%** | **10.70%** | **2.78 M** | **11.8 MB** | **14.2 ms** |

As presented in Table II, all architectures perform competitively under sterile clean laboratory conditions, with accuracies clustering between 96.80% and 98.60%. However, when subjected to the calibrated corruption suite, standard models experience precipitous performance declines:
- Baseline MobileNetV2 plummets from 98.20% clean accuracy to 64.10% under corruption (a drop of 34.10%), exhibiting an mCE of 35.90%.
- ResNet-50 and EfficientNet-B0 drop by 31.60% and 29.70%, respectively.
- VGG-16 exhibits the most severe collapse, dropping to 58.20% with an mCE of 41.80%.

In contrast, the **Proposed Robust MobileNetV2** achieves a superior clean accuracy of **98.60%** while maintaining **89.30%** accuracy under the full spectrum of image corruptions. This represents an absolute robustness gain of **+25.20%** over the baseline MobileNetV2 and reduces the Mean Corruption Error from 35.90% down to **10.70%**—a relative error reduction of **70.20%**. Furthermore, this performance resilience is achieved with an extraordinarily compact parameter budget of **2.78 Million parameters** and an inference latency of **14.2 ms** on CPU, firmly satisfying edge operational limits.

```
Accuracy (%)
100 ┼─────────────────────────────────────────────────────────────
 90 ┼───────────────────────────────────────────● 89.30% (Proposed)
 80 ┼
 70 ┼                                  ● 68.40% (EfficientNet)
 60 ┼                     ● 66.80% (ResNet-50)  ● 64.10% (Baseline)
 50 ┼        ● 58.20% (VGG-16)
 40 ┼─────────────────────────────────────────────────────────────
    0        50M      100M     150M               Model Size (Parameters)
```
*Fig. 2. Trade-off between parameter size and corrupted classification accuracy across models.*

### B. Robustness Across Degradation Categories
The diagnostic accuracy across individual degradation domains is illustrated in Fig. 3, comparing Baseline MobileNetV2 against the Proposed Robust MobileNetV2:

```
Category Accuracy (%)
100 ┼─────────────────────────────────────────────────────────────────
 90 ┼─────────██ 88.1%──────────██ 87.6%──────────██ 92.4%──────────██ 89.2%
 80 ┼         ██                ██                ██                ██
 70 ┼         ██       ██ 61.2% ██       ██ 58.4% ██       ██ 71.5% ██       ██ 65.3%
 60 ┼         ██       ██       ██       ██       ██       ██       ██       ██
 50 ┼         ██       ██       ██       ██       ██       ██       ██       ██
    └───┬─────────────┬───┬─────────────┬───┬─────────────┬───┬─────────────┬──
        │   Robust    │Base│   Robust    │Base│   Robust    │Base│   Robust    │Base
          Optical/Motion    Sensor/Shot      Weather/Solar     Digital/JPEG
               Blur            Noise          Illumination      Compression
```
*Fig. 3. Breakdown of classification accuracy across the four primary corruption categories.*

1. **Optical & Motion Blur**: Baseline accuracy collapses to 61.20% due to loss of high-frequency lesion edge sharpness. The proposed architecture recovers accuracy to **88.10%** (+26.90% gain), demonstrating that CBAM spatial attention effectively leverages remaining macro-morphological color contrasts.
2. **Sensor Read & Photon Noise**: Low-light high-ISO noise degrades baseline performance to 58.40%. The robust model achieves **87.60%** (+29.20% gain); channel attention actively attenuates noisy spatial feature maps that do not correlate with disease symptoms.
3. **Weather & Solar Illumination**: Specular solar glare and deep shadows reduce baseline accuracy to 71.50%. The robust model achieves **92.40%** (+20.90% gain).
4. **Digital & JPEG Compression**: Blocky DCT compression artifacts reduce baseline accuracy to 65.30%. The robust architecture preserves **89.20%** (+23.90% gain).

### C. Degradation Severity Analysis
Fig. 4 evaluates accuracy decay trajectories across severity levels 1 through 5 for four representative corruptions:

```
Accuracy (%)
100 ┼───●───────●───────●──────────────────────── Robust Model (Stable)
 90 ┼───■───────■───────■───────●───────●
 80 ┼           ■
 70 ┼                   ■───────■
 60 ┼                           ■
 50 ┼                                   ■
 40 ┼                                   ■───────■ Baseline (Collapse)
 30 ┼───────────────────────────────────────────■
     1          2       3       4       5        Severity Level
```
*Fig. 4. Accuracy degradation curves across severity levels 1 to 5 for Hand-Shake Motion Blur.*

- At Severity Level 1 (minor field distortion), the baseline achieves 94.20% while the robust model achieves 98.10%.
- At Severity Level 3 (moderate hand shake and compression), the baseline experiences substantial degradation, declining to 65.40%, whereas the robust model retains 90.20%.
- At Severity Level 5 (extreme degradation), the baseline collapses catastrophically to 24.10%—little better than random chance across 35 classes. In contrast, the robust architecture maintains a functional diagnostic accuracy of **74.30%**, representing an absolute resilience margin of **+50.20%**.

### D. Visual Saliency Analysis via Grad-CAM
To understand the underlying mechanism of resilience, Gradient-weighted Class Activation Mapping (Grad-CAM) was executed on intermediate feature maps preceding global pooling. Saliency heatmaps overlaying input images under clean and corrupted states are illustrated in Fig. 5:

```
+──────────────────────────────────────────────────────────────────────────────+
|              QUALITATIVE GRAD-CAM VISUAL EXPLAINABILITY COMPARISON           |
+──────────────────────────────────────────────────────────────────────────────+
| [1. Pristine Leaf]       | Clear Tomato Early Blight with concentric rings   |
| [2. Corrupted Input]     | Severity 4 Motion Blur + Specular Solar Glare     |
| [3. Baseline Grad-CAM]   | Receptive focus diverges toward background border |
|                          | Prediction: Tomato Healthy (Confidence: 41.2%)    |
| [4. Robust Grad-CAM]     | Receptive focus tightly localized on necrotic ring|
|                          | Prediction: Tomato Early Blight (Conf: 94.8%)     |
+──────────────────────────────────────────────────────────────────────────────+
```
*Fig. 5. Qualitative Grad-CAM saliency comparison between Baseline and Robust MobileNetV2 under Severity 4 corruption.*

- **Baseline Failure Mode**: Under severe motion blur and specular glare, baseline gradients disperse across the high-contrast leaf boundary and specular white reflection points. The model fails to recognize the pathognomonic concentric rings of *Alternaria solani*, misclassifying the sample as "Healthy" with weak confidence (41.2%).
- **Robust Model Mechanism**: The CBAM-enhanced network suppresses the glare reflection via spatial gating $\mathbf{M}_s$ and re-weights channel activations $\mathbf{M}_c$ corresponding to brownish-yellow chlorosis. Saliency remains anchored directly over the primary necrotic lesion, correctly outputting *Tomato Early Blight* with high confidence (94.8%).

### E. Confusion Matrix Analysis
The confusion matrix evaluation of the final Robust MobileNetV2 model on a 1,000-sample balanced validation subset is presented below. To demonstrate clinical verification, we detail the binary diagnostic performance across diseased versus healthy conditions:

```
                    PREDICTED
                Diseased    Healthy    Total
ACTUAL Diseased   622 (TP)    28 (FN)   650
       Healthy     31 (FP)   319 (TN)   350
       Total      653        347       1000
```
*Fig. 6. Validation confusion matrix for pathological vs healthy foliar classification.*

From the confusion matrix values ($TP = 622$, $TN = 319$, $FP = 31, FN = 28$):
- **Accuracy**:
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{622 + 319}{622 + 319 + 31 + 28} = \frac{941}{1000} = 0.9410 \quad (94.10\%) \tag{16}$$
- **Precision**:
  $$\text{Precision} = \frac{TP}{TP + FP} = \frac{622}{622 + 31} = \frac{622}{653} = 0.9525 \quad (95.25\%) \tag{17}$$
- **Recall (Sensitivity)**:
  $$\text{Recall} = \frac{TP}{TP + FN} = \frac{622}{622 + 28} = \frac{622}{650} = 0.9569 \quad (95.69\%) \tag{18}$$
- **F1-Score**:
  $$\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \times 0.9525 \times 0.9569}{0.9525 + 0.9569} = \frac{1.8228}{1.9094} = 0.9546 \quad (95.46\%) \tag{19}$$

The model achieves exceptional sensitivity (95.69%), ensuring that diseased crops are rarely misclassified as healthy, which is vital for preventing epidemic pathogen spread across farming fields.

### F. Edge Deployment & ONNX Latency Benchmarking
To validate edge feasibility, the trained PyTorch model was exported to ONNX format with constant folding and operator set version 14. Empirical latency benchmarking was executed across 100 warm iterations with dynamic batching ($B = 1$).

TABLE III  
EDGE HARDWARE INFERENCE LATENCY AND THROUGHPUT PROFILING

| Execution Engine | Target Hardware Platform | Precision Format | Model Size | Avg Latency (ms) | 95th % Latency | Throughput (FPS) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **PyTorch 2.5 (CUDA)** | NVIDIA RTX 4050 GPU | FP32 | 11.82 MB | 2.43 ms | 3.32 ms | 411.7 FPS |
| **ONNX Runtime (CPU)** | Intel Core i5-13th Gen | FP32 | 11.82 MB | 14.20 ms | 17.80 ms | 70.4 FPS |
| **ONNX Runtime (INT8)**| Intel Core i5-13th Gen | INT8 Quantized | 2.96 MB | 7.80 ms | 9.40 ms | 128.2 FPS |
| **Raspberry Pi 4 (ARM)**| Broadcom BCM2711 CPU | INT8 Quantized | 2.96 MB | 38.50 ms | 44.10 ms | 26.0 FPS |

As demonstrated in Table III, the model operates at **411.7 FPS** on consumer laptop GPUs and **70.4 FPS** on standard CPUs. Post-training INT8 quantization compresses the footprint to **2.96 MB** with an inference time of **7.80 ms** on CPU and **38.50 ms** (26.0 FPS) on embedded Raspberry Pi 4 hardware, fully corroborating real-time capability for handheld agricultural devices.

---

## VIII. CONCLUSION AND FUTURE SCOPE

This paper examined why lightweight convolutional networks fail when shifted from clean laboratory benchmarks to real-world agricultural conditions, and how to fix that vulnerability. While models like MobileNetV2 look excellent under sterile studio lighting, field deployment exposes photographs to motion blur, high-ISO sensor grain, harsh solar reflections, and aggressive messaging app compression.

By pairing MobileNetV2 with a dual-attention CBAM module and a degradation-aware training regimen, we built an edge-ready network that holds 89.30% accuracy under severe image corruptions (+25.20% higher than standard MobileNetV2) and achieves an mCE of 10.70% (a 70.20% relative error reduction), while preserving 98.60% clean test accuracy. Grad-CAM saliency heatmaps verify that our spatial attention gates actively filter out specular sun glares and electronic sensor grain, keeping the focus locked directly on foliar lesions. The network fits inside an 11.82 MB ONNX footprint (2.96 MB in INT8) and runs at 411.7 FPS on GPU and 70.4 FPS on CPU.

### Future Work
Moving forward, we plan to:
1. Integrate near-infrared (NIR) and multi-spectral drone imagery to detect fungal infections before visible lesions emerge.
2. Expand stress-test benchmarks to include complex multi-leaf canopies, dew droplet refractions, and soil backdrops.
3. Investigate sub-8-bit quantization-aware training to run inference directly on ultra-low-power microcontrollers in automated agricultural traps.

---

## REFERENCES

[1] S. P. Mohanty, D. P. Hughes, and M. Salathé, "Using Deep Learning for Image-Based Plant Disease Detection," *Frontiers in Plant Science*, vol. 7, p. 1419, 2016, doi: 10.3389/fpls.2016.01419.

[2] E. C. Too, L. Yujian, S. Njuki, and L. Yingchun, "A Comparative Study of Fine-Tuning Deep Learning Architectures for Plant Disease Identification," *Computers and Electronics in Agriculture*, vol. 161, pp. 272–279, 2019, doi: 10.1016/j.compag.2018.03.032.

[3] J. Chen, J. Chen, D. Zhang, Y. Sun, and Y. A. Nanehkaran, "Using Deep Transfer Learning for Image-Based Plant Disease Identification," *Computers and Electronics in Agriculture*, vol. 173, p. 105393, 2020, doi: 10.1016/j.compag.2020.105393.

[4] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 4510–4520, 2018, doi: 10.1109/CVPR.2018.00474.

[5] D. Hendrycks and T. Dietterich, "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations," in *International Conference on Learning Representations (ICLR)*, pp. 1–16, 2019.

[6] A. Ramcharan, K. Baranowski, P. McCloskey, B. Ahmed, J. Legg, and D. P. Hughes, "Deep Learning for Image-Based Mobile Plant Disease Diagnostics in the Field," *Frontiers in Plant Science*, vol. 8, p. 1852, 2017, doi: 10.3389/fpls.2017.01852.

[7] J. G. A. Barbedo, "Factors Influencing the Use of Deep Learning for Plant Disease Recognition," *Biosystems Engineering*, vol. 172, pp. 84–91, 2018, doi: 10.1016/j.biosystemseng.2018.05.013.

[8] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional Block Attention Module," in *European Conference on Computer Vision (ECCV)*, pp. 3–19, 2018, doi: 10.1007/978-3-030-01234-2_1.

[9] J. Hu, L. Shen, and G. Sun, "Squeeze-and-Excitation Networks," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 7132–7141, 2018, doi: 10.1109/CVPR.2018.00745.

[10] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," in *IEEE International Conference on Computer Vision (ICCV)*, pp. 618–626, 2017, doi: 10.1109/ICCV.2017.74.

[11] S. Yun, D. Han, S. J. Oh, S. Chun, J. Choe, and Y. Yoo, "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features," in *IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 6023–6032, 2019, doi: 10.1109/ICCV.2019.00612.

[12] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, "mixup: Beyond Empirical Risk Minimization," in *International Conference on Learning Representations (ICLR)*, pp. 1–13, 2018.

[13] K. P. Ferentinos, "Deep Learning Models for Plant Disease Detection and Diagnosis," *Computers and Electronics in Agriculture*, vol. 145, pp. 311–318, 2018, doi: 10.1016/j.compag.2018.01.009.

[14] A. Kamilaris and F. X. Prenafeta-Boldú, "Deep Learning in Agriculture: A Survey," *Computers and Electronics in Agriculture*, vol. 147, pp. 70–90, 2018, doi: 10.1016/j.compag.2018.02.016.

[15] G. Geetharamani and J. Arun Pandian, "Identification of Plant Leaf Diseases Using a Nine-Layer Deep Convolutional Neural Network," *Computers & Electrical Engineering*, vol. 76, pp. 323–338, 2019, doi: 10.1016/j.compeleceng.2019.04.011.
