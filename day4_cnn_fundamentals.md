# Day 4: Computer Vision Fundamentals

## Overview

Computer Vision (CV) is one of the most successful applications of deep learning. Understanding CV is essential for Principal AI Architects because:
- Many enterprise AI systems involve visual data
- CV techniques apply to other domains (time-series as images, graph convolutions)
- CV architectures inspired many modern deep learning innovations
- Interview questions often use CV as concrete examples

---

## 1. Why Computer Vision is Different

### Traditional ML vs Deep Learning for Vision

**Traditional Approach (Pre-2012):**
```
Image → Hand-crafted features → Classifier → Prediction
         (SIFT, HOG, SURF)      (SVM, RF)

Problems:
- Features don't generalize across tasks
- Requires domain expertise
- Limited by feature quality
- Cannot learn hierarchical representations
```

**Deep Learning Approach (Post-2012):**
```
Image → CNN → Learned features → Prediction
              (automatic hierarchy)

Advantages:
- End-to-end learning (no manual feature engineering)
- Hierarchical features (edges → textures → parts → objects)
- Transfer learning (pre-trained features reusable)
- State-of-the-art performance
```

**The 2012 Revolution: AlexNet**
- ImageNet competition: 26% error → 16% error (10% absolute improvement)
- First time deep learning dominated computer vision
- Sparked the deep learning revolution

---

## 2. Convolutional Neural Networks (CNNs)

### Why Convolutions for Images?

**Problem with Fully Connected Networks:**
```
Image: 224×224×3 = 150,528 pixels
First layer: 150,528 × 1000 neurons = 150M parameters
Just for first layer!

Issues:
- Too many parameters (overfitting)
- No spatial structure captured
- Position-dependent (cat in top-left ≠ cat in bottom-right)
- Not scalable
```

**Solution: Convolutions**
```
Key insights:
1. Local connectivity (neurons only see local regions)
2. Parameter sharing (same filter across entire image)
3. Translation invariance (detects features anywhere)

Result:
- 3×3 filter: Only 9 parameters (vs 150M)
- Applied across entire image
- Learns hierarchical features
```

### How Convolutions Work

**Conceptual Understanding:**

**Layer 1 (Low-level features):**
- Filters detect: edges, corners, colors
- Small receptive field (3×3 or 5×5)
- Example: Horizontal edge detector, vertical edge detector

**Layer 2 (Mid-level features):**
- Combines edges to detect: textures, simple shapes
- Larger receptive field (each neuron sees more of image)
- Example: Grid pattern, circular texture

**Layer 3 (High-level features):**
- Combines textures to detect: parts, patterns
- Even larger receptive field
- Example: Eye, wheel, door handle

**Layer 4+ (Object parts):**
- Combines parts to detect: objects
- Very large receptive field (sees significant portion of image)
- Example: Face, car, building

**Hierarchical Learning:**
```
Layer 1: | — (edges)
         ↓
Layer 2: ⊞ (textures from edges)
         ↓
Layer 3: ◉ (parts from textures)
         ↓
Layer 4: 🐱 (objects from parts)
```

### Key CNN Operations

**1. Convolution**
- Slides filter across image
- Computes dot product at each position
- Produces feature map (activation map)
- Learns what patterns to detect

**2. Activation (ReLU)**
- Non-linearity: ReLU(x) = max(0, x)
- Enables learning complex patterns
- Computationally efficient
- Most common: ReLU, Leaky ReLU, GELU

**3. Pooling**
- Downsamples feature maps
- Reduces spatial dimensions
- Provides translation invariance
- Types: Max pooling (takes max), Average pooling

**Purpose of Pooling:**
- Reduces computation (fewer parameters in next layer)
- Increases receptive field (neurons see larger areas)
- Provides robustness (small translations don't change output)

**4. Batch Normalization**
- Normalizes layer inputs (mean=0, std=1)
- Stabilizes training (allows higher learning rates)
- Reduces internal covariate shift
- Acts as regularization

**5. Dropout**
- Randomly drops neurons during training
- Prevents overfitting
- Forces network to learn robust features
- Typical rate: 0.5 (drop 50%)

---

## 3. Evolution of CNN Architectures

### 3.1 LeNet-5 (1998) - The Pioneer

**Architecture:**
```
Input (32×32) 
→ Conv 6 filters 
→ Pool 
→ Conv 16 filters 
→ Pool 
→ FC 120 
→ FC 84 
→ Output 10

Parameters: ~60K
Use case: Handwritten digit recognition (MNIST)
```

**Key Innovations:**
- Introduced convolution + pooling pattern
- Proved CNNs work for visual tasks
- Too small for complex images

**Why Historical:**
- MNIST too simple (compared to modern tasks)
- Limited compute prevented scaling
- But established foundational pattern

---

### 3.2 AlexNet (2012) - The Revolution

**Architecture:**
```
Input (227×227×3)
→ Conv 96 filters (11×11, stride 4)
→ Pool
→ Conv 256 filters (5×5)
→ Pool
→ Conv 384 filters (3×3)
→ Conv 384 filters (3×3)
→ Conv 256 filters (3×3)
→ Pool
→ FC 4096
→ Dropout
→ FC 4096
→ Dropout
→ Output 1000

Parameters: 60M
Depth: 8 layers
```

**Key Innovations:**
1. **ReLU activation** (faster than sigmoid/tanh)
2. **Dropout** (prevents overfitting)
3. **Data augmentation** (random crops, flips)
4. **GPU training** (2 GPUs in parallel)
5. **Local Response Normalization** (later replaced by BatchNorm)

**Impact:**
- Won ImageNet 2012 (16.4% error vs 26% previous)
- Proved deep learning works at scale
- Sparked deep learning revolution

**Why Important:**
- First modern CNN architecture
- Showed that depth + compute = better performance
- Established pattern: Conv layers → FC layers → Softmax

---

### 3.3 VGG (2014) - Simplicity and Depth

**Architecture Philosophy:**
- Use only 3×3 convolutions (simplest reasonable size)
- Stack many layers (16-19 layers deep)
- Double channels when spatial size halves

**VGG-16:**
```
Input (224×224×3)
→ Conv 64 (3×3) × 2    } Block 1
→ Pool
→ Conv 128 (3×3) × 2   } Block 2
→ Pool
→ Conv 256 (3×3) × 3   } Block 3
→ Pool
→ Conv 512 (3×3) × 3   } Block 4
→ Pool
→ Conv 512 (3×3) × 3   } Block 5
→ Pool
→ FC 4096
→ FC 4096
→ Output 1000

Parameters: 138M
Depth: 16-19 layers
```

**Key Insight: Stacking 3×3 Convolutions**
```
Two 3×3 convs = 5×5 receptive field
Three 3×3 convs = 7×7 receptive field

Benefits of stacking 3×3:
- Fewer parameters: (3×3×2 = 18) vs (5×5 = 25)
- More non-linearity: 2 ReLUs vs 1 ReLU
- Deeper network: Better feature learning
```

**Pros:**
- Simple, uniform architecture (easy to understand)
- Excellent feature extractor (great for transfer learning)
- Strong performance

**Cons:**
- Too many parameters (138M)
- Slow to train and deploy
- Memory-intensive

**Legacy:**
- VGG features still used for style transfer, perceptual loss
- Established principle: deeper is better (if trained properly)
- Showed simple architectures can be powerful

---

### 3.4 GoogLeNet / Inception (2014) - Efficiency

**Problem VGG Didn't Solve:**
- Deep networks have many parameters
- Computational cost very high
- How to go deeper without more parameters?

**Solution: Inception Module**

**Key Idea:**
Instead of choosing filter size (1×1, 3×3, 5×5), use all of them in parallel!

**Inception Module:**
```
Input
  ├─→ 1×1 conv ──────────────────────┐
  ├─→ 1×1 conv → 3×3 conv ───────────┤
  ├─→ 1×1 conv → 5×5 conv ───────────┤  → Concatenate → Output
  └─→ 3×3 max pool → 1×1 conv ───────┘

Why 1×1 convolutions before 3×3 and 5×5?
- Dimensionality reduction (fewer channels)
- Reduces computation dramatically
- Called "bottleneck layers"
```

**1×1 Convolution Trick:**
```
Without 1×1:
Input: 256 channels → 3×3 conv (256 filters) → Output: 256 channels
Computation: H × W × 256 × 3 × 3 × 256 = H × W × 589,824

With 1×1:
Input: 256 channels 
→ 1×1 conv (64 filters) → 64 channels
→ 3×3 conv (256 filters) → 256 channels
Computation: H × W × (256×1×1×64 + 64×3×3×256)
           = H × W × (16,384 + 147,456) = H × W × 163,840

Savings: 72% less computation!
```

**Architecture:**
```
22 layers deep
9 Inception modules stacked
Parameters: 6.8M (20× fewer than VGG!)

Notable features:
- No FC layers at end (uses global average pooling)
- Auxiliary classifiers (intermediate supervision)
- Much more efficient than VGG
```

**Innovations:**
1. **Multi-scale feature extraction** (different kernel sizes)
2. **1×1 bottleneck layers** (dimensionality reduction)
3. **Global average pooling** (replaces FC layers)
4. **Auxiliary classifiers** (helps gradient flow in deep networks)

**Impact:**
- Showed depth doesn't require more parameters
- Introduced 1×1 convolutions for efficiency
- Inspired many efficient architectures

---

### 3.5 ResNet (2015) - The Breakthrough

**The Problem: Degradation**

When networks get very deep (>20 layers), training becomes difficult:
- Gradients vanish (or explode)
- Deeper networks perform worse than shallow ones
- Not overfitting (training error also higher!)
- Called "degradation problem"

**The Solution: Residual Connections**

**Key Insight:**
Instead of learning direct mapping H(x), learn residual F(x) = H(x) - x

**Residual Block:**
```
Input x
  │
  ├─────────────────────────┐  (identity skip connection)
  │                         │
  ↓                         │
Conv 3×3                    │
  ↓                         │
BatchNorm                   │
  ↓                         │
ReLU                        │
  ↓                         │
Conv 3×3                    │
  ↓                         │
BatchNorm                   │
  │                         │
  └────────→ Add ←──────────┘
              ↓
            ReLU
              ↓
           Output
```

**Why This Works:**

**1. Gradient Flow:**
```
Without skip: gradient must flow through many layers → vanishing
With skip: gradient has direct path → stronger signal
```

**2. Identity Mapping:**
```
If optimal mapping is close to identity:
- Without skip: Network must learn H(x) ≈ x (hard)
- With skip: Network only learns F(x) ≈ 0 (easy, just set weights to zero)
```

**3. Ensemble Effect:**
```
ResNet can be viewed as ensemble of many shallow networks
Skip connections create exponentially many paths
Network implicitly does ensemble learning
```

**ResNet-50 Architecture:**
```
Input (224×224×3)
→ Conv 7×7, 64 filters, stride 2
→ Pool 3×3, stride 2

→ Residual Block × 3  (64 filters)   } Stage 1
→ Residual Block × 4  (128 filters)  } Stage 2
→ Residual Block × 6  (256 filters)  } Stage 3
→ Residual Block × 3  (512 filters)  } Stage 4

→ Global Average Pool
→ FC 1000 (softmax)

Depths: ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152
Parameters (ResNet-50): 25M
```

**Bottleneck Block (for deeper ResNets):**
```
Input (256 channels)
  │
  ├─────────────────────────────┐
  │                             │
  ↓                             │
1×1 conv, 64 filters (reduce)   │
  ↓                             │
3×3 conv, 64 filters            │
  ↓                             │
1×1 conv, 256 filters (expand)  │
  │                             │
  └──────→ Add ←────────────────┘

Saves computation (bottleneck principle)
```

**Impact:**
- Enabled training of very deep networks (152+ layers)
- Won ImageNet 2015 (3.6% error)
- Residual connections now ubiquitous (used in Transformers, GANs, etc.)
- Most influential CNN architecture

**Why ResNet Dominates:**
- Easy to train (even very deep networks)
- Strong performance across tasks
- Transfer learning works excellently
- Architecture is now standard baseline

---

### 3.6 Modern Architectures (2016+)

**DenseNet (2017):**
- Connect every layer to every other layer (dense connections)
- Extreme feature reuse
- Very parameter efficient
- Can be memory-intensive

**MobileNet (2017):**
- Designed for mobile/edge devices
- Depthwise separable convolutions (split spatial and channel ops)
- 10-20× fewer parameters than ResNet
- Small accuracy drop for massive efficiency gain

**EfficientNet (2019):**
- Compound scaling (balance depth, width, resolution)
- Neural architecture search (NAS)
- State-of-the-art accuracy with fewer parameters
- EfficientNet-B7: 84% ImageNet accuracy

**Vision Transformer (ViT, 2020):**
- Apply Transformers to images (treat image as sequence of patches)
- No convolutions at all!
- Needs massive data (>100M images) to work well
- Challenges CNN supremacy

**ConvNeXt (2022):**
- Modernized ResNet using Transformer-inspired techniques
- Pure convolutions (no attention)
- Competitive with Vision Transformers
- Shows CNNs still relevant

---

## 4. Key Computer Vision Tasks

### 4.1 Image Classification

**Task:** Assign single label to entire image

**Examples:**
- Is this image a cat or dog?
- What digit is handwritten? (MNIST)
- What object is in this image? (ImageNet)

**Output:**
- Class probabilities: [cat: 0.92, dog: 0.08]
- Single prediction per image

**Architecture Pattern:**
```
Input image
→ CNN backbone (ResNet, EfficientNet)
→ Global Average Pool (reduce spatial dims to 1×1)
→ Fully Connected layer
→ Softmax (probabilities)
→ Class prediction
```

**Loss Function:**
- Cross-entropy loss (for classification)

**Metrics:**
- Top-1 accuracy (correct prediction)
- Top-5 accuracy (correct label in top 5 predictions)

**Use Cases:**
- Quality inspection (defect vs no defect)
- Medical imaging (disease present/absent)
- Content moderation (safe/unsafe image)

---

### 4.2 Object Detection

**Task:** Locate and classify multiple objects in image

**Examples:**
- Find all cars, pedestrians, traffic signs in driving scene
- Detect tumors in medical scan
- Count products on shelf

**Output:**
- Bounding boxes: [(x, y, width, height), class, confidence]
- Multiple predictions per image

**Approaches:**

**Two-Stage Detectors (Accurate but Slower):**

**R-CNN Family (R-CNN → Fast R-CNN → Faster R-CNN):**
```
Stage 1: Region Proposal
  - Generate ~2000 candidate boxes (regions of interest)
  - Selective Search or Region Proposal Network (RPN)

Stage 2: Classification
  - For each proposal, extract features (CNN)
  - Classify object (what is it?)
  - Refine bounding box (where exactly?)

Speed: ~5 FPS
Accuracy: High (good for complex scenes)
```

**One-Stage Detectors (Fast but Less Accurate):**

**YOLO (You Only Look Once):**
```
Single pass through network:
  - Divide image into grid (e.g., 13×13)
  - Each grid cell predicts:
    • Bounding boxes (x, y, w, h)
    • Confidence scores
    • Class probabilities
  - Post-processing: Non-max suppression (remove duplicate boxes)

Speed: 45-155 FPS (real-time!)
Accuracy: Good (improved in YOLOv5, YOLOv8)

Versions: YOLOv1 (2015) → YOLOv8 (2023)
```

**SSD (Single Shot Detector):**
```
Multi-scale detection:
  - Predictions at multiple feature map resolutions
  - Small objects: detected in high-resolution layers
  - Large objects: detected in low-resolution layers

Speed: 45-60 FPS
Accuracy: Between Faster R-CNN and YOLO
```

**Architecture Pattern (YOLO example):**
```
Input image (416×416×3)
→ CNN backbone (Darknet-53 or ResNet)
→ Feature Pyramid Network (multi-scale features)
→ Detection heads (at multiple scales)
→ Bounding boxes + classes
→ Non-max suppression
→ Final detections
```

**Loss Function:**
- Localization loss (bounding box coordinates)
- Confidence loss (object presence)
- Classification loss (object class)

**Metrics:**
- mAP (mean Average Precision) - most common
- IoU (Intersection over Union) - box overlap
- Precision-Recall curves

**Challenges:**
- Small objects hard to detect
- Occluded objects
- Dense scenes (many overlapping objects)
- Class imbalance (some objects rare)

**Use Cases:**
- Autonomous driving (detect cars, pedestrians, signs)
- Surveillance (detect people, suspicious objects)
- Retail (count products, detect shelf gaps)
- Agriculture (count crops, detect diseases)

---

### 4.3 Semantic Segmentation

**Task:** Classify every pixel in image

**Examples:**
- Label each pixel as road, car, building, sky, etc.
- Separate foreground from background
- Medical image segmentation (tumor boundaries)

**Output:**
- Segmentation mask (same size as input image)
- Each pixel has class label
- Multiple objects of same class merged

**Architecture: U-Net**
```
Encoder (downsampling):
  Conv → Conv → Pool
  Conv → Conv → Pool
  Conv → Conv → Pool
  Conv → Conv → Pool
            ↓
       Bottleneck
            ↓
Decoder (upsampling):
  Upsample → Conv → Conv ← Skip connection from encoder
  Upsample → Conv → Conv ← Skip connection
  Upsample → Conv → Conv ← Skip connection
  Upsample → Conv → Conv ← Skip connection
            ↓
  Final Conv (pixel-wise classification)
```

**Key Innovation: Skip Connections**
```
Why needed?
- Encoder loses spatial detail (through pooling)
- Decoder needs fine details for precise boundaries
- Skip connections provide high-res features from encoder

Result:
- Sharp, accurate boundaries
- Preserves spatial information
```

**Architecture Pattern:**
```
Input image (256×256×3)
→ Encoder (CNN downsampling)
  - Extract features
  - Reduce spatial resolution
  - Increase channels
→ Bottleneck (lowest resolution, highest channels)
→ Decoder (upsampling)
  - Increase spatial resolution
  - Reduce channels
  - Concatenate with encoder features (skip connections)
→ Final conv (1×1)
→ Segmentation mask (256×256×num_classes)
```

**Loss Function:**
- Pixel-wise cross-entropy
- Dice loss (for medical imaging)
- Focal loss (for class imbalance)

**Metrics:**
- Pixel accuracy (simple but not robust)
- IoU (Intersection over Union) per class
- Mean IoU (average across classes)
- Dice coefficient (medical imaging)

**Variants:**
- **FCN (Fully Convolutional Network):** First end-to-end segmentation
- **SegNet:** Encoder-decoder with pooling indices
- **DeepLab:** Atrous convolution (dilated convolutions)
- **PSPNet:** Pyramid pooling module (multi-scale context)

**Use Cases:**
- Autonomous driving (drivable area, lane detection)
- Medical imaging (tumor segmentation, organ boundaries)
- Satellite imagery (land use classification)
- Photo editing (background removal)

---

### 4.4 Instance Segmentation

**Task:** Detect and segment each individual object

**Difference from Semantic:**
- Semantic: All cars labeled as "car" (merged)
- Instance: Each car labeled separately (car_1, car_2, car_3)

**Examples:**
- Count individual cells in microscopy
- Track individual people in crowd
- Separate overlapping objects

**Output:**
- Multiple masks (one per object instance)
- Each mask has class label
- Bounding boxes included

**Architecture: Mask R-CNN**
```
Based on Faster R-CNN + mask branch

Stage 1: Region Proposal Network (RPN)
  - Propose candidate object boxes

Stage 2: For each proposal:
  - Classification (what is it?)
  - Bounding box regression (where is it?)
  - Mask prediction (pixel-wise segmentation)

Key addition: Mask branch
  - Small FCN (Fully Convolutional Network)
  - Predicts binary mask for object
  - Runs in parallel with box regression
```

**Architecture Pattern:**
```
Input image
→ CNN backbone (ResNet-FPN)
→ RPN (region proposals)
→ RoI Align (extract features for each proposal)
→ Three parallel heads:
  ├─ Classification head (object class)
  ├─ Box regression head (bounding box)
  └─ Mask head (pixel mask)
→ Final predictions: boxes + masks + classes
```

**RoI Align (Important Detail):**
```
Problem: RoI Pooling (used in Faster R-CNN) is not pixel-aligned
  - Causes misalignment when extracting features
  - Results in blurry masks

Solution: RoI Align
  - Bilinear interpolation for precise alignment
  - Preserves spatial correspondence
  - Sharp, accurate masks
```

**Loss Function:**
- Classification loss (cross-entropy)
- Box regression loss (smooth L1)
- Mask loss (binary cross-entropy per pixel)

**Metrics:**
- AP (Average Precision) at different IoU thresholds
- AP50, AP75, AP (small/medium/large objects)

**Use Cases:**
- Cell counting in biology
- Industrial inspection (separate overlapping defects)
- Sports analytics (track individual players)
- Robotics (grasp individual objects)

---

## 5. Transfer Learning in Computer Vision

### Why Transfer Learning?

**Problem:**
- Training from scratch needs millions of images
- Most tasks don't have millions of labeled images
- Training is expensive (days/weeks on GPUs)

**Solution: Transfer Learning**
```
Pre-train on large dataset (ImageNet: 1.4M images)
  → Learn general visual features
  
Fine-tune on small target dataset (1K-100K images)
  → Adapt features to specific task
  
Result: Better performance with less data
```

### What Networks Learn

**Layer-by-layer analysis:**

**Early Layers (Conv1-2):**
- Edge detectors (horizontal, vertical, diagonal)
- Color blobs
- Gabor-like filters
- **Universal across all vision tasks**

**Middle Layers (Conv3-4):**
- Textures (wood grain, fabric, fur)
- Patterns (grids, circles, stripes)
- Simple shapes
- **Somewhat task-specific but still transferable**

**Late Layers (Conv5, FC):**
- Object parts (eyes, wheels, windows)
- Complex patterns
- Task-specific features
- **Very task-specific, may need retraining**

**Key Insight:**
Early layers learn general features, late layers learn specific features.
This enables transfer learning!

### Transfer Learning Strategies

**Strategy 1: Feature Extraction (Frozen Backbone)**
```
Use case: Very small dataset (<1K images), target task similar to source

Approach:
1. Load pre-trained model (ResNet-50 on ImageNet)
2. Freeze all convolutional layers (don't update weights)
3. Remove last fully connected layer
4. Add new FC layer for target task (e.g., 10 classes instead of 1000)
5. Train only the new FC layer

Why:
- Prevents overfitting (most weights frozen)
- Fast training (only update last layer)
- Works when data is scarce

Example:
ImageNet → Flower classification (5 species, 500 images)
```

**Strategy 2: Fine-Tuning (Partial Unfreezing)**
```
Use case: Medium dataset (1K-100K images), related task

Approach:
1. Load pre-trained model
2. Freeze early layers (Conv1-3)
3. Unfreeze late layers (Conv4-5, FC)
4. Add new classification head
5. Train with low learning rate (e.g., 0.001)
   - Low LR prevents destroying pre-trained features
6. Optionally: Gradually unfreeze earlier layers

Why:
- Adapts high-level features to new task
- Preserves low-level features (edges, textures)
- Better than feature extraction for more data

Example:
ImageNet → Medical X-ray classification (20K images)
```

**Strategy 3: Full Fine-Tuning**
```
Use case: Large dataset (>100K images), different task

Approach:
1. Load pre-trained model
2. Unfreeze all layers
3. Add new classification head
4. Train entire network with low learning rate

Why:
- Adapts all features to new domain
- Needed when target domain very different from ImageNet
- Still better than training from scratch (pre-training provides good initialization)

Example:
ImageNet → Satellite imagery classification (500K images)
```

**Strategy 4: Training from Scratch**
```
Use case: Massive dataset (>1M images), very different from ImageNet

Approach:
1. Initialize weights randomly
2. Train entire network

When to use:
- Have huge dataset
- Target domain very different (medical, satellite, microscopy)
- Computational budget allows

Rarely needed: Transfer learning almost always helps
```

### How to Choose Strategy

**Decision Tree:**
```
How much data?
  ├─ <1K images
  │   └─ Feature extraction (freeze all CNN layers)
  │
  ├─ 1K-10K images
  │   ├─ Similar to ImageNet? (cats, dogs, everyday objects)
  │   │   └─ Fine-tune last 1-2 layers
  │   └─ Different from ImageNet? (medical, satellite)
  │       └─ Fine-tune last 3-4 layers
  │
  ├─ 10K-100K images
  │   └─ Fine-tune most layers (freeze only Conv1-2)
  │
  └─ >100K images
      ├─ Similar to ImageNet?
      │   └─ Full fine-tuning
      └─ Very different?
          └─ Consider training from scratch OR full fine-tuning
```

### Best Practices

**1. Always Start with Pre-trained Weights**
- Even if planning to train from scratch
- Pre-training provides better initialization
- Faster convergence

**2. Use Lower Learning Rate**
```
From scratch: LR = 0.1 (common)
Fine-tuning: LR = 0.001 (10-100× smaller)

Why?
- Avoid destroying pre-trained features
- Make small adjustments, not big changes
```

**3. Different Learning Rates per Layer**
```
Discriminative learning rates:
  Early layers: LR = 0.0001 (small changes)
  Middle layers: LR = 0.001
  Late layers: LR = 0.01 (larger changes)
  New head: LR = 0.1 (train freely)

Intuition: Early layers already good, late layers need more adaptation
```

**4. Gradual Unfreezing**
```
Epoch 1-5: Train only new head (frozen backbone)
Epoch 6-10: Unfreeze last 2 layers
Epoch 11-15: Unfreeze last 4 layers
Epoch 16+: Unfreeze all layers

Benefits:
- Stable training
- Prevents catastrophic forgetting
- Better final performance
```

**5. Data Augmentation is Critical**
```
With transfer learning and small data, augmentation is essential:
- Random crops
- Horizontal flips
- Color jittering
- Random rotations
- Mixup / Cutmix

Can effectively 10× your dataset size
```

---

## 6. Data Augmentation

### Why Augmentation?

**Problem: Limited Training Data**
```
Deep networks need lots of data (millions of images)
Most tasks have limited labeled data (thousands)
Result: Overfitting
```

**Solution: Data Augmentation**
```
Create variations of existing images
Network sees "different" images each epoch
Effectively increases dataset size
Prevents overfitting
```

### Common Augmentation Techniques

**Geometric Transformations:**

**1. Random Crop**
```
Original: 256×256 image
Random crop: 224×224 region

Effect:
- Network sees different parts of image
- Learn to recognize objects in different positions
- Most common augmentation
```

**2. Horizontal Flip**
```
Mirror image left-to-right

Use case:
- Good for: Cars, animals, people (symmetrical)
- Bad for: Text, signs with text (asymmetric meaning)

Doubles dataset size (flip or no flip)
```

**3. Vertical Flip**
```
Flip upside-down

Use case:
- Good for: Satellite imagery, microscopy
- Bad for: Natural images (cars don't appear upside down)
```

**4. Random Rotation**
```
Rotate by random angle (e.g., -15° to +15°)

Effect:
- Handles objects at different orientations
- Good for medical imaging, satellite imagery
- Use small angles for natural images (large rotations unrealistic)
```

**5. Random Scale / Zoom**
```
Scale image up/down, then crop to original size

Effect:
- Handles objects at different distances
- Network learns scale invariance
```

**Color Transformations:**

**6. Brightness Adjustment**
```
Multiply pixel values by random factor (0.7 to 1.3)

Effect:
- Handles different lighting conditions
- Robust to exposure variations
```

**7. Contrast Adjustment**
```
Stretch or compress pixel value range

Effect:
- Handles low-contrast images
- Robust to camera differences
```

**8. Saturation / Hue Adjustment**
```
Modify color intensity and hue

Effect:
- Handles color variations
- Different camera color profiles
```

**9. Random Grayscale**
```
Convert to grayscale with some probability (e.g., 10%)

Effect:
- Network learns not to rely only on color
- Robust to color variations
```

**Advanced Augmentations:**

**10. Cutout / Random Erasing**
```
Randomly mask out rectangular regions

Effect:
- Network learns to recognize partial objects
- Prevents reliance on specific image regions
- Improves robustness to occlusion
```

**11. Mixup**
```
Mix two images and their labels:
  new_image = 0.7 × image1 + 0.3 × image2
  new_label = 0.7 × label1 + 0.3 × label2

Effect:
- Smooths decision boundaries
- Prevents overconfident predictions
- Improves generalization
- State-of-the-art technique
```

**12. CutMix**
```
Cut and paste regions between images:
  Take patch from image1, paste into image2
  Mix labels proportionally to patch size

Effect:
- Better than Mixup for localization tasks
- More realistic than Mixup (no blending)
```

**13. AutoAugment / RandAugment**
```
Automatically learn best augmentation policy:
  Search over combinations of augmentations
  Find optimal transformations for specific dataset

Effect:
- Dataset-specific augmentation strategy
- Better than manual selection
- Requires extra computation to search
```

### Augmentation Best Practices

**1. Use Different Augmentations for Different Tasks**
```
Classification:
  - All augmentations applicable
  - Focus on photometric (color, brightness)

Object Detection:
  - Must transform bounding boxes too!
  - Avoid aggressive crops (may cut objects)

Segmentation:
  - Must transform masks too!
  - Elastic deformations useful (medical imaging)
```

**2. Augmentation During Training Only**
```
Training:
  - Apply random augmentations
  - Each epoch sees different variations

Validation/Test:
  - No augmentation (or only center crop)
  - Fair evaluation on original images
```

**3. Augmentation Magnitude Matters**
```
Too weak: Not enough variation, still overfit
Too strong: Unrealistic images, harder to learn
Sweet spot: Challenging but realistic

Example:
  Rotation: ±15° (good), ±90° (too much for cars)
  Brightness: ±30% (good), ±100% (too extreme)
```

**4. Domain-Specific Augmentations**
```
Medical imaging:
  - Elastic deformations (tissue is elastic)
  - Gaussian noise (sensor noise)
  - No horizontal flips (anatomy not symmetric)

Satellite imagery:
  - All rotations valid (no "up" in space)
  - Seasonal color changes
  - Cloud simulation

Autonomous driving:
  - Weather simulation (rain, fog, snow)
  - Different times of day
  - No vertical flips (roads not upside down)
```

---

## 7. Common Computer Vision Interview Questions

### Q1: "Explain why CNNs work better than fully connected networks for images"

**Good Answer:**
"CNNs have three key advantages for images:

**1. Parameter Efficiency:**
Fully connected: 224×224×3 input with 1000 hidden units = 150M parameters just for first layer.
CNN: 3×3 filter with 64 channels = 1,728 parameters, applied across entire image.

**2. Translation Invariance:**
Same filter detects features anywhere in image. A cat detector works whether cat is top-left or bottom-right. Fully connected treats each position independently - learns separate detector for each location.

**3. Hierarchical Features:**
CNNs naturally learn hierarchy: edges → textures → parts → objects. Each layer builds on previous layer. Fully connected networks don't capture spatial relationships.

**Real Impact:**
AlexNet (CNN): 60M parameters, 16% error on ImageNet.
Equivalent fully connected: Billions of parameters, doesn't even train successfully.

CNNs are specifically designed for image structure: local connectivity, parameter sharing, spatial hierarchy."

---

### Q2: "Why do we need skip connections in ResNet?"

**Good Answer:**
"Skip connections solve the degradation problem in deep networks:

**The Problem:**
Networks deeper than 20-30 layers start performing worse than shallow ones, even on training data. This isn't overfitting (which would show better training performance). Gradients vanish as they propagate backwards through many layers.

**How Skip Connections Help:**

**1. Gradient Flow:**
Without skip: gradient must flow through 50+ layers of transformations.
With skip: gradient has direct path backward through identity connections.
Result: Gradients remain strong even in very deep networks.

**2. Easier Optimization:**
If optimal mapping is close to identity (output ≈ input), network just needs to learn small residual adjustments.
Without skip: Must learn H(x) ≈ x (difficult).
With skip: Only learn F(x) ≈ 0 (easy - set weights to zero).

**3. Ensemble Effect:**
Skip connections create exponentially many paths through network. Network implicitly learns ensemble of shallower networks.

**Real Impact:**
Plain network: 34 layers worse than 18 layers (degradation).
ResNet: 152 layers better than 34 layers (no degradation).
ResNet won ImageNet 2015 with 3.6% error (vs 7.3% without skip connections).

Skip connections are now standard: used in U-Net, DenseNet, Transformers, even GANs."

---

### Q3: "How would you approach a computer vision project with only 1,000 labeled images?"

**Good Answer:**
"With limited data, I'd use a comprehensive strategy:

**1. Transfer Learning (Most Important):**
Start with ResNet-50 pre-trained on ImageNet.
Freeze early layers (Conv1-3), only train late layers + new head.
This leverages features learned from 1.4M ImageNet images.

**2. Aggressive Data Augmentation:**
- Geometric: Random crops, horizontal flips, small rotations (±15°)
- Photometric: Brightness (±30%), contrast, saturation
- Advanced: Mixup (mix images and labels), CutMix
- Effectively multiplies dataset size by 10-20×

**3. Regularization:**
- Dropout (0.5) in classification head
- L2 weight decay
- Early stopping (monitor validation loss)

**4. If More Data Needed:**
- Pseudo-labeling: Use model to label unlabeled images, retrain
- Self-supervised pre-training: Train on unlabeled data first (rotation prediction, jigsaw puzzles)
- Data collection: More labeled data has highest ROI

**5. Validation Strategy:**
- Stratified k-fold cross-validation (5 folds)
- Ensures robust performance estimate
- 1,000 images means each fold has only 200 - need careful validation

**Expected Results:**
With this approach, can achieve good performance even with 1K images. I've seen 85-90% accuracy on custom tasks (vs 95% with 100K images).

**Real Example:**
Plant disease detection: Started with 800 images, used ResNet-50 + augmentation, achieved 88% accuracy. Client was satisfied for pilot project."

---

### Q4: "Explain the difference between semantic segmentation and instance segmentation. When would you use each?"

**Good Answer:**
"Both segment images at pixel level, but handle multiple objects differently:

**Semantic Segmentation:**
- Classifies every pixel (road, car, person, sky)
- All objects of same class merged into one mask
- Output: Single mask with class per pixel
- Example: All cars labeled as 'car' (no distinction between individual cars)

**Architecture:** U-Net, DeepLab
**Use When:**
- Don't need to distinguish individual objects
- Interested in scene understanding (what regions are what)
- Examples:
  • Autonomous driving: Drivable area detection (don't need to count individual road pixels)
  • Medical imaging: Tumor region identification (one contiguous region)
  • Photo editing: Background separation (foreground vs background)

**Instance Segmentation:**
- Detects and segments each object separately
- Each object gets unique mask (car_1, car_2, car_3)
- Output: Multiple masks, each with bounding box and class
- Example: Three cars → three separate masks

**Architecture:** Mask R-CNN, YOLACT
**Use When:**
- Need to count or track individual objects
- Objects overlap (must separate them)
- Examples:
  • Cell counting: Biology research (count individual cells in microscope image)
  • Crowd analytics: Track individual people (each person = one instance)
  • Robotics: Grasp individual objects (need to separate overlapping items)
  • Industrial inspection: Count individual defects

**Computational Cost:**
Semantic segmentation: Faster (single forward pass)
Instance segmentation: Slower (detect then segment each object)

**Typical Choice:**
If objects don't overlap or counting not needed → Semantic (simpler, faster)
If need individual object tracking/counting → Instance (more complex, necessary)"

---

### Q5: "How would you design a real-time object detection system for autonomous driving?"

**Good Answer:**
"Real-time driving needs speed (>30 FPS) and accuracy (safety-critical). Here's my design:

**Model Selection:**
Use YOLOv8 or EfficientDet (one-stage detectors).
- Two-stage (Faster R-CNN) too slow (5-10 FPS)
- YOLOv8 achieves 45+ FPS on edge GPUs
- Good accuracy-speed trade-off

**Architecture Decisions:**

**1. Multi-Scale Detection:**
- Small objects (pedestrians far away): Detect in high-res feature maps
- Large objects (nearby cars): Detect in low-res feature maps
- Feature Pyramid Network (FPN) crucial

**2. Backbone:**
- EfficientNet or CSPDarknet (YOLOv8 backbone)
- Balance: Accuracy vs speed vs model size
- Mobile deployment: MobileNet variant

**3. Hardware:**
- NVIDIA Jetson AGX Xavier (30W, edge GPU)
- TensorRT optimization (FP16 inference)
- Model quantization (INT8) if needed for speed

**Data Strategy:**

**1. Training Data:**
- BDD100K (diverse driving dataset: 100K images, 10 classes)
- Augmentation: Weather simulation (rain, fog), day/night variations
- Long-tail scenarios: Edge cases important for safety

**2. Class Balance:**
- Cars common, pedestrians/cyclists less common but more important
- Focal loss to handle class imbalance
- Oversample rare-but-critical classes

**System Architecture:**

**1. Multi-Camera Setup:**
- Front, rear, left, right cameras (360° coverage)
- Run model on each camera independently
- Fuse detections in bird's-eye view

**2. Temporal Consistency:**
- Track objects across frames (SORT or DeepSORT)
- Smooth bounding boxes (Kalman filter)
- Reduce false positives (require detection in N consecutive frames)

**3. Fallback Mechanisms:**
- If GPU overloaded: Process every 2nd frame
- If model fails: Radar/LiDAR backup
- Safe fallback: Slow down vehicle

**Monitoring & Safety:**

**1. Runtime Checks:**
- Latency monitoring (<33ms for 30 FPS)
- Confidence threshold enforcement (>0.7 for critical objects)
- Sanity checks (no floating objects, physics constraints)

**2. Continuous Learning:**
- Log edge cases (model uncertain)
- Periodic model updates with new data
- A/B testing new models in shadow mode

**Performance Targets:**
- Latency: <30ms (33 FPS minimum)
- Accuracy: >95% recall for pedestrians/cyclists (safety-critical)
- mAP: >60% (competitive with state-of-the-art)
- False positive rate: <1% (avoid phantom braking)

**Real Trade-off:**
Safety requires detecting all pedestrians (high recall), even if means more false alarms. Would rather brake unnecessarily than miss pedestrian."

---

## Summary: Key Takeaways

**1. CNNs are Standard for Vision:**
- Convolutional layers learn hierarchical features
- ResNet with skip connections is baseline
- Vision Transformers emerging but CNNs still dominant

**2. Architecture Evolution:**
- LeNet → AlexNet → VGG → GoogLeNet → ResNet → EfficientNet → ViT
- Key trends: Deeper, more efficient, specialized

**3. Transfer Learning is Essential:**
- Pre-train on ImageNet, fine-tune on target task
- Works with small datasets (<10K images)
- Almost always better than training from scratch

**4. Data Augmentation Multiplies Dataset:**
- Geometric + photometric transformations
- Mixup, CutMix for advanced augmentation
- Critical with limited data

**5. Task Determines Architecture:**
- Classification: ResNet + Global Pool + FC
- Detection: YOLO (fast) or Faster R-CNN (accurate)
- Segmentation: U-Net (semantic) or Mask R-CNN (instance)

**6. Real-Time vs Accuracy Trade-off:**
- Real-time: One-stage detectors (YOLO, SSD)
- High accuracy: Two-stage detectors (Faster R-CNN)
- Choose based on application constraints

**For Interviews:**
- Know CNN fundamentals (why convolutions work)
- Explain key architectures (ResNet, YOLO, U-Net)
- Understand transfer learning strategies
- Apply to real scenarios (limited data, real-time, safety-critical)
- Show systems thinking (data, model, deployment, monitoring)

---

**END OF DAY 4**
