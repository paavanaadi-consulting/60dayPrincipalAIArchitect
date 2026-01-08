# Day 4: Computer Vision - Interview Questions

## Section 1: CNN Fundamentals

### Basic Concepts (1-3 years experience)

**Q1: Why are Convolutional Neural Networks (CNNs) preferred over standard Multilayer Perceptrons (MLPs) for image tasks?**

**Expected Answer:**
CNNs are preferred for images due to three key properties that MLPs lack:
1.  **Parameter Sharing:** A single filter (kernel) is used across the entire image, detecting the same feature (e.g., a vertical edge) everywhere. This drastically reduces the number of parameters compared to an MLP where every input pixel would need a unique weight for every neuron in the first hidden layer.
2.  **Translation Invariance:** Because the same filter is applied everywhere, the network can detect a feature regardless of its position in the image. A cat in the top-left corner is detected by the same features as a cat in the bottom-right.
3.  **Hierarchical Feature Learning:** CNNs naturally learn a hierarchy of features. Early layers learn simple features like edges and colors. Deeper layers combine these to learn more complex features like textures, patterns, and eventually object parts (like eyes or wheels). MLPs treat the input as a flat vector and struggle to learn these spatial hierarchies.

**Q2: Explain the purpose of the convolution and pooling layers in a CNN.**

**Expected Answer:**
-   **Convolution Layer:** This is the core building block. Its purpose is to detect local features in the input. It works by sliding a small filter (or kernel) over the image and computing a dot product at each position. The output is a "feature map" or "activation map" that highlights where a specific feature (like a horizontal edge) was detected in the image.
-   **Pooling Layer (e.g., Max Pooling):** The main purpose of pooling is to downsample the feature maps, making the representation smaller and more manageable. This has two key benefits:
    1.  **Reduces Computational Load:** By shrinking the spatial dimensions, subsequent layers have fewer parameters and require less computation.
    2.  **Provides Translation Invariance:** By taking the maximum value in a local neighborhood, the network becomes more robust to small shifts and distortions in the feature's position.

**Q3: What is the role of the activation function (like ReLU) in a CNN?**

**Expected Answer:**
The activation function's primary role is to introduce **non-linearity** into the network. A convolution is a linear operation (a dot product). If we stacked only convolution layers, the entire network would just be one large linear function, which couldn't model complex patterns.

-   **ReLU (Rectified Linear Unit)**, defined as `max(0, x)`, is the most common activation. It's computationally very fast and helps mitigate the vanishing gradient problem that plagued older activation functions like sigmoid or tanh. By introducing non-linearity, CNNs can learn much more complex relationships between pixels and the final output.

### Intermediate Questions (3-5 years experience)

**Q4: Explain the concept of a "receptive field" in a CNN. How does it change as you go deeper into the network?**

**Expected Answer:**
The receptive field of a neuron is the specific region of the input image that affects that neuron's activation.

-   In the **first convolutional layer**, the receptive field is simply the size of the filter (e.g., 3x3 or 5x5). Neurons here only "see" a small local patch of the input image.
-   As you go **deeper into the network**, the receptive field of neurons in subsequent layers grows. A neuron in Layer 2 is looking at a feature map from Layer 1. Since each neuron in Layer 1 saw a 3x3 patch of the input, a 3x3 convolution in Layer 2 will have an effective receptive field of 5x5 on the original input.
-   This increasing receptive field is what allows the network to learn a hierarchy. Early layers with small receptive fields learn simple local features (edges), while later layers with large receptive fields can combine these to recognize larger, more complex objects (faces, cars). Pooling layers also significantly increase the receptive field size.

**Q5: What is a 1x1 convolution and why is it useful?**

**Expected Answer:**
A 1x1 convolution is a filter of size 1x1. While it can't detect spatial patterns, it's extremely useful for channel-wise operations. Its main uses are:
1.  **Dimensionality Reduction (Bottleneck Layer):** This is its most important use, popularized by the Inception network. You can use a 1x1 convolution to reduce the number of channels (feature maps) in the data. For example, you can take a 256-channel input and apply 64 1x1 filters to produce a 64-channel output. This is much cheaper computationally than applying a 3x3 or 5x5 convolution directly to the 256-channel input.
2.  **Increasing Non-linearity:** A 1x1 convolution followed by a ReLU activation adds another layer of non-linearity without changing the spatial dimensions.
3.  **Network-in-Network:** It can be seen as a small, fully connected network operating on the channels at each pixel location.

### Advanced Questions (5+ years experience)

**Q6: You're training a very deep CNN (e.g., 50+ layers) and notice that its training error is higher than a shallower version (e.g., 20 layers). What is this problem called, and what is the architectural innovation that solves it? Explain how it works.**

**Expected Answer:**
This is called the **degradation problem**. It's counterintuitive because it's not overfitting (as training error itself is worse). It shows that simply stacking more layers makes the network harder to train effectively, likely due to the vanishing gradient problem making it difficult for optimizers to find a good path.

The architectural innovation that solves this is the **residual connection (or skip connection)**, introduced in the **ResNet** architecture.

**How it works:**
A standard layer tries to learn a direct mapping, `H(x)`. A residual block instead learns a residual function, `F(x) = H(x) - x`. The output of the block is `F(x) + x`.

This is implemented by adding an "identity shortcut" that bypasses one or more layers and adds the original input `x` to the output of the convolutional block.

**Why it solves the problem:**
1.  **Improved Gradient Flow:** The skip connection provides a direct, uninterrupted path for gradients to flow backward during backpropagation. This prevents the gradient signal from vanishing or exploding as it passes through many layers, allowing for the effective training of networks with 100+ layers.
2.  **Easier Identity Mapping:** In the worst-case scenario, if an added layer is not useful, the network can easily learn to make it an identity mapping by driving the weights of the convolutional block to zero. The output will then just be the input from the skip connection (`F(x) = 0`, so `output = x`). Without a skip connection, the layer would have to struggle to learn the identity function itself, which is much harder.

---

## Section 2: Key Architectures & Tasks

### Basic Concepts (1-3 years experience)

**Q7: What is the difference between image classification, object detection, and semantic segmentation?**

**Expected Answer:**
-   **Image Classification:** Assigns a single label to an entire image. The output is one class for the whole image (e.g., "cat").
-   **Object Detection:** Identifies the location and class of multiple objects in an image. The output is a list of bounding boxes, each with a class label and a confidence score (e.g., "cat at [x,y,w,h]", "dog at [x,y,w,h]").
-   **Semantic Segmentation:** Classifies every single pixel in the image. The output is a segmentation mask the same size as the input, where each pixel has a class label (e.g., all pixels belonging to any cat are labeled "cat", all pixels for the sky are labeled "sky"). It doesn't distinguish between different instances of the same class.

**Q8: What is transfer learning and why is it so common in computer vision?**

**Expected Answer:**
Transfer learning is the practice of taking a model that has been pre-trained on a very large dataset (like ImageNet, with over a million images) and adapting it for a new, specific task that usually has a much smaller dataset.

It's extremely common in computer vision because the features learned by the early layers of a CNN are highly generic and reusable.
-   **Early layers** learn universal features like edges, colors, and textures.
-   **Middle layers** learn more complex patterns and parts of objects.
These learned features are useful for almost any vision task. By using a pre-trained model, you get the benefit of all this learned knowledge without needing to train a massive network from scratch, which would require a huge amount of data and computational resources. This allows us to achieve high performance on tasks with as few as a thousand labeled images.

### Intermediate Questions (3-5 years experience)

**Q9: Compare one-stage and two-stage object detectors. Give an example of each and explain their primary trade-offs.**

**Expected Answer:**
The main difference is how they approach the problem of finding and classifying objects.

**Two-Stage Detectors (e.g., Faster R-CNN):**
1.  **Stage 1 (Region Proposal):** First, a Region Proposal Network (RPN) scans the image to generate a set of candidate bounding boxes ("regions of interest") that are likely to contain an object.
2.  **Stage 2 (Classification & Refinement):** For each proposed region, the detector extracts features and performs classification (what is it?) and refines the bounding box coordinates.

**One-Stage Detectors (e.g., YOLO, SSD):**
They perform localization and classification in a single pass. A YOLO model, for example, divides the image into a grid and has each grid cell directly predict bounding boxes and class probabilities for the objects it contains.

**Primary Trade-off:**
-   **Speed vs. Accuracy:** This is the classic trade-off.
    -   **Two-stage detectors** are generally **more accurate**, especially for small or overlapping objects, because they can focus their attention on high-quality proposals. However, they are **slower**.
    -   **One-stage detectors** are significantly **faster** (often achieving real-time performance), but historically have been slightly less accurate than their two-stage counterparts, though this gap has narrowed considerably with recent models like YOLOv8.

**When to use which:**
-   Use a **two-stage detector** when accuracy is the absolute priority and you can afford the computational cost (e.g., medical image analysis).
-   Use a **one-stage detector** when you need real-time performance (e.g., autonomous driving, video surveillance).

**Q10: Explain the U-Net architecture. What is its key feature and why is it so effective for semantic segmentation?**

**Expected Answer:**
U-Net is a fully convolutional network architecture designed for biomedical image segmentation, but it has become a standard for many semantic segmentation tasks. It has a characteristic "U" shape.

**Architecture:**
1.  **Encoder (Downsampling Path):** This is a typical contracting path of a CNN. It consists of repeated blocks of convolutions and max pooling. Its job is to capture the context and learn feature representations at different scales, while reducing the spatial dimensions.
2.  **Decoder (Upsampling Path):** This is an expansive path that symmetrically mirrors the encoder. It uses upsampling (or transposed convolutions) to gradually increase the spatial resolution of the feature maps back to the original image size.

**Key Feature: Skip Connections**
The most important feature of U-Net is the **skip connections** that link the output of the encoder layers directly to the correspondingly-sized layers in the decoder.

**Why it's effective:**
The encoder, through its pooling layers, captures the "what" (the semantic context) but loses the "where" (precise spatial information). The decoder's job is to reconstruct the "where". The skip connections provide the decoder with the high-resolution feature maps from the encoder, which contain the fine-grained spatial detail that was lost during downsampling. By concatenating these high-resolution features with the upsampled features, the decoder can produce much more precise segmentation masks with sharp boundaries.

### Advanced Questions (5+ years experience)

**Q11: What is the difference between semantic segmentation and instance segmentation? Name an architecture for instance segmentation and explain how it extends a standard object detector.**

**Expected Answer:**
-   **Semantic Segmentation** classifies every pixel into a category (e.g., car, road, person). It does not distinguish between different instances of the same class. If there are three cars in an image, all their pixels are simply labeled "car".
-   **Instance Segmentation** goes a step further. It detects *and* segments each individual object instance. In the same image with three cars, it would produce three separate masks, one for each car, and label them all as "car". It's a combination of object detection and semantic segmentation.

**Architecture: Mask R-CNN**
Mask R-CNN is a classic architecture for instance segmentation. It extends the **Faster R-CNN** (a two-stage object detector) in a simple but powerful way.

**How it extends Faster R-CNN:**
Faster R-CNN has two outputs for each proposed region: a class label and a bounding-box offset. Mask R-CNN adds a **third parallel branch** that outputs a pixel-level object mask.

1.  Like Faster R-CNN, it uses a Region Proposal Network (RPN) to generate candidate object boxes.
2.  For each proposal, it extracts features using a technique called **RoIAlign**. This is a crucial improvement over the RoIPooling used in Faster R-CNN, as it avoids quantization and preserves precise pixel-level alignment, which is critical for generating high-quality masks.
3.  After RoIAlign, the features are fed into three parallel heads:
    -   A classification head (to predict the object's class).
    -   A bounding-box regression head (to refine the box coordinates).
    -   A **mask head** (a small Fully Convolutional Network - FCN) that generates a binary mask for the object within the bounding box.

The total loss is the sum of the classification, box, and mask losses. This allows the model to be trained end-to-end to perform all three tasks simultaneously.

---

## Section 3: Practical Application & System Design

### Scenario-Based Questions

**Q12: You are tasked with building a system to classify 10 different types of manufacturing defects from images. You only have about 200 images per defect type. Walk me through your approach.**

**Expected Answer:**
With only 200 images per class (2,000 total), this is a classic small-data problem where training from scratch is not feasible. My approach would be centered around transfer learning and aggressive data augmentation.

**1. Model Selection & Transfer Learning:**
-   I would start with a proven, pre-trained CNN architecture like **ResNet-50** or **EfficientNet-B0**, trained on ImageNet. These models have already learned powerful, general-purpose visual features.
-   I would employ a **fine-tuning** strategy. I'd freeze the early convolutional layers (which learned generic features like edges and textures) and only train the later layers and a new, custom classification head (with 10 outputs for the defect types).
-   I'd use a very **low learning rate** (e.g., 1e-4 or 1e-5) to avoid destroying the valuable pre-trained weights.

**2. Data Augmentation:**
-   This is critical. I would apply a strong set of augmentations on-the-fly during training to artificially expand the dataset.
-   **Geometric:** Random horizontal/vertical flips (if defects are orientation-agnostic), small rotations (±15 degrees), random crops, and scaling.
-   **Photometric:** Adjustments to brightness, contrast, and saturation to simulate different lighting conditions on the factory floor.
-   **Advanced:** Techniques like **Cutout** (randomly erasing parts of the image) would force the model to learn from partial evidence, making it more robust.

**3. Training & Validation:**
-   Given the small dataset, I would use **stratified k-fold cross-validation** (e.g., 5 folds) to get a more reliable estimate of the model's performance and ensure it's not overfitting to a specific train/validation split.
-   I would monitor the validation loss and use **early stopping** to prevent overfitting.

**4. Deployment & Iteration:**
-   The resulting model would be lightweight enough to deploy on an edge device near the assembly line for real-time inspection.
-   In production, I would set up a system to log and save images where the model has low confidence. These "hard examples" can be manually labeled and added to the training set to iteratively improve the model over time (active learning).

**Q13: You've deployed a computer vision model in production. How would you monitor its performance and detect when it needs to be retrained?**

**Expected Answer:**
Monitoring a CV model requires tracking more than just accuracy. I'd set up a multi-layered monitoring system:

**1. Infrastructure & Operational Metrics:**
-   **Latency:** P95 and P99 prediction time. A sudden increase could indicate an issue.
-   **Throughput:** Predictions per second.
-   **Error Rate:** Rate of failed requests or exceptions.

**2. Model Performance Metrics:**
-   **Accuracy/Precision/Recall/F1-Score:** This is the ground truth, but often has a delay. I'd track this on a dashboard as soon as labels become available (e.g., from manual QA). A sustained drop below a set threshold (e.g., 5%) would trigger an alert.
-   **Confidence Score Distribution:** I'd log the model's prediction confidence for every inference. A shift in this distribution is a powerful leading indicator of a problem. For example, if the average confidence drops from 90% to 70%, the model is becoming less certain, likely due to seeing new data. This can be detected long before ground truth labels are available.

**3. Data Drift Detection:**
-   This is the most important proactive monitoring. I would monitor the statistical properties of the input images.
-   **Image-level metrics:** I'd track the distribution of brightness, contrast, and sharpness of incoming images and compare it to the training set's distribution. A significant shift (detected with a statistical test like the Kolmogorov-Smirnov test) indicates **data drift**.
-   **Feature-level drift:** I'd also pass incoming images through the frozen backbone of the model and monitor the distribution of the resulting feature vectors. A drift here is a very strong signal that the model is seeing data it wasn't trained on.

**Retraining Triggers:**
I would set up automated alerts to trigger a retraining pipeline based on:
1.  **Performance-based trigger:** Validation accuracy drops by more than 5% over a 7-day rolling window.
2.  **Drift-based trigger:** Data drift is detected on key image properties for more than 3 consecutive days.
3.  **Scheduled trigger:** A mandatory retraining every quarter to incorporate new data and patterns, regardless of performance.
4.  **Volume-based trigger:** Retrain after collecting a certain number of new labeled images (e.g., 5,000).

---
