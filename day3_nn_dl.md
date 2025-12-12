# Day 3: Neural Networks & Deep Learning Essentials

## 1. Neural Network Architecture Fundamentals

### What is a Neural Network?
A computational model inspired by biological neurons, consisting of layers of interconnected nodes that learn to transform input data into desired outputs.

### Basic Architecture Components:

**Input Layer:**
- Receives raw features (e.g., pixel values, text embeddings)
- One node per feature
- No computation, just passes data forward

**Hidden Layers:**
- Transform input through weighted connections
- Each neuron computes: `output = activation(Σ(weight × input) + bias)`
- Deep networks have multiple hidden layers (2+ = "deep learning")

**Output Layer:**
- Final predictions
- Size depends on task:
  - Binary classification: 1 node (sigmoid)
  - Multi-class: N nodes (softmax)
  - Regression: 1 node (linear)

### Types of Architectures:

**1. Feedforward Neural Networks (FNN/MLP):**
```
Input → Hidden1 → Hidden2 → ... → Output
```
- Data flows in one direction
- Each layer fully connected to next
- Use case: Tabular data, traditional ML tasks

**2. Convolutional Neural Networks (CNN):**
```
Input → Conv → Pool → Conv → Pool → Flatten → Dense → Output
```
- Specialized for grid-like data (images, time series)
- Convolution layers detect local patterns
- Pooling reduces dimensionality
- Use case: Image classification, object detection

**3. Recurrent Neural Networks (RNN/LSTM/GRU):**
```
Input_t → Hidden_t → Output_t
    ↑         ↓
    └─────────┘ (feedback loop)
```
- Processes sequential data with memory
- LSTM/GRU solve vanishing gradient problem
- Use case: Text, speech, time series

**4. Transformers:**
```
Input → Self-Attention → Feed-Forward → Output
```
- Attention mechanism weighs importance of different inputs
- Processes sequences in parallel (faster than RNN)
- Use case: NLP (BERT, GPT), modern vision tasks

### Network Capacity:
- **Width**: Number of neurons per layer (controls capacity)
- **Depth**: Number of layers (learns hierarchical features)
- More parameters = more capacity BUT higher overfitting risk
- Universal Approximation Theorem: Single hidden layer with enough neurons can approximate any function

---

## 2. Activation Functions, Loss Functions, Optimizers

### Activation Functions
Transform weighted sums into non-linear outputs. Without them, network = linear regression.

**1. ReLU (Rectified Linear Unit):**
```python
f(x) = max(0, x)
```
- **Pros**: Fast, solves vanishing gradient, sparse activation
- **Cons**: Dead neurons (outputs always 0 if weights push x < 0)
- **Use**: Default choice for hidden layers in most networks

**2. Leaky ReLU / PReLU:**
```python
f(x) = max(0.01x, x)  # Leaky ReLU
f(x) = max(αx, x)     # PReLU (α is learned)
```
- **Pros**: Prevents dead neurons
- **Use**: When ReLU neurons are dying

**3. Sigmoid:**
```python
f(x) = 1 / (1 + e^(-x))
```
- **Output**: (0, 1)
- **Pros**: Smooth, interpretable as probability
- **Cons**: Vanishing gradient, not zero-centered
- **Use**: Binary classification output layer, gates in LSTM

**4. Tanh:**
```python
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```
- **Output**: (-1, 1)
- **Pros**: Zero-centered (better than sigmoid)
- **Cons**: Vanishing gradient
- **Use**: Hidden layers in RNNs

**5. Softmax:**
```python
f(x_i) = e^(x_i) / Σ(e^(x_j))
```
- **Output**: Probability distribution (sum = 1)
- **Use**: Multi-class classification output layer

**6. GELU (Gaussian Error Linear Unit):**
```python
f(x) = x × Φ(x)  # Φ = cumulative distribution function
```
- **Use**: Modern transformers (BERT, GPT)
- **Pros**: Smooth, stochastic regularization effect

**Quick Selection Guide:**
- Hidden layers: ReLU (default), Leaky ReLU if dying neurons
- Binary output: Sigmoid
- Multi-class output: Softmax
- Regression output: Linear (no activation)
- Transformers: GELU

### Loss Functions
Measure difference between predictions and actual values.

**Classification:**

**1. Binary Cross-Entropy:**
```python
L = -[y log(ŷ) + (1-y) log(1-ŷ)]
```
- For binary classification (0 or 1)
- Use with sigmoid output

**2. Categorical Cross-Entropy:**
```python
L = -Σ(y_i × log(ŷ_i))
```
- For multi-class classification (one-hot encoded)
- Use with softmax output

**3. Sparse Categorical Cross-Entropy:**
- Same as above but accepts integer labels (not one-hot)
- More memory efficient

**Regression:**

**1. Mean Squared Error (MSE):**
```python
L = (1/n) × Σ(y - ŷ)²
```
- Penalizes large errors heavily
- Use for continuous targets

**2. Mean Absolute Error (MAE):**
```python
L = (1/n) × Σ|y - ŷ|
```
- More robust to outliers
- Use when outliers present

**3. Huber Loss:**
- Combination of MSE (small errors) and MAE (large errors)
- Balances both approaches

**Advanced:**

**4. Focal Loss:**
```python
L = -α(1-ŷ)^γ × log(ŷ)
```
- Down-weights easy examples, focuses on hard ones
- Use for extreme class imbalance

**5. Contrastive/Triplet Loss:**
- For similarity learning
- Use in face recognition, metric learning

### Optimizers
Algorithms to update weights during training.

**1. Stochastic Gradient Descent (SGD):**
```python
w = w - learning_rate × gradient
```
- **Pros**: Simple, generalizes well, works on large datasets
- **Cons**: Slow convergence, sensitive to learning rate
- **Use**: When you have time to tune, need best generalization

**2. SGD with Momentum:**
```python
v = β×v + gradient
w = w - learning_rate × v
```
- Accumulates gradients (like a ball rolling)
- Faster convergence, less oscillation
- β typically 0.9

**3. Adam (Adaptive Moment Estimation):**
```python
# Combines momentum + adaptive learning rates
m = β1×m + (1-β1)×gradient  # First moment
v = β2×v + (1-β2)×gradient²  # Second moment
w = w - lr × m/√(v + ε)
```
- **Pros**: Fast convergence, adaptive per-parameter learning rates, good default
- **Cons**: May generalize worse than SGD, requires more memory
- **Hyperparameters**: lr=0.001, β1=0.9, β2=0.999 (defaults work well)
- **Use**: Default choice, especially for transformers

**4. AdamW:**
- Adam with decoupled weight decay
- Better regularization than Adam
- **Use**: Modern best practice for transformers

**5. RMSprop:**
- Adapts learning rate per parameter
- **Use**: Good for RNNs

**6. Adagrad:**
- Adapts learning rate based on historical gradients
- **Use**: Sparse data, NLP

**Quick Selection Guide:**
- Starting out: Adam (lr=0.001)
- Best generalization needed: SGD with momentum (lr=0.01-0.1)
- Transformers/NLP: AdamW
- RNNs: RMSprop or Adam
- Computer vision: SGD with momentum or Adam

---

## 3. Backpropagation and Gradient Descent Variants

### Backpropagation
The algorithm to compute gradients efficiently using chain rule.

**Forward Pass:**
1. Input flows through network
2. Each layer computes output = activation(weights × input + bias)
3. Loss computed at output

**Backward Pass:**
1. Start with loss gradient
2. Chain rule: ∂Loss/∂w = ∂Loss/∂output × ∂output/∂activation × ∂activation/∂w
3. Propagate gradients backward through each layer
4. Update weights using optimizer

**Key Insight:** Gradients flow backward, each layer receives gradient from next layer and passes gradient to previous layer.

**Computational Graph Example:**
```
x → w1 → z1 → a1 → w2 → z2 → a2 → Loss
```
Backward:
```
∂Loss/∂w1 = ∂Loss/∂a2 × ∂a2/∂z2 × ∂z2/∂a1 × ∂a1/∂z1 × ∂z1/∂w1
```

### Gradient Descent Variants

**1. Batch Gradient Descent:**
- Uses entire dataset for each update
- **Pros**: Smooth convergence, stable gradients
- **Cons**: Slow, memory intensive, stuck in local minima
- **Use**: Small datasets only

**2. Stochastic Gradient Descent (SGD):**
- Uses one sample per update
- **Pros**: Fast, escapes local minima, online learning
- **Cons**: Noisy, erratic convergence
- **Use**: Large datasets, online learning

**3. Mini-Batch Gradient Descent:**
- Uses small batch (16-512 samples) per update
- **Pros**: Balance of speed and stability, GPU efficient
- **Cons**: Requires batch size tuning
- **Use**: Standard practice (default)
- **Batch size selection:**
  - Smaller (16-32): Better generalization, more noise
  - Larger (128-512): Faster training, more stable, needs more memory
  - Rule of thumb: Start with 32, increase if training slow

**Learning Rate Scheduling:**

**1. Step Decay:**
- Reduce lr by factor every N epochs
```python
lr = initial_lr × 0.1^(epoch // 10)
```

**2. Cosine Annealing:**
- Gradually decrease following cosine curve
- Smooth transitions

**3. ReduceLROnPlateau:**
- Reduce when validation metric stops improving
- Adaptive, works well in practice

**4. Warmup:**
- Start with small lr, gradually increase
- Important for large batch training, transformers

**5. Cyclical Learning Rates:**
- Cycle between min and max lr
- Helps escape local minima

---

## 4. Overfitting, Regularization (Dropout, L1/L2, Early Stopping)

### Overfitting Problem
Model memorizes training data, fails on new data. Signs:
- Training accuracy ↑, validation accuracy ↓
- Training loss ↓, validation loss ↑
- Model performs well on seen data, poorly on unseen

### Regularization Techniques

**1. Dropout:**
```python
# During training: randomly set p% of neurons to 0
layer = Dropout(rate=0.5)  # Drop 50% of neurons
```
- **How**: Randomly drops neurons during training, forces redundancy
- **Effect**: Prevents co-adaptation, acts like ensemble
- **Rate**: 0.2-0.5 for dense layers, 0.1-0.2 for CNNs
- **Important**: Only active during training, disabled during inference
- **Use**: Hidden layers, especially in fully connected layers

**2. L2 Regularization (Weight Decay):**
```python
Loss = Original_Loss + λ × Σ(weights²)
```
- **How**: Penalizes large weights, keeps them small
- **Effect**: Smoother model, less sensitive to individual features
- **λ (lambda)**: 0.001-0.01 typical, smaller = less regularization
- **Use**: All layers, especially effective with SGD

**3. L1 Regularization:**
```python
Loss = Original_Loss + λ × Σ|weights|
```
- **How**: Penalizes absolute weight values
- **Effect**: Drives weights to exactly 0, creates sparse models
- **Use**: Feature selection, when you want sparse networks

**4. Early Stopping:**
```python
# Stop training when validation loss stops improving
patience = 10  # Wait 10 epochs before stopping
monitor = 'val_loss'
```
- **How**: Track validation loss, stop when no improvement
- **Pros**: Simple, no hyperparameters to tune
- **Use**: Always! Standard practice
- **Tips**: Save best model (not final), use patience 5-20

**5. Data Augmentation:**
```python
# Images: rotation, flip, crop, color jitter
# Text: synonym replacement, back-translation
# Audio: time stretch, pitch shift, noise
```
- **How**: Create variations of training data
- **Effect**: More diverse training data = better generalization
- **Use**: Computer vision (essential), NLP, audio

**6. Ensemble Methods:**
- Train multiple models, average predictions
- Reduces variance, improves robustness
- **Use**: When accuracy critical, have compute budget

**Strategy for Combating Overfitting:**
1. **First**: Add more training data (best solution)
2. **Second**: Apply data augmentation
3. **Third**: Add dropout (0.5 for dense, 0.2 for CNN)
4. **Fourth**: Add L2 regularization (0.01)
5. **Fifth**: Reduce model complexity (fewer layers/neurons)
6. **Always**: Use early stopping
7. **Last resort**: Get more data or change problem formulation

---

## 5. Batch Normalization, Layer Normalization

### Why Normalize?
Internal Covariate Shift: As network learns, layer inputs' distributions change, making training unstable and slow.

### Batch Normalization (BatchNorm)

**How it works:**
```python
# For each feature in mini-batch:
mean = calculate_mean_across_batch(x)
variance = calculate_variance_across_batch(x)
x_normalized = (x - mean) / sqrt(variance + ε)
x_scaled = γ × x_normalized + β  # γ, β are learned parameters
```

**Benefits:**
- Faster training (can use higher learning rates)
- Reduces sensitivity to initialization
- Acts as regularization (slight noise from batch statistics)
- Allows deeper networks

**When to use:**
```python
# Typical placement:
x = Dense(128)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
```

**Important considerations:**
- Different behavior in training vs inference
- Training: Uses batch statistics
- Inference: Uses running average of statistics
- Works best with batch size ≥ 16
- Not ideal for very small batches or RNNs

**Position in layer:**
- Original paper: Before activation
- Modern practice: After activation (both work)

### Layer Normalization (LayerNorm)

**How it works:**
```python
# Normalize across features (not batch):
mean = calculate_mean_across_features(x)
variance = calculate_variance_across_features(x)
x_normalized = (x - mean) / sqrt(variance + ε)
x_scaled = γ × x_normalized + β
```

**Key difference from BatchNorm:**
- BatchNorm: Normalizes across batch dimension (all samples for each feature)
- LayerNorm: Normalizes across feature dimension (all features for each sample)

**Benefits:**
- Works with any batch size (including batch=1)
- Same computation in training and inference
- Better for RNNs and Transformers

**When to use:**
- Transformers (standard practice)
- RNNs/LSTMs
- Small batch sizes
- Online/streaming inference

### Comparison:

| Aspect | Batch Normalization | Layer Normalization |
|--------|-------------------|-------------------|
| **Normalizes** | Across batch | Across features |
| **Best for** | CNNs, large batches | RNNs, Transformers |
| **Batch size** | Needs ≥16 | Works with any |
| **Training/Inference** | Different | Same |
| **Sequence data** | Problematic | Excellent |

### Other Normalization Techniques:

**Group Normalization:**
- Divide features into groups, normalize each group
- Middle ground between Layer and Batch
- Use for: Small batches, computer vision

**Instance Normalization:**
- Normalize each sample independently
- Use for: Style transfer, GANs

**Weight Normalization:**
- Normalizes weight vectors instead of activations
- Use for: Alternative to BatchNorm

---

## 6. Transfer Learning Concepts

### What is Transfer Learning?
Using knowledge from a model trained on one task to improve learning on a different but related task.

**Core Idea:** Pre-trained models learned useful feature representations that generalize across tasks.

### Why Transfer Learning?

**Advantages:**
- Requires less training data (critical for small datasets)
- Faster training (start from good initialization)
- Better performance (leverages large-scale pre-training)
- Lower computational cost

**Example:** ImageNet-pretrained models recognize edges, textures, shapes → useful for any image task

### Transfer Learning Strategies

**1. Feature Extraction (Frozen Base):**
```python
# Load pre-trained model
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,  # Remove final classification layer
    input_shape=(224, 224, 3)
)

# Freeze all layers
base_model.trainable = False

# Add custom head
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Train only new layers
model.compile(optimizer='adam', ...)
```

**When to use:**
- Small dataset (<1000 samples per class)
- Similar to source task
- Limited compute

**2. Fine-Tuning (Gradual Unfreezing):**
```python
# First: Train head with frozen base (few epochs)
base_model.trainable = False
model.fit(X_train, y_train, epochs=5)

# Then: Unfreeze some layers and fine-tune
base_model.trainable = True
# Freeze early layers, unfreeze later layers
for layer in base_model.layers[:100]:
    layer.trainable = False

# Use smaller learning rate for fine-tuning
model.compile(optimizer=Adam(lr=1e-5), ...)
model.fit(X_train, y_train, epochs=20)
```

**When to use:**
- Medium dataset (1000-10,000 samples)
- Moderate similarity to source task
- Have compute resources

**3. Full Fine-Tuning:**
- Unfreeze entire model
- Use small learning rate (1e-4 to 1e-5)
- When to use: Large dataset (10,000+ samples), different domain

### Transfer Learning Best Practices

**1. Learning Rate Strategy:**
- Frozen layers: Normal lr (1e-3)
- Fine-tuning: Very small lr (1e-5 to 1e-4)
- Discriminative learning rates: Different lr for different layers
  - Early layers: 1e-5 (general features)
  - Later layers: 1e-4 (task-specific)
  - New head: 1e-3 (random initialization)

**2. How Many Layers to Unfreeze:**
- Small dataset: Only head (0 base layers)
- Medium dataset: Last 25-50% of layers
- Large dataset: All layers
- Rule: More data = unfreeze more layers

**3. Domain Similarity:**
- **Very similar** (dogs → cats): Feature extraction works
- **Somewhat similar** (ImageNet → medical images): Fine-tune
- **Very different** (images → audio): May not help or need full retraining

### Common Pre-trained Models

**Computer Vision:**
- **ResNet50/101/152**: Reliable baseline, medium size
- **EfficientNet**: Best accuracy/efficiency trade-off
- **VGG16**: Simple, large, older but stable
- **MobileNet**: Fast inference, mobile devices
- **Vision Transformer (ViT)**: Modern, requires more data

**NLP:**
- **BERT**: Bidirectional, best for understanding
- **GPT**: Autoregressive, best for generation
- **RoBERTa**: Improved BERT
- **T5**: Unified text-to-text framework
- **DistilBERT**: Smaller, faster BERT

**Multi-modal:**
- **CLIP**: Vision + language
- **DALL-E**: Text to image

### Transfer Learning Workflow

1. **Choose pre-trained model** based on:
   - Source task similarity
   - Model size (compute/memory budget)
   - Inference speed requirements

2. **Assess dataset size:**
   - <1K: Feature extraction only
   - 1K-10K: Feature extraction + fine-tune top layers
   - >10K: Full fine-tuning

3. **Start conservative:**
   - Freeze base, train head (5-10 epochs)
   - Check if learning, adjust head architecture if needed

4. **Gradually unfreeze:**
   - Unfreeze top layers (5-10 epochs)
   - Unfreeze more layers if needed
   - Use smaller learning rates each step

5. **Monitor carefully:**
   - Watch for overfitting (validation loss)
   - Use early stopping
   - Save best checkpoint

---

## 7. Interview Focus: When to Use Deep Learning vs Classical ML

### Decision Framework

**Use Classical ML (Tree-based, Linear Models) When:**

✅ **Small to medium tabular data** (<100K rows)
- Reason: DL needs large data to shine, classical ML performs better with less
- Example: Predicting house prices (10K samples, 50 features)

✅ **Interpretability critical**
- Reason: Business/regulatory needs to understand decisions
- Example: Credit scoring, medical diagnosis

✅ **Limited compute/infrastructure**
- Reason: Classical ML trains on CPU, no GPU needed
- Example: Small startup, edge devices

✅ **Fast prototyping needed**
- Reason: Classical ML trains in minutes vs hours/days for DL
- Example: Proof of concept, A/B testing

✅ **Features are well-engineered**
- Reason: If domain knowledge captured in features, classical ML sufficient
- Example: Financial ratios for stock prediction

✅ **Linear/simple relationships**
- Reason: No need for complex models
- Example: Click-through rate prediction with sparse features

**Real Interview Answer:**
"For tabular data with 50K rows and 100 features predicting customer churn, I'd start with LightGBM because: (1) Dataset size doesn't justify deep learning overhead, (2) Tree models handle mixed feature types naturally, (3) Feature importance helps business understand drivers, (4) Training takes minutes not hours, (5) Easier to deploy and maintain."

### Use Deep Learning When:**

✅ **Unstructured data**
- Images: Object detection, segmentation
- Text: Sentiment analysis, translation
- Audio: Speech recognition
- Video: Action recognition
- Reason: DL automatically learns representations, classical ML needs manual feature engineering

✅ **Very large datasets** (>100K samples)
- Reason: DL capacity scales with data
- Example: Million user interactions for recommendation

✅ **Complex patterns/relationships**
- Reason: Deep hierarchical representations
- Example: Image classification (edges → textures → objects)

✅ **End-to-end learning beneficial**
- Reason: Joint optimization of feature extraction + prediction
- Example: Raw audio → transcript (no manual features)

✅ **Transfer learning available**
- Reason: Leverage pre-trained models
- Example: Medical image classification (use ImageNet models)

✅ **State-of-the-art performance needed**
- Reason: DL holds records on most benchmarks
- Example: Kaggle competitions, research papers

**Real Interview Answer:**
"For classifying 100K medical images as normal/abnormal, I'd use deep learning because: (1) Unstructured image data—DL automatically learns features, (2) Transfer learning from ImageNet provides strong baseline, (3) CNNs specifically designed for spatial data, (4) Classical ML would need manual feature engineering (edges, textures), (5) Large dataset justifies DL complexity."

### Hybrid Approaches

**When to combine both:**

**1. Feature extraction with DL + Classical ML:**
```python
# Use DL for feature extraction
features = pretrained_model.predict(images)
# Use classical ML for final prediction
xgb_model.fit(features, labels)
```
**Why:** DL for complex features, classical ML for interpretability/speed
**Example:** Extract image embeddings, use XGBoost for classification

**2. Ensemble of both:**
```python
# Average predictions from both model types
final_pred = 0.5 * dl_model.predict(X) + 0.5 * xgb_model.predict(X)
```
**Why:** Combine strengths, reduce variance
**Example:** Kaggle competitions often win with ensemble

**3. DL for feature engineering:**
- Train autoencoder to learn compressed representations
- Use compressed features in classical ML
**Why:** Dimensionality reduction with DL, fast inference with classical ML

### Common Interview Questions & Answers

**Q: Client has 10K samples, 100 features, wants to predict customer churn. Deep learning or classical ML?**

A: "Classical ML, specifically Gradient Boosting (LightGBM/XGBoost):
1. **Data size**: 10K samples insufficient for deep learning to outperform
2. **Feature type**: Likely tabular mixed types—tree models handle naturally
3. **Interpretability**: Business needs to understand churn drivers (feature importance)
4. **Speed**: Train in minutes, iterate quickly on features
5. **Deployment**: Simpler infrastructure, no GPU needed
6. **Cost**: Lower compute costs

Would only consider DL if: (1) We can collect 100K+ samples, or (2) Features are sequences/text requiring specialized architectures."

**Q: When does deep learning start to outperform gradient boosting on tabular data?**

A: "General guidelines (not absolute):
- **Sample size**: >100K rows where DL capacity utilizes more data
- **Feature interactions**: Extremely high-order interactions trees can't capture
- **Mixed modalities**: Combining tabular + images/text
- **Computation**: When inference time matters less than accuracy

However, recent models like TabNet, SAINT show promise for tabular data. Still, in practice:
- Kaggle tabular competitions: Usually won by ensembles of GBMs
- Production tabular: 95% use XGBoost/LightGBM
- Reason: Better performance/effort ratio for typical business datasets"

**Q: You have 1M images, 100 categories. Architecture choice?**

A: "Deep Learning, specifically:
1. **First choice**: Transfer learning with EfficientNet or ResNet50
   - Why: Proven architecture, leverages ImageNet features
   - Start: Freeze base, train head
   - Then: Fine-tune if needed

2. **If accuracy critical**: Vision Transformer (ViT)
   - Why: State-of-the-art results
   - Con: Needs more data/compute

3. **If speed critical**: MobileNetV3
   - Why: Optimized for fast inference
   - Use case: Mobile deployment

Would benchmark 2-3 architectures, select based on accuracy/speed/deployment tradeoff."

**Q: Project has limited GPU budget. How do you decide?**

A: "Decision tree:
1. **Data type check**:
   - Tabular → Classical ML (no GPU needed)
   - Image/Text → Consider transfer learning

2. **For transfer learning**:
   - Use smaller models (MobileNet, DistilBERT)
   - Feature extraction only (freeze base)
   - Train on cloud GPU, deploy on CPU

3. **Cost optimization**:
   - Start with classical ML baseline
   - If DL needed, use:
     - Google Colab free tier (prototyping)
     - AWS Spot instances (70% discount)
     - Gradient checkpointing (reduce memory)
     - Mixed precision training (faster)

4. **Only go full DL if**:
   - Classical ML hits accuracy ceiling
   - ROI justifies compute cost"

**Q: Explain bias-variance tradeoff in deep learning vs classical ML.**

A: "Key differences:

**Classical ML (e.g., Random Forest, XGBoost):**
- High bias: Shallow model, limited capacity
- Low variance: Ensemble averaging stabilizes
- Balance: Easy to control via hyperparameters (depth, n_estimators)

**Deep Learning:**
- Low bias: High capacity, can fit complex functions
- High variance: Prone to overfitting
- Balance: Requires multiple techniques:
  - Dropout, BatchNorm, L2 regularization
  - Data augmentation
  - Early stopping
  - More data (DL benefits greatly from more data)

**Practical implication:**
- Classical ML: Easier to achieve good bias-variance balance with less tuning
- Deep Learning: Requires careful regularization but reaches higher ceiling with sufficient data

**Modern insight:**
- DL often operates in 'overparameterized' regime (more parameters than data)
- Still generalizes due to implicit regularization in SGD
- Double descent: More capacity can reduce both bias AND variance"

### Key Takeaways for Interviews

**Framework to answer "DL vs Classical ML":**
1. **Data type**: Structured → Classical, Unstructured → DL
2. **Data size**: <100K → Classical, >100K → Consider DL
3. **Complexity**: Simple → Classical, Complex patterns → DL
4. **Resources**: Limited → Classical, Ample → DL
5. **Interpretability**: Critical → Classical, Less critical → DL
6. **Time to deploy**: Fast → Classical, Can wait → DL

**Always mention:**
- No free lunch—depends on specific problem
- Start simple, add complexity only if needed
- Measure on your specific data (what works elsewhere may not work here)
- Consider full lifecycle: training, deployment, maintenance, monitoring

---

## Summary

**Neural Networks Fundamentals:**
- Architecture types: FNN, CNN, RNN, Transformer
- Key: Depth (hierarchical features), Width (capacity)

**Activation Functions:**
- Hidden layers: ReLU (default), Leaky ReLU (if dead neurons)
- Output: Sigmoid (binary), Softmax (multi-class), Linear (regression)

**Optimizers:**
- Quick start: Adam (lr=0.001)
- Best generalization: SGD with momentum
- Modern NLP: AdamW

**Regularization:**
- Always: Early stopping
- Dense layers: Dropout (0.5)
- All weights: L2 regularization (0.01)
- Best: More data + augmentation

**Normalization:**
- CNNs: Batch Normalization
- Transformers/RNNs: Layer Normalization
- Position: After dense, before activation

**Transfer Learning:**
- Small data: Feature extraction (frozen)
- Medium data: Fine-tuning (partial unfreezing)
- Large data: Full fine-tuning
- Always use pre-trained models when available

**DL vs Classical ML:**
- Tabular + small data → Classical ML
- Images/Text/Audio → Deep Learning
- Need interpretability → Classical ML
- State-of-the-art accuracy → Deep Learning

---

**Tomorrow: Convolutional Neural Networks (CNNs) - Architecture, Applications & Best Practices**
