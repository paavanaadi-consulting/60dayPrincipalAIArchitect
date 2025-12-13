# Day 3: Neural Networks & Deep Learning - Cheatsheet

## 🧠 Quick Reference Guide for Neural Networks & Deep Learning

---

## 1. 📐 Neural Network Architectures

### Architecture Types & Use Cases

| **Architecture** | **Best For** | **Key Features** | **Example** |
|------------------|--------------|------------------|-------------|
| **MLP/FNN** | Tabular data | Fully connected, feedforward | Customer churn, fraud detection |
| **CNN** | Images, spatial data | Convolution, pooling, parameter sharing | Image classification, object detection |
| **RNN/LSTM** | Sequential data | Memory, temporal dependencies | NLP, speech, time series |
| **Transformer** | Sequential data | Attention mechanism, parallel processing | GPT, BERT, machine translation |

### Layer Components

```
Input Layer → Hidden Layers → Output Layer

Hidden Layer Computation:
output = activation(Σ(weight × input) + bias)

Output Layer Size:
• Binary classification: 1 node (sigmoid)
• Multi-class: N nodes (softmax)  
• Regression: 1 node (linear)
```

### Network Capacity
- **Width**: Neurons per layer (↑ capacity)
- **Depth**: Number of layers (↑ hierarchical features)
- **Rule**: More parameters = more capacity BUT higher overfitting risk

---

## 2. ⚡ Activation Functions

### Quick Selection Guide

| **Use Case** | **Activation** | **Formula** | **Range** |
|--------------|----------------|-------------|-----------|
| **Hidden layers (default)** | ReLU | `max(0, x)` | [0, ∞) |
| **Hidden layers (dying ReLU)** | Leaky ReLU | `max(0.01x, x)` | (-∞, ∞) |
| **Binary output** | Sigmoid | `1/(1+e^(-x))` | (0, 1) |
| **Multi-class output** | Softmax | `e^(xi)/Σe^(xj)` | [0, 1], Σ=1 |
| **Regression output** | Linear | `x` | (-∞, ∞) |
| **Modern transformers** | GELU | `x × Φ(x)` | (-∞, ∞) |
| **RNN hidden** | Tanh | `(e^x-e^(-x))/(e^x+e^(-x))` | (-1, 1) |

### Activation Function Properties

| **Function** | **Pros** | **Cons** | **When to Use** |
|--------------|----------|----------|-----------------|
| **ReLU** | Fast, sparse, no vanishing gradient | Dead neurons | Default choice |
| **Leaky ReLU** | Prevents dead neurons | Slight negative slope | When ReLU neurons dying |
| **Sigmoid** | Smooth, probabilistic | Vanishing gradient, not zero-centered | Binary classification output |
| **Tanh** | Zero-centered | Vanishing gradient | RNN hidden layers |
| **Softmax** | Probability distribution | Only for output | Multi-class output |

---

## 3. 📉 Loss Functions

### Classification Losses

| **Task** | **Loss Function** | **Formula** | **Output Activation** |
|----------|-------------------|-------------|----------------------|
| **Binary** | Binary Cross-Entropy | `-[y log(ŷ) + (1-y)log(1-ŷ)]` | Sigmoid |
| **Multi-class** | Categorical Cross-Entropy | `-Σ(yi × log(ŷi))` | Softmax |
| **Multi-class (int labels)** | Sparse Categorical | Same as above | Softmax |
| **Imbalanced** | Focal Loss | `-α(1-ŷ)^γ log(ŷ)` | Sigmoid/Softmax |

### Regression Losses

| **Scenario** | **Loss Function** | **Formula** | **When to Use** |
|--------------|-------------------|-------------|-----------------|
| **Standard** | MSE | `(1/n)Σ(y-ŷ)²` | Normal distribution, penalize large errors |
| **Outliers present** | MAE | `(1/n)Σ|y-ŷ|` | Robust to outliers |
| **Balanced** | Huber | MSE + MAE hybrid | Balance smoothness & robustness |

---

## 4. 🎯 Optimizers

### Optimizer Selection Guide

| **Optimizer** | **Learning Rate** | **Use Case** | **Pros** | **Cons** |
|---------------|-------------------|--------------|----------|----------|
| **Adam** | 0.001 | Default choice, transformers | Fast convergence, adaptive LR | May generalize worse |
| **SGD + Momentum** | 0.01-0.1 | Best generalization, CV | Best final performance | Slow convergence, needs tuning |
| **AdamW** | 0.001 | Modern transformers | Better regularization than Adam | More complex |
| **RMSprop** | 0.001 | RNNs | Good for RNNs | Less popular |

### Optimizer Formulas

```python
# SGD with Momentum
v = β × v + gradient
w = w - lr × v

# Adam
m = β1 × m + (1-β1) × gradient      # First moment
v = β2 × v + (1-β2) × gradient²     # Second moment  
w = w - lr × m/√(v + ε)             # Weight update
```

### Learning Rate Scheduling

| **Method** | **When to Use** | **Implementation** |
|------------|-----------------|-------------------|
| **Step Decay** | Simple baseline | Reduce by 0.1 every N epochs |
| **Cosine Annealing** | Smooth decrease | Follow cosine curve |
| **ReduceLROnPlateau** | Adaptive | Reduce when validation stops improving |
| **Warmup** | Large batch, transformers | Gradually increase initially |

---

## 5. 🔄 Gradient Descent Variants

### Batch Size Selection

| **Batch Size** | **Pros** | **Cons** | **Use Case** |
|----------------|----------|----------|--------------|
| **1 (SGD)** | Fast, escapes local minima | Noisy, erratic | Large datasets, online learning |
| **16-128 (Mini-batch)** | Balanced speed/stability | Need tuning | **Standard practice** |
| **Full dataset (Batch)** | Smooth convergence | Slow, memory intensive | Small datasets only |

### Batch Size Guidelines
- **Start with**: 32 (good default)
- **Increase if**: Training is noisy, have more memory
- **Decrease if**: Overfitting, memory issues
- **Modern trend**: Large batches (512-4096) with LR scaling

---

## 6. 🛡️ Regularization Techniques

### Overfitting Detection
- **Signs**: Training ↑, Validation ↓
- **Training loss** ↓, **Validation loss** ↑
- Large gap between train/val performance

### Regularization Methods (Priority Order)

| **Priority** | **Method** | **Implementation** | **Typical Values** |
|--------------|------------|-------------------|-------------------|
| **1. Always** | Early Stopping | Monitor val_loss, patience=10-20 | Stop when no improvement |
| **2. Data** | Data Augmentation | Rotation, flip, crop (images) | Essential for CV |
| **3. Architecture** | Dropout | Randomly zero neurons | 0.5 (dense), 0.2 (CNN) |
| **4. Weights** | L2 Regularization | Add λΣw² to loss | λ = 0.001-0.01 |
| **5. Last resort** | Reduce Model Size | Fewer layers/neurons | If above doesn't work |

### Dropout Implementation

```python
# Dense layers
x = Dense(256)(x)
x = Dropout(0.5)(x)  # Drop 50% of neurons
x = Activation('relu')(x)

# CNN layers  
x = Conv2D(64, 3)(x)
x = Dropout(0.2)(x)  # Drop 20% of neurons
x = Activation('relu')(x)
```

### L1 vs L2 Regularization

| **Type** | **Formula** | **Effect** | **Use Case** |
|----------|-------------|------------|--------------|
| **L1** | `λΣ|w|` | Sparse weights (exactly 0) | Feature selection |
| **L2** | `λΣw²` | Small weights | General regularization |

---

## 7. 📊 Normalization

### When to Use Each

| **Normalization** | **Best For** | **Normalizes** | **Batch Size** |
|-------------------|--------------|----------------|----------------|
| **Batch Norm** | CNNs, large batches | Across batch | ≥16 required |
| **Layer Norm** | Transformers, RNNs | Across features | Any size |
| **Group Norm** | Small batches, CV | Feature groups | Any size |

### BatchNorm vs LayerNorm

```python
# Batch Normalization (across batch dimension)
x_norm = (x - mean_batch) / sqrt(var_batch + ε)

# Layer Normalization (across feature dimension)  
x_norm = (x - mean_features) / sqrt(var_features + ε)
```

### Placement in Network

```python
# Common pattern
x = Dense(128)(x)
x = BatchNormalization()(x)  # After dense, before activation
x = Activation('relu')(x)
x = Dropout(0.5)(x)
```

---

## 8. 🔄 Transfer Learning

### Strategy Selection

| **Dataset Size** | **Domain Similarity** | **Strategy** | **Layers to Unfreeze** |
|------------------|----------------------|--------------|----------------------|
| **Small (<1K)** | High | Feature Extraction | 0% (freeze all) |
| **Medium (1K-10K)** | Medium | Fine-tuning | 25-50% (top layers) |
| **Large (>10K)** | Low | Full Fine-tuning | 100% (all layers) |

### Transfer Learning Workflow

```python
# 1. Feature Extraction (Small Data)
base_model.trainable = False
model.fit(X, y, epochs=10, lr=0.001)

# 2. Fine-tuning (Medium/Large Data)  
base_model.trainable = True
# Freeze early layers
for layer in base_model.layers[:100]:
    layer.trainable = False
    
model.compile(optimizer=Adam(lr=1e-5))  # Small LR!
model.fit(X, y, epochs=20)
```

### Learning Rate Strategy

| **Layer Type** | **Learning Rate** | **Reasoning** |
|----------------|-------------------|---------------|
| **Frozen layers** | 0 | Don't update |
| **Early layers** | 1e-5 | General features, small changes |
| **Middle layers** | 1e-4 | Domain adaptation |
| **New head** | 1e-3 | Random initialization, needs more updates |

### Popular Pre-trained Models

| **Domain** | **Model** | **Use Case** | **Size** |
|------------|-----------|--------------|----------|
| **Vision** | ResNet50 | General baseline | Medium |
| **Vision** | EfficientNet | Best accuracy/efficiency | Small-Large |
| **Vision** | MobileNet | Mobile/edge deployment | Small |
| **NLP** | BERT | Text understanding | Large |
| **NLP** | GPT | Text generation | Large |
| **NLP** | DistilBERT | Faster BERT | Medium |

---

## 9. ⚖️ Deep Learning vs Classical ML

### Decision Framework

```
1. Data Type?
   ├─ Tabular → Classical ML (XGBoost/LightGBM)
   └─ Images/Text/Audio → Deep Learning

2. Dataset Size?
   ├─ <100K rows → Classical ML  
   └─ >100K rows → Consider both

3. Interpretability needed?
   ├─ Yes → Classical ML
   └─ No → Deep Learning OK

4. Infrastructure/Time?
   ├─ Limited → Classical ML
   └─ Ample → Deep Learning OK

5. Performance requirements?
   ├─ Good enough → Classical ML (faster)
   └─ State-of-art → Deep Learning
```

### Use Classical ML When:
- ✅ Tabular data (<100K rows)
- ✅ Need interpretability  
- ✅ Limited compute/time
- ✅ Fast prototyping needed
- ✅ Well-engineered features exist

### Use Deep Learning When:
- ✅ Unstructured data (images/text/audio)
- ✅ Large datasets (>100K samples)
- ✅ Complex patterns expected
- ✅ Transfer learning available
- ✅ State-of-art performance needed

---

## 10. 🛠️ Implementation Checklist

### Model Building Checklist

```python
# 1. Data Preparation
□ Normalize/scale features
□ Handle missing values  
□ Train/val/test split
□ Data augmentation (if applicable)

# 2. Architecture Design
□ Choose appropriate architecture (MLP/CNN/RNN/Transformer)
□ Select activation functions (ReLU hidden, sigmoid/softmax output)
□ Add normalization (BatchNorm/LayerNorm)
□ Add dropout for regularization

# 3. Training Setup
□ Choose loss function (binary/categorical cross-entropy, MSE)
□ Select optimizer (Adam default, SGD for best performance)
□ Set learning rate (0.001 Adam, 0.01-0.1 SGD)
□ Configure callbacks (early stopping, LR scheduling)

# 4. Training Process
□ Monitor train/val curves
□ Check for overfitting (train↑ val↓)
□ Apply regularization if needed
□ Save best model (not final)

# 5. Evaluation
□ Test on held-out data
□ Check for data leakage
□ Analyze failure cases
□ Compare with baseline
```

### Common Training Issues & Fixes

| **Problem** | **Symptoms** | **Solutions** |
|-------------|--------------|---------------|
| **Overfitting** | Train↑ Val↓ | Dropout, early stopping, more data |
| **Underfitting** | Both train/val low | More capacity, less regularization |
| **Vanishing gradients** | Loss plateaus early | ReLU, skip connections, BatchNorm |
| **Exploding gradients** | Loss spikes/NaN | Gradient clipping, lower LR |
| **Dead ReLUs** | Many 0 activations | Leaky ReLU, lower LR, better init |
| **Slow convergence** | Loss decreases slowly | Higher LR, Adam optimizer, BatchNorm |

---

## 11. 📝 Interview Quick Answers

### Key Concepts (30 seconds each)

**Q: Why activation functions?**
A: "Without activation functions, neural network = linear regression. Activations introduce non-linearity, enabling universal approximation and deep learning."

**Q: ReLU vs Sigmoid?**  
A: "ReLU: Fast, no vanishing gradients, sparse. Use for hidden layers. Sigmoid: Smooth, outputs probabilities. Use for binary classification output."

**Q: When Adam vs SGD?**
A: "Adam: Fast prototyping, good default (lr=0.001). SGD: Best generalization, needs tuning (lr=0.01-0.1). I start with Adam, switch to SGD if time permits."

**Q: How prevent overfitting?**
A: "Priority order: (1) More data + augmentation, (2) Early stopping, (3) Dropout (0.5 dense, 0.2 CNN), (4) L2 regularization (0.01), (5) Reduce model size."

**Q: Transfer learning strategy?**
A: "Depends on data size and similarity. Small data: freeze base, train head. Medium data: unfreeze top layers with small LR. Large data: fine-tune everything."

**Q: Deep Learning vs Classical ML?**
A: "Tabular data <100K: Classical ML (XGBoost). Images/text or >100K samples: Deep Learning. Classical ML easier/faster, DL higher ceiling with sufficient data."

---

## 12. 🔢 Key Formulas

### Neural Network Forward Pass
```
z = W·x + b           # Linear transformation
a = activation(z)     # Non-linear activation  
loss = L(y, ŷ)       # Loss computation
```

### Backpropagation (Chain Rule)
```
∂L/∂W = ∂L/∂a × ∂a/∂z × ∂z/∂W
∂L/∂b = ∂L/∂a × ∂a/∂z × ∂z/∂b
```

### Common Activations
```
ReLU: f(x) = max(0, x)
Sigmoid: f(x) = 1/(1 + e^(-x))  
Tanh: f(x) = (e^x - e^(-x))/(e^x + e^(-x))
Softmax: f(xi) = e^(xi) / Σe^(xj)
```

### Optimization Updates
```
SGD: w = w - lr × ∂L/∂w
Momentum: v = βv + ∂L/∂w; w = w - lr × v
Adam: Combines momentum + adaptive learning rates
```

---

## 13. 🎯 Performance Benchmarks

### Typical Training Times (Single GPU)
- **MNIST (60K images)**: 5-10 minutes
- **CIFAR-10 (50K images)**: 30-60 minutes  
- **ImageNet (1.3M images)**: 1-3 days
- **BERT (large text corpus)**: 1-4 days

### Memory Requirements
- **Parameters**: 4 bytes per parameter (float32)
- **Activations**: Batch size × layer size × 4 bytes
- **Gradients**: Same as parameters
- **Rule of thumb**: 2-3x parameter memory for training

### Accuracy Benchmarks
- **MNIST**: 99%+ (simple CNN)
- **CIFAR-10**: 95%+ (ResNet)
- **ImageNet**: 80%+ (EfficientNet)
- **BERT**: 90%+ (GLUE benchmark)

---

## 14. 🔗 Quick Reference Links

### Activation Functions
- Use **ReLU** by default for hidden layers
- Use **Leaky ReLU** if ReLU neurons die  
- Use **Sigmoid** for binary output
- Use **Softmax** for multi-class output
- Use **Linear** for regression output

### Optimizers
- Start with **Adam** (lr=0.001)
- Use **SGD + momentum** for best final performance
- Use **AdamW** for transformers
- Use **RMSprop** for RNNs

### Regularization
- Always use **early stopping**
- Use **dropout** (0.5 dense, 0.2 conv)
- Use **L2** regularization (0.001-0.01)
- Use **data augmentation** for images

### Architecture
- **MLP**: Tabular data
- **CNN**: Images, spatial data
- **RNN/LSTM**: Sequences, time series
- **Transformer**: Modern NLP, some vision

---

## 💡 Pro Tips

1. **Always start simple**: Single layer → Add complexity gradually
2. **Monitor train/val curves**: Most important debugging tool
3. **Use transfer learning**: Don't train from scratch if pre-trained available
4. **Save best model**: Use validation loss, not training loss
5. **Reproducibility**: Set random seeds for debugging
6. **Baseline first**: Compare against simple models (logistic regression)
7. **Feature engineering**: Still matters even in deep learning
8. **GPU utilization**: Batch size affects GPU efficiency
9. **Learning rate**: Most important hyperparameter to tune
10. **Patience**: Deep learning takes time; don't expect immediate results

---

**🎯 Remember**: Neural networks are function approximators. With enough data and compute, they can learn complex patterns. The art is in the architecture design and training process!
