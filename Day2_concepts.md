# Day 2: Classical ML Algorithms Deep Dive

## 1. Decision Trees, Random Forests, Gradient Boosting

### Decision Trees
**What it is:** A tree structure where each node asks a yes/no question about a feature, splitting data until reaching a prediction at the leaves.

**How it works:**
- Starts with all data at the root
- Finds the best feature and split point that separates classes/reduces variance
- Recursively splits until stopping criteria (max depth, min samples)
- Uses metrics like Gini impurity (classification) or MSE (regression)

**Pros:** Interpretable, handles non-linear relationships, no feature scaling needed
**Cons:** Overfits easily, unstable (small data changes cause big tree changes), high variance

### Random Forests
**What it is:** Ensemble of decision trees trained on random subsets of data and features.

**How it works:**
- Creates N trees, each on a bootstrap sample (sampling with replacement)
- Each split considers only a random subset of features (√n for classification, n/3 for regression)
- Final prediction: majority vote (classification) or average (regression)

**Pros:** Reduces overfitting, handles high-dimensional data, robust to outliers
**Cons:** Less interpretable, slower prediction, needs more memory

### Gradient Boosting
**What it is:** Sequential ensemble where each tree corrects errors of previous trees.

**How it works:**
- Train first tree on original data
- Calculate residuals (actual - predicted)
- Train next tree to predict these residuals
- Add new tree's predictions (weighted by learning rate)
- Repeat, gradually reducing errors

**Pros:** High accuracy, handles complex patterns, flexible loss functions
**Cons:** Prone to overfitting, requires careful tuning, slower training

---

## 2. When to Use Tree-Based vs Linear Models

### Use Tree-Based Models When:
- **Non-linear relationships** between features and target
- **Feature interactions** are important (e.g., age matters differently for different genders)
- **Mixed data types** (categorical + numerical)
- **No feature engineering** time (trees find patterns automatically)
- **Outliers present** (trees are robust)
- **Interpretability** needed at feature level (feature importance)

**Examples:** Fraud detection, customer churn, ranking problems

### Use Linear Models When:
- **Linear relationships** or you can engineer features to be linear
- **High-dimensional sparse data** (text, genomics) - regularization works well
- **Speed critical** (linear models predict faster)
- **Extrapolation needed** (trees can't predict outside training range)
- **Memory constrained** (smaller models)
- **Interpretability** needed at coefficient level (understand direction/magnitude)

**Examples:** Price prediction with known relationships, text classification, online learning

---

## 3. XGBoost, LightGBM, CatBoost Comparison

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| **Split Strategy** | Level-wise (layer by layer) | Leaf-wise (best leaf first) | Symmetric trees |
| **Speed** | Moderate | Fastest | Moderate-Fast |
| **Memory** | Higher | Lower | Moderate |
| **Categorical Features** | Manual encoding needed | Basic support | Native handling (best) |
| **Small Datasets** | Good | Can overfit | Best |
| **GPU Support** | Yes | Yes (best) | Yes |
| **Default Performance** | Good | Good | Best out-of-box |

**XGBoost:**
- Most mature, widely used
- Excellent regularization (L1, L2, gamma)
- Best for structured/tabular data competitions
- Use when: you need battle-tested, well-documented solution

**LightGBM:**
- Fastest training on large datasets (>10K rows)
- Memory efficient (histogram-based)
- Use when: speed critical, large datasets, high-dimensional

**CatBoost:**
- Handles categorical features automatically (target encoding)
- Less tuning required (best defaults)
- Ordered boosting reduces overfitting
- Use when: many categorical features, need quick baseline, less tuning time

---

## 4. Feature Importance and Interpretability

### Types of Feature Importance:

**1. Gain/Split-based (built-in):**
- Measures average gain when feature is used to split
- Fast but biased toward high-cardinality features
```python
xgb_model.feature_importances_
```

**2. Permutation Importance:**
- Shuffle feature values, measure performance drop
- More reliable, model-agnostic
- Slower but captures true importance
```python
from sklearn.inspection import permutation_importance
perm_imp = permutation_importance(model, X_val, y_val)
```

**3. SHAP (SHapley Additive exPlanations):**
- Shows how each feature contributes to individual predictions
- Based on game theory, theoretically sound
- Provides both global and local interpretability
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
```

### Interpretability Best Practices:
- Use multiple methods (don't rely on one)
- Check for correlated features (importance splits between them)
- Validate with domain experts
- Use partial dependence plots to see feature effects

---

## 5. Hyperparameter Tuning Strategies

### Key Parameters by Algorithm:

**Random Forest:**
- `n_estimators`: More trees = better (100-500), diminishing returns
- `max_depth`: Control overfitting (10-50)
- `min_samples_split`: Minimum samples to split (2-20)
- `max_features`: Features per split (sqrt(n) for classification)

**Gradient Boosting (XGB/LGBM/CatBoost):**
- `learning_rate`: Slower = better but more trees needed (0.01-0.3)
- `n_estimators`: With early stopping (500-5000)
- `max_depth`: Shallow trees for boosting (3-10)
- `subsample`: Row sampling (0.6-1.0)
- `colsample_bytree`: Feature sampling (0.6-1.0)
- `reg_alpha/lambda`: L1/L2 regularization

### Tuning Approaches:

**1. Grid Search:**
- Exhaustive search over parameter grid
- Use for small grids (<100 combinations)
```python
from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth': [3,5,7], 'learning_rate': [0.01, 0.1]}
grid = GridSearchCV(model, param_grid, cv=5)
```

**2. Random Search:**
- Random sampling from parameter distributions
- Better for large spaces, finds good solutions faster
```python
from sklearn.model_selection import RandomizedSearchCV
param_dist = {'max_depth': range(3,15), 'learning_rate': uniform(0.01, 0.3)}
random = RandomizedSearchCV(model, param_dist, n_iter=50)
```

**3. Bayesian Optimization:**
- Intelligently samples based on previous results
- Best for expensive models, fewer iterations needed
```python
from skopt import BayesSearchCV
opt = BayesSearchCV(model, search_spaces, n_iter=30)
```

**Strategy:**
1. Start with defaults
2. Tune learning_rate + n_estimators (with early stopping)
3. Tune tree-specific parameters (depth, samples)
4. Fine-tune regularization
5. Use cross-validation (5-fold minimum)

---

## 6. Handling Imbalanced Data in Production

### Problem:
When one class is rare (fraud: 0.1%, not fraud: 99.9%), model predicts majority class and gets 99.9% accuracy but fails on minority class.

### Solutions:

**1. Evaluation Metrics:**
- Don't use accuracy!
- Use: Precision-Recall AUC, F1-score, Matthews Correlation Coefficient
- Focus on recall for minority class (fraud detection)
- Focus on precision when false positives are costly

**2. Data-Level Approaches:**

**Undersampling Majority:**
- Remove majority samples to balance
- Pro: Faster training, less memory
- Con: Loses information
- Best for: Very large datasets

**Oversampling Minority:**
- Duplicate minority samples
- Pro: No data loss
- Con: Overfitting risk
- Use with cross-validation properly

**SMOTE (Synthetic Minority Oversampling):**
- Creates synthetic samples between minority examples
- Pro: Adds diversity, reduces overfitting
- Con: Can create noise in overlapping regions
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.5)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

**3. Algorithm-Level Approaches:**

**Class Weights:**
- Give higher penalty for misclassifying minority class
- Built into most algorithms
```python
model = XGBClassifier(scale_pos_weight=99)  # ratio of neg/pos
```

**4. Threshold Adjustment:**
- Default: predict 1 if probability > 0.5
- Adjust based on business cost/benefit
```python
# Find optimal threshold
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = [f1_score(y_val, pred_proba > t) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
```

**5. Ensemble Approaches:**
- BalancedBagging: Bootstrap minority class in each tree
- EasyEnsemble: Multiple undersampled datasets, train separate models
- Train multiple models on different balanced samples

**Production Considerations:**
- Monitor class distribution drift
- Retrain with recent balanced data
- Use appropriate metrics in monitoring
- Consider business costs in threshold selection
- A/B test different balancing strategies

---

## 7. Interview Focus: Algorithm Selection Trade-offs

### Common Interview Questions & Answers:

**Q: Why would you choose Random Forest over Gradient Boosting?**
A: Random Forest when:
- Need parallelization (trains trees independently)
- Less prone to overfitting on small datasets
- Want more robustness to outliers
- Less tuning time available
- Need good baseline quickly

**Q: When would linear models outperform tree-based models?**
A:
- High-dimensional sparse data (text: 100K features, 1M samples)
- Need to extrapolate beyond training data range
- Production latency critical (linear models 10-100x faster)
- Need coefficient interpretability for regulatory compliance
- Online learning required (partial_fit)

**Q: How do you handle categorical features with high cardinality (100K+ categories)?**
A:
1. CatBoost (best) - native handling
2. Target encoding with regularization (avoid leakage)
3. Frequency encoding
4. Embedding approaches
5. Feature hashing for linear models
6. Avoid one-hot encoding (creates sparse 100K dimensions)

**Q: Production model is overfitting. What do you try first?**
A: Systematic approach:
1. Add more diverse training data (best solution)
2. Increase regularization (lambda, alpha)
3. Reduce model complexity (max_depth, num_leaves)
4. Increase min_samples_split/min_child_weight
5. Add feature selection
6. Use ensemble methods
7. Cross-validate rigorously

**Q: How do you choose between XGBoost and LightGBM?**
A: Consider:
- Dataset size: LightGBM for >10K rows (faster)
- Memory: LightGBM uses less
- Categorical features: CatBoost > LightGBM > XGBoost
- Small datasets: XGBoost or CatBoost (less overfitting)
- GPU available: LightGBM has best GPU support
- **Benchmark both** - performance varies by dataset

**Q: Explain bias-variance tradeoff in tree models.**
A:
- Single tree: Low bias (fits complex patterns), high variance (unstable)
- Random Forest: Keeps low bias, reduces variance through averaging
- Gradient Boosting: Reduces bias sequentially, controls variance with learning_rate + regularization
- Tuning: Depth/leaves control complexity (bias), ensembling reduces variance

### Key Takeaways for Interviews:
- Always mention trade-offs (no free lunch)
- Connect to business context (speed vs accuracy)
- Show awareness of production constraints
- Discuss validation strategy
- Mention monitoring and maintenance

---

## Practical Tips:
- Start with LightGBM/CatBoost for quick baseline
- Use cross-validation religiously (never trust single split)
- Feature engineering often beats algorithm choice
- Simpler models (fewer parameters) preferred if performance close
- Always validate on holdout set with business metric

---

**Tomorrow: Deep Learning Fundamentals - Neural Networks & Backpropagation**
