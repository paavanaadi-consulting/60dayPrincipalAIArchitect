# Day 7: Advanced ML Topics

## Overview

Advanced ML topics represent specialized techniques for specific problem domains. As a Principal AI Architect, you must know **when and why** to apply each technique, not just how they work.

**Why This Day Matters:**
- Interviews often ask "Which approach for X problem?"
- Many enterprise problems require specialized architectures
- Knowing trade-offs demonstrates architectural maturity
- These topics appear in 60% of senior ML interviews

**Key Principle:** Don't use advanced techniques when simple methods work. Always start simple, add complexity only when justified.

---

## 1. Time Series Forecasting

### What Makes Time Series Different?

**Time series:** Ordered sequence of observations over time
- Stock prices, sensor data, sales, website traffic
- **Key property:** Temporal dependency (past influences future)

**Why traditional ML doesn't work well:**
```
Traditional ML: Assumes samples are independent (IID)
Reality: Today's sales depend on yesterday's sales
Example: Black Friday spike affects next week's inventory
```

### Common Time Series Patterns

**1. Trend:** Long-term increase or decrease
```
Example: Company revenue growing 20% year-over-year
Visualization: Upward slope over time
```

**2. Seasonality:** Repeating patterns at fixed intervals
```
Example: Ice cream sales peak every summer
Patterns: Daily, weekly, monthly, yearly
```

**3. Cyclic:** Fluctuations without fixed period
```
Example: Economic recessions (not regular intervals)
Difference from seasonality: Variable duration
```

**4. Noise:** Random variation
```
Unpredictable fluctuations
Cannot be modeled
```

### Approach 1: ARIMA (Classical Statistics)

**ARIMA = AutoRegressive Integrated Moving Average**

**Components:**

**AR (AutoRegressive):** Predict from past values
```
y_t = c + φ₁·y_{t-1} + φ₂·y_{t-2} + ... + φ_p·y_{t-p} + ε_t

Current value depends on p previous values
Example: Tomorrow's temperature based on last 3 days
```

**I (Integrated):** Make series stationary (remove trend)
```
Differencing: y'_t = y_t - y_{t-1}
Do d times until stationary
Stationary: Mean and variance constant over time
```

**MA (Moving Average):** Predict from past errors
```
y_t = μ + ε_t + θ₁·ε_{t-1} + θ₂·ε_{t-2} + ... + θ_q·ε_{t-q}

Current value depends on q previous forecast errors
```

**ARIMA(p,d,q):**
- p: AR order (how many past values)
- d: Differencing order (make stationary)
- q: MA order (how many past errors)

**Example: ARIMA(2,1,1)**
```
Use 2 past values, difference once, 1 past error
```

**Pros:**
```
- Well-understood, interpretable
- Works with small data (<1000 points)
- Fast to train and predict
- No feature engineering needed
- Confidence intervals (uncertainty quantification)
```

**Cons:**
```
- Assumes linear relationships
- Cannot handle multiple variables easily
- Manual parameter selection (p, d, q)
- Doesn't capture complex patterns
- No external features (holidays, promotions)
```

**When to use:**
```
- Short-term forecasting (hours to weeks)
- Univariate data (single time series)
- Linear trends and patterns
- Need interpretability
- Limited data (<10K points)
```

### Approach 2: Prophet (Facebook)

**Designed for business time series (sales, revenue, traffic)**

**Model:**
```
y(t) = g(t) + s(t) + h(t) + ε_t

g(t): Trend (piecewise linear or logistic)
s(t): Seasonality (Fourier series)
h(t): Holidays/events
ε_t: Error term
```

**Key Features:**

**1. Automatic seasonality detection:**
```
Yearly: Captures annual patterns (summer/winter)
Weekly: Day-of-week effects (weekends vs weekdays)
Daily: Hour-of-day patterns (morning/evening)
```

**2. Holiday effects:**
```
Can specify: Christmas, Black Friday, Super Bowl
Model learns impact automatically
Accounts for irregular events
```

**3. Trend changepoints:**
```
Automatically detects when trend changes
Example: Product launch, market shift
No manual intervention needed
```

**4. Robust to missing data:**
```
Handles gaps naturally
Doesn't require complete series
```

**Pros:**
```
- Easy to use (minimal tuning)
- Handles multiple seasonalities
- Interpretable components (trend, seasonality)
- Works with messy data (missing values, outliers)
- Supports external regressors (promotions, weather)
- Fast (seconds to minutes)
```

**Cons:**
```
- Additive model (doesn't capture multiplicative effects)
- Less accurate than deep learning for complex patterns
- Assumes piecewise linear trend
- Not ideal for high-frequency data (milliseconds)
```

**When to use:**
```
- Business forecasting (sales, revenue, metrics)
- Multiple seasonalities (yearly + weekly + daily)
- Irregular events (holidays, promotions)
- Need quick prototype
- Non-experts doing forecasting (data analysts)
```

### Approach 3: LSTM (Deep Learning)

**LSTM = Long Short-Term Memory (type of RNN)**

**Why for time series?**
```
Can learn complex non-linear patterns
Handles long-term dependencies
Captures multiple variables (multivariate)
Learns interactions between features
```

**Architecture:**
```
Input: Historical window (e.g., past 30 days)
       Shape: [batch_size, sequence_length, features]
       Example: [32, 30, 5] (32 samples, 30 days, 5 features)
       
LSTM layers: 2-3 stacked layers
       Each layer: 64-256 hidden units
       Learns temporal patterns
       
Dense layer: Final prediction
       Output: Next value(s)
```

**Training:**
```
Sliding window approach:
Days 1-30 → Predict day 31
Days 2-31 → Predict day 32
...

Loss: MSE (Mean Squared Error)
Optimizer: Adam
Epochs: 50-100
```

**Multivariate forecasting:**
```
Input features:
- Past sales
- Price
- Promotions (binary)
- Day of week (one-hot)
- Month (one-hot)
- Weather (temperature, rain)

LSTM learns how all features interact
```

**Pros:**
```
- Captures complex non-linear patterns
- Handles multiple input features (multivariate)
- Long-term dependencies (months/years)
- Can model interactions between variables
- State-of-the-art accuracy for complex series
```

**Cons:**
```
- Needs lots of data (10K+ points)
- Black box (not interpretable)
- Slow to train (hours on GPU)
- Hyperparameter tuning required
- Overfitting risk with small data
- No uncertainty estimates (without Bayesian methods)
```

**When to use:**
```
- Large dataset (>10K time points)
- Complex patterns (non-linear, interactions)
- Multiple input features (multivariate)
- High-frequency data (seconds, minutes)
- Accuracy critical (business impact high)
- Have ML expertise and infrastructure
```

### Approach Comparison

| Aspect | ARIMA | Prophet | LSTM |
|--------|-------|---------|------|
| **Data Size** | <1K points | 1K-100K | >10K |
| **Complexity** | Linear | Additive | Non-linear |
| **Features** | Univariate | Univariate + regressors | Multivariate |
| **Seasonality** | Manual | Automatic (multiple) | Learned |
| **Interpretability** | High | Medium | Low |
| **Speed** | Seconds | Seconds | Hours |
| **Accuracy** | Low | Medium | High |
| **Best For** | Simple forecasts | Business metrics | Complex patterns |

### Interview Decision Framework

**Question:** "We need to forecast daily website traffic. Which approach?"

**Ask clarifying questions:**
```
1. How much historical data? 
   <6 months → Prophet
   >2 years → LSTM

2. Any external factors?
   Marketing campaigns → Prophet (handles regressors)
   Weather, events → LSTM (multivariate)

3. Accuracy requirements?
   General trends → Prophet
   Need <5% error → LSTM

4. Who will maintain?
   Data analysts → Prophet (easy)
   ML engineers → LSTM (complex)

5. Latency requirements?
   Daily batch → Any approach
   Real-time → ARIMA/Prophet (fast)
```

**Recommended approach:**
```
Start: Prophet (quick prototype)
  ↓
Evaluate: Does it meet accuracy target?
  Yes → Deploy Prophet (simple wins)
  No → Try LSTM (if have data and expertise)
  ↓
Production: Ensemble (Prophet + LSTM)
  Combine predictions (weighted average)
  Often 5-10% better than either alone
```

---

## 2. Recommendation Systems

### The Recommendation Problem

**Goal:** Suggest items user might like

**Examples:**
- Netflix: Recommend movies
- Amazon: Recommend products  
- Spotify: Recommend songs
- LinkedIn: Recommend jobs

**Formulation:**
```
Users: U = {u₁, u₂, ..., u_m}
Items: I = {i₁, i₂, ..., i_n}
Ratings: R (m × n matrix, sparse)

Task: Predict R[u, i] for unobserved (u, i) pairs
```

### Approach 1: Collaborative Filtering

**Idea:** Users with similar tastes like similar items

**Two types:**

**User-based CF:**
```
"Users similar to you also liked..."

Steps:
1. Find users similar to target user
   Similarity: Cosine, Pearson correlation
   
2. Get items those similar users liked
   
3. Recommend top-N items

Example:
You liked: [Movie A, Movie B, Movie C]
Similar user liked: [Movie A, Movie B, Movie D]
Recommend: Movie D
```

**Item-based CF:**
```
"People who liked X also liked Y"

Steps:
1. Find items similar to items user liked
   Similarity: Based on user ratings
   
2. Recommend similar items

Example:
You liked: Movie A
Similar to Movie A: Movie X (cosine similarity = 0.85)
Recommend: Movie X

Why item-based often better:
- Items change less than user preferences
- Pre-compute item similarities (fast)
- More stable over time
```

**Matrix Factorization (Modern CF):**
```
Decompose rating matrix:
R ≈ U × I^T

U: User embeddings (m × k)
I: Item embeddings (n × k)
k: Latent factors (typically 50-200)

Each user: Vector in k-dimensional space
Each item: Vector in k-dimensional space
Prediction: Dot product of user and item vectors

Example (k=3):
user_vector = [0.8, -0.3, 0.5]  (likes action, dislikes romance, likes sci-fi)
item_vector = [0.7, -0.2, 0.6]  (action movie)
rating = 0.8*0.7 + (-0.3)*(-0.2) + 0.5*0.6 = 0.92 (predicted high rating)

Training: Minimize MSE between actual and predicted ratings
Algorithm: Alternating Least Squares (ALS) or SGD
```

**Pros:**
```
- No item features needed (just user-item interactions)
- Discovers latent patterns (genres, styles)
- Works across different item types
- Personalized (different for each user)
```

**Cons:**
```
- Cold start: New users/items have no ratings
- Sparsity: Most users rate <1% of items
- Popularity bias: Recommends popular items
- Cannot explain recommendations
```

### Approach 2: Content-Based Filtering

**Idea:** Recommend items similar to what user liked before

**Process:**
```
1. Extract item features
   Movie: Genre, director, actors, year
   Product: Category, brand, price range, specs
   
2. Build user profile
   Aggregate features of items user liked
   User profile = weighted average of liked item features
   
3. Find items similar to user profile
   Cosine similarity between user profile and item features
   
4. Recommend top-N similar items
```

**Example:**
```
User watched:
- Star Wars (Sci-Fi, Action, George Lucas)
- Lord of the Rings (Fantasy, Action, Peter Jackson)

User profile: [Action: 1.0, Sci-Fi: 0.5, Fantasy: 0.5, ...]

Candidate movie:
- Avatar (Sci-Fi, Action, James Cameron)
- Feature vector: [Action: 1.0, Sci-Fi: 1.0, ...]

Similarity: Cosine(user_profile, avatar_features) = 0.88 (high)
Recommend: Avatar
```

**Pros:**
```
- No cold start for items (just need features)
- Explainable ("Recommended because you liked X")
- No need for other users' data (privacy-friendly)
- Can recommend niche items (not just popular)
```

**Cons:**
```
- Requires feature engineering (domain knowledge)
- Limited diversity (only similar items)
- Cold start for new users (no history)
- Cannot discover new interests
```

### Approach 3: Hybrid Systems

**Combine collaborative filtering + content-based**

**Methods:**

**1. Weighted hybrid:**
```
score = α × CF_score + (1-α) × CB_score

α = 0.7 (70% collaborative, 30% content-based)

Tune α based on validation set
```

**2. Switching hybrid:**
```
IF user has >10 ratings:
    Use collaborative filtering (enough data)
ELSE:
    Use content-based (cold start)
```

**3. Feature combination:**
```
Neural network with both:
- User/item IDs → Embeddings (collaborative)
- Item features → Dense layer (content-based)

Concatenate and predict rating
```

**4. Cascade:**
```
Stage 1: Collaborative filtering (get 1000 candidates)
Stage 2: Content-based ranking (rank top 100)
Stage 3: Diversity filter (final top 10)

Fast retrieval → Precise ranking → Diverse results
```

### Modern Approaches: Neural Collaborative Filtering

**Deep learning for recommendations**

**Architecture:**
```
User ID → Embedding (128-dim)
Item ID → Embedding (128-dim)
    ↓
Concatenate
    ↓
Dense layer (256 units, ReLU)
    ↓
Dense layer (128 units, ReLU)
    ↓
Dense layer (64 units, ReLU)
    ↓
Output (1 unit, sigmoid)
    ↓
Predicted rating (0-1)
```

**Why better than matrix factorization:**
```
- Non-linear interactions (not just dot product)
- Can incorporate side features (age, location, time)
- Captures complex patterns
- State-of-the-art accuracy
```

**Real-world system (e.g., Netflix):**
```
Stage 1: Candidate generation (retrieve 1000s)
  - Collaborative filtering
  - Popular items
  - Recently released
  
Stage 2: Ranking (top 100)
  - Neural network with features:
    • User: Demographics, viewing history
    • Item: Genre, actors, ratings
    • Context: Time of day, device
  - Predict: P(user will watch)
  
Stage 3: Re-ranking (final 10)
  - Diversity (not all same genre)
  - Freshness (recent releases)
  - Business rules (promote originals)
```

### Evaluation Metrics

**Offline metrics:**
```
Accuracy: RMSE, MAE (rating prediction)
Ranking: Precision@K, Recall@K, NDCG

Example (Precision@10):
Recommended 10 items → User liked 3
Precision@10 = 3/10 = 0.3
```

**Online metrics (A/B test):**
```
Click-through rate (CTR)
Conversion rate
Watch time (video)
User engagement
Revenue impact
```

**Trade-offs:**
```
Accuracy vs Diversity:
- High accuracy → Recommends safe/popular items
- High diversity → Broader exploration
- Need balance

Personalization vs Serendipity:
- Personalized → More of same
- Serendipitous → Surprising recommendations
```

### Interview Decision Framework

**Question:** "Design a recommendation system for e-commerce"

**Clarifying questions:**
```
1. User base size?
   <10K users → Content-based (simple)
   >1M users → Collaborative filtering (leverage data)

2. Item catalog size?
   <1K items → Simple similarity
   >100K items → Need efficient retrieval (ANN)

3. Cold start frequency?
   High (new products daily) → Hybrid (content-based for new)
   Low → Pure collaborative

4. Data available?
   Just clicks → Implicit feedback
   Ratings → Explicit feedback

5. Business constraints?
   Promote new items → Boost new in ranking
   Regional inventory → Filter by availability
```

**Recommended architecture:**
```
Two-stage:

Stage 1: Candidate Generation (1000 items)
  - Collaborative filtering (ALS or embeddings)
  - Content-based for new items
  - Popular items (fallback)
  
Stage 2: Ranking (Top 10)
  - LightGBM/XGBoost with features:
    • User: Purchase history, demographics
    • Item: Category, price, rating, inventory
    • Context: Time, device, search query
  - Re-rank for diversity and business rules
  
Evaluation:
  - Offline: NDCG@10 (ranking quality)
  - Online A/B: CTR, conversion rate
```

---

## 3. Anomaly Detection

### What is Anomaly Detection?

**Definition:** Identify data points that deviate significantly from normal

**Applications:**
- Fraud detection (credit cards, insurance)
- Network intrusion detection
- Manufacturing defects
- Medical diagnosis (abnormal test results)
- System monitoring (server crashes, performance issues)

### Challenge: Imbalanced Data

```
Normal: 99.9%
Anomalies: 0.1%

Standard classification doesn't work:
Model predicting "normal" for everything = 99.9% accuracy!
But useless (misses all anomalies)
```

### Approach 1: Statistical Methods

**Z-Score (Univariate):**
```
Assume normal distribution
Anomaly: More than 3 standard deviations from mean

z = (x - μ) / σ

If |z| > 3: Anomaly

Example:
Response time: Mean = 100ms, Std = 20ms
Observation: 180ms
z = (180 - 100) / 20 = 4 → Anomaly!
```

**Pros:**
```
- Simple, fast
- Interpretable (how many std deviations)
- No training needed
```

**Cons:**
```
- Assumes normal distribution
- Univariate only (one variable)
- Fixed threshold (not adaptive)
```

**Multivariate Gaussian:**
```
Model joint distribution of all features
Anomaly: Low probability under distribution

P(x) < ε → Anomaly

Handles correlations between features
```

### Approach 2: Isolation Forest

**Idea:** Anomalies are easier to isolate (fewer splits needed)

**Algorithm:**
```
Build random trees:
1. Randomly select feature
2. Randomly select split value
3. Split data
4. Repeat until each point isolated

Anomaly score:
- Normal points: Many splits needed (deep in tree)
- Anomalies: Few splits needed (shallow in tree)

Average depth across trees → Anomaly score
```

**Pros:**
```
- Fast (linear time)
- Works with high-dimensional data
- No assumption about distribution
- Handles mixed data types
```

**Cons:**
```
- Cannot explain why point is anomaly
- Needs tuning (contamination parameter)
```

**When to use:**
```
- High-dimensional data (100+ features)
- Real-time detection (fast)
- Unknown distribution
```

### Approach 3: Autoencoders (Deep Learning)

**Idea:** Normal data reconstructs well, anomalies reconstruct poorly

**Architecture:**
```
Input (n features)
    ↓
Encoder: n → k (compress)
    ↓
Latent representation (k < n)
    ↓
Decoder: k → n (reconstruct)
    ↓
Output (n features)

Loss: MSE(input, output)
```

**Training:**
```
Train on normal data only
Learn to compress and reconstruct normal patterns
```

**Anomaly detection:**
```
Reconstruction error = MSE(input, output)

If error > threshold → Anomaly

Example:
Normal transaction: Error = 0.02
Fraudulent: Error = 0.87 (doesn't match learned patterns)
```

**Pros:**
```
- Handles complex patterns
- Unsupervised (no labels needed)
- Captures non-linear relationships
- Works with high-dimensional data
```

**Cons:**
```
- Needs lots of data
- Slow to train
- Threshold selection tricky
- May not generalize to new anomaly types
```

**Variants:**

**Variational Autoencoder (VAE):**
```
Learns distribution (not just reconstruction)
Can sample new normal data
Better for novelty detection
```

**LSTM Autoencoder (Time Series):**
```
For temporal data (sensor readings, logs)
Encoder: LSTM processes sequence
Decoder: LSTM reconstructs sequence
Detects anomalous patterns over time
```

### Approach 4: One-Class SVM

**Idea:** Learn boundary around normal data

**Algorithm:**
```
Find hyperplane that separates normal data from origin
Maximize margin (like SVM)
Points far from hyperplane → Anomalies
```

**Pros:**
```
- Theory-backed (optimization)
- Works with non-linear kernels (RBF)
- Good for small datasets
```

**Cons:**
```
- Slow on large data
- Sensitive to hyperparameters (kernel, nu)
- Not interpretable
```

### Comparison & Decision Framework

| Method | Data Size | Interpretability | Speed | Best For |
|--------|-----------|------------------|-------|----------|
| **Z-Score** | Any | High | Fastest | Simple, univariate |
| **Isolation Forest** | 1K-1M | Low | Fast | High-dimensional |
| **Autoencoder** | >10K | Low | Slow | Complex patterns |
| **One-Class SVM** | <10K | Low | Medium | Small, clean data |

**Interview question:** "Detect fraudulent credit card transactions"

**Analysis:**
```
Data: 1M transactions/day, 0.1% fraud
Features: Amount, merchant, location, time, user history

Challenges:
- Extreme imbalance (0.1% fraud)
- Need real-time detection (<100ms)
- Fraud patterns evolve (adversarial)

Recommended approach:
1. Isolation Forest (fast, handles high-dim)
   - Online learning (update daily)
   - Score in real-time (<10ms)
   
2. Autoencoder for complex patterns
   - Batch update (retrain weekly)
   - Higher accuracy for new fraud types
   
3. Ensemble: Combine both
   - If either flags → Review
   - Reduces false negatives
   
4. Human-in-loop:
   - Threshold tuning (balance precision/recall)
   - Feedback loop (label flagged transactions)
   - Continuous learning
```

---

## 4. Reinforcement Learning Basics

### What is RL?

**Definition:** Agent learns by interacting with environment

**Components:**
```
Agent: Learner/decision maker
Environment: External world
State (s): Current situation
Action (a): What agent can do
Reward (r): Feedback from environment
Policy (π): Strategy (state → action mapping)

Goal: Maximize cumulative reward
```

**Difference from supervised learning:**
```
Supervised: Given (x, y) pairs, learn f(x) = y
RL: No labels! Learn from trial and error

Example:
Supervised: Given moves in chess games, predict next move
RL: Play chess, win/lose is only feedback
```

### Key Concepts

**Markov Decision Process (MDP):**
```
(S, A, P, R, γ)

S: State space (all possible states)
A: Action space (all possible actions)
P: Transition probability P(s'|s, a)
R: Reward function R(s, a, s')
γ: Discount factor (0 ≤ γ ≤ 1)

Future reward discounted:
Return = r₀ + γr₁ + γ²r₂ + ... 
```

**Value function:**
```
V(s): Expected return starting from state s
Q(s, a): Expected return taking action a in state s

Goal: Learn V or Q to choose best actions
```

**Exploration vs Exploitation:**
```
Exploration: Try new actions (learn more)
Exploitation: Choose best known action (maximize reward)

Trade-off: ε-greedy strategy
  With probability ε: Explore (random action)
  With probability 1-ε: Exploit (best action)
  
Start: ε = 1.0 (full exploration)
End: ε = 0.1 (mostly exploitation)
```

### Approach 1: Q-Learning

**Idea:** Learn Q(s, a) for all state-action pairs

**Q-Table:**
```
Rows: States
Columns: Actions
Values: Q(s, a)

Example (Grid World):
      Left  Right  Up    Down
s₁    0.5   0.8   0.2   0.3
s₂    0.1   0.3   0.9   0.2
...
```

**Update rule:**
```
Q(s, a) ← Q(s, a) + α[r + γ max Q(s', a') - Q(s, a)]
                         a'

α: Learning rate (how much to update)
γ: Discount factor (value future rewards)
r: Immediate reward
max Q(s', a'): Best future value
```

**Algorithm:**
```
Initialize Q(s, a) randomly
For each episode:
  s = initial state
  While not terminal:
    Choose action: a = ε-greedy(Q, s)
    Take action a, observe r, s'
    Update: Q(s, a) using update rule
    s ← s'
```

**Pros:**
```
- Simple, well-understood
- Guaranteed convergence (with conditions)
- Off-policy (learn from any experience)
```

**Cons:**
```
- Only works for discrete, small state/action spaces
- Q-table grows exponentially (states × actions)
- Cannot generalize to unseen states
```

**When to use:**
```
- Small state space (<10K states)
- Small action space (<100 actions)
- Educational/simple problems
```

### Approach 2: Deep Q-Network (DQN)

**Idea:** Use neural network to approximate Q(s, a)

**Why:**
```
Q-table doesn't scale:
- Atari game: 256^(210×160) states (pixels)
- Cannot store table

Solution: Function approximation
Q(s, a) ≈ Neural_Network(s, a)
```

**Architecture:**
```
Input: State (e.g., 84×84×4 image frames)
    ↓
Conv layers (learn visual features)
    ↓
Dense layers
    ↓
Output: Q(s, a) for each action

Example (Atari):
Input: [84, 84, 4] (grayscale, 4 frames stacked)
Conv1: 32 filters, 8×8, stride 4
Conv2: 64 filters, 4×4, stride 2
Conv3: 64 filters, 3×3, stride 1
Dense: 512 units
Output: Q-values for each action (e.g., 18 actions)
```

**Key innovations:**

**1. Experience Replay:**
```
Problem: Online learning unstable (correlated samples)
Solution: Store experiences in buffer, sample randomly

Buffer: (s, a, r, s') tuples (capacity: 1M)
Training: Sample mini-batch (32) randomly

Benefits:
- Breaks correlation (more stable)
- Reuse data (sample efficient)
- Avoid catastrophic forgetting
```

**2. Target Network:**
```
Problem: Q-network updates using its own predictions (unstable)
Solution: Use separate target network (frozen)

Main network: Updated every step
Target network: Copied from main every 10K steps

Update:
Q(s, a) ← r + γ max Q_target(s', a')
                 a'

More stable training
```

**Pros:**
```
- Scales to large state spaces (images, continuous)
- Generalizes to unseen states
- Human-level performance on Atari games
```

**Cons:**
```
- Sample inefficient (millions of interactions)
- Unstable training (hyperparameter sensitive)
- Still only discrete actions
- Overestimation bias (max operator)
```

### Approach 3: Policy Gradients

**Idea:** Directly optimize policy π(a|s)

**Difference from Q-learning:**
```
Q-learning: Learn value function → Derive policy
Policy gradient: Learn policy directly

Advantage: Can handle continuous actions
Example: Robot arm control (infinite action space)
```

**REINFORCE algorithm:**
```
Policy network: π(a|s; θ)
  Input: State
  Output: Probability distribution over actions

Training:
1. Collect episode: s₀, a₀, r₀, s₁, a₁, r₁, ..., s_T
2. Calculate returns: G_t = Σ γ^k r_{t+k}
3. Update policy:
   θ ← θ + α G_t ∇log π(a_t|s_t; θ)

Intuition: Increase probability of actions that led to high return
```

**Actor-Critic (combines value and policy):**
```
Actor: Policy network π(a|s)
Critic: Value network V(s)

Update:
- Critic learns V(s) (like Q-learning)
- Actor uses V(s) to guide policy

Benefits:
- Lower variance than REINFORCE
- Faster convergence
```

**Modern algorithms:**
```
PPO (Proximal Policy Optimization):
- Most popular
- Stable, sample efficient
- Used in ChatGPT (RLHF)

A3C (Asynchronous Actor-Critic):
- Parallel training
- Fast

SAC (Soft Actor-Critic):
- Continuous control
- Robotics
```

**Pros:**
```
- Continuous actions (robotics, control)
- Stochastic policies (exploration built-in)
- Better convergence properties
```

**Cons:**
```
- High variance (needs many samples)
- Sensitive to hyperparameters
- Slower than DQN for discrete actions
```

### RL Applications

**Robotics:**
```
Problem: Robot navigation, grasping, walking
State: Sensor readings, joint positions
Action: Motor commands (continuous)
Reward: Distance to goal, energy efficiency

Algorithm: SAC (continuous control)
```

**Game playing:**
```
AlphaGo: Defeated world champion (Go)
OpenAI Five: Defeated Dota 2 pros
Algorithm: Self-play + PPO
```

**Recommendation systems:**
```
State: User history, context
Action: Which item to recommend
Reward: Click, purchase, engagement

Algorithm: Contextual bandits, DQN
```

**Autonomous driving:**
```
State: Camera, lidar, sensors
Action: Steering, acceleration, braking
Reward: Safety, efficiency, comfort

Algorithm: Imitation learning + RL fine-tuning
```

### Interview Decision Framework

**Question:** "When would you use RL vs supervised learning?"

**Use RL when:**
```
1. Sequential decisions (action affects future state)
   Example: Game playing, robotics
   
2. No labeled data (only outcomes)
   Example: AlphaGo (no expert move labels)
   
3. Exploration needed (discover optimal strategy)
   Example: Resource allocation, scheduling
   
4. Delayed reward (action pays off later)
   Example: Drug dosage (effect after days)
```

**Use Supervised Learning when:**
```
1. Clear input-output pairs
   Example: Image classification
   
2. Static decisions (no sequential dependency)
   Example: Spam detection
   
3. Labeled data available
   Example: Medical diagnosis with historical cases
   
4. Immediate feedback
   Example: Credit scoring
```

**Hybrid approaches:**
```
Imitation learning: Learn from expert demonstrations (supervised) → Fine-tune with RL
Example: Self-driving cars (start with human driving data)

Offline RL: Learn from logged data (no environment interaction)
Example: Healthcare (can't experiment on patients)
```

---

## 5. Graph Neural Networks (GNNs)

### What is Graph Data?

**Graph:** Nodes connected by edges

**Examples:**
```
Social networks: Users (nodes), friendships (edges)
Molecules: Atoms (nodes), bonds (edges)
Road networks: Intersections (nodes), roads (edges)
Knowledge graphs: Entities (nodes), relationships (edges)
Citation networks: Papers (nodes), citations (edges)
```

**Why traditional ML doesn't work:**
```
Images: Grid structure (CNNs work)
Text: Sequence structure (RNNs/Transformers work)
Graphs: Irregular structure (need specialized approach)

Challenge: Nodes have variable number of neighbors
Cannot use fixed-size convolution
```

### Graph Properties

**Node features:**
```
Social network: Age, location, interests
Molecule: Atom type, charge
```

**Edge features:**
```
Social network: Friendship strength
Molecule: Bond type (single, double)
```

**Graph structure:**
```
Adjacency matrix A (n × n):
A[i,j] = 1 if edge between node i and j, 0 otherwise
```

### How GNNs Work

**Key idea: Message passing**

```
Each node aggregates information from neighbors

Iteration 1:
Node learns from immediate neighbors (1-hop)

Iteration 2:
Node learns from 2-hop neighbors

After k iterations:
Node has information from k-hop neighborhood
```

**Message passing:**
```
For each node v:
1. Collect messages from neighbors: {h_u for u in neighbors(v)}
2. Aggregate: aggregate({h_u}) → m_v
3. Update: h_v^{new} = UPDATE(h_v^{old}, m_v)

Aggregate functions:
- Sum: Σ h_u
- Mean: (1/|N|) Σ h_u
- Max: max{h_u}

Update functions:
- MLP: h_v^{new} = MLP([h_v^{old}, m_v])
- GRU: h_v^{new} = GRU(h_v^{old}, m_v)
```

**Example (social network):**
```
Node: Alice
Neighbors: Bob, Carol, Dave

Step 1: Get neighbor embeddings
  h_Bob, h_Carol, h_Dave

Step 2: Aggregate (mean)
  m_Alice = (h_Bob + h_Carol + h_Dave) / 3

Step 3: Update Alice's embedding
  h_Alice^{new} = σ(W · [h_Alice^{old}, m_Alice] + b)
  
After 2 layers:
  Alice's embedding contains info from 2-hop friends
  (friends of friends)
```

### GNN Architectures

**Graph Convolutional Network (GCN):**
```
H^{(l+1)} = σ(D^{-1/2} A D^{-1/2} H^{(l)} W^{(l)})

A: Adjacency matrix (with self-loops)
D: Degree matrix
H: Node features
W: Learnable weights

Normalized aggregation (considers node degree)
```

**GraphSAGE:**
```
Sample fixed number of neighbors (scalable)
Different aggregators: Mean, LSTM, Pool

Good for large graphs (billions of nodes)
```

**Graph Attention Network (GAT):**
```
Learn attention weights for neighbors

α_{ij} = attention(h_i, h_j)

Aggregate with learned weights:
h_i^{new} = Σ α_{ij} h_j

Benefits: Some neighbors more important than others
```

### GNN Tasks

**1. Node classification:**
```
Predict label for each node

Example: Social network
  Given: User features, friend connections
  Predict: User interests, demographics

Semi-supervised: Few labeled nodes, many unlabeled
GNN propagates labels through graph
```

**2. Link prediction:**
```
Predict if edge exists between two nodes

Example: Friend recommendation
  Given: Current friendships
  Predict: Who should be friends?

Approach: 
  - Get node embeddings (GNN)
  - Score pairs: similarity(h_i, h_j)
  - High score → Likely edge
```

**3. Graph classification:**
```
Predict label for entire graph

Example: Molecule property prediction
  Given: Molecular structure (graph)
  Predict: Toxicity, solubility

Approach:
  - Node embeddings (GNN)
  - Graph pooling (aggregate to single vector)
  - Classification layer
```

### Applications

**Drug discovery:**
```
Molecules as graphs
Predict: Binding affinity, toxicity, side effects

GNN learns molecular patterns
Faster than traditional chemistry
```

**Recommendation:**
```
User-item bipartite graph
Predict: User will like item?

GNN captures collaborative filtering
Better than matrix factorization
```

**Traffic prediction:**
```
Road network as graph
Predict: Traffic speed on each road

GNN propagates traffic patterns
Captures spatial dependencies
```

**Fraud detection:**
```
Transaction network
Detect: Fraud rings, money laundering

GNN finds suspicious patterns
Better than individual transaction analysis
```

### Interview Perspective

**Question:** "When would you use GNNs?"

**Use GNNs when:**
```
1. Data has graph structure
   Example: Social networks, molecules
   
2. Relationships matter
   Example: Fraud rings (fraudsters connected)
   
3. Need to propagate information
   Example: Semi-supervised learning on graphs
```

**Don't use GNNs when:**
```
1. No graph structure
   Example: Independent samples (images, text)
   
2. Graph is complete (all nodes connected)
   Example: Use standard neural network
   
3. Very large graphs (billions of nodes)
   Example: Sampling needed, may use simpler methods
```

---

## 6. Multi-Modal Learning

### What is Multi-Modal?

**Definition:** Learning from multiple types of data simultaneously

**Modalities:**
```
Vision: Images, videos
Language: Text, speech
Audio: Music, sounds
Sensors: Temperature, pressure
Structured: Tables, graphs
```

**Examples:**
```
Image + Text: Image captioning, visual question answering
Video + Audio: Action recognition, video understanding
Text + Audio: Speech recognition, emotion detection
Medical: Images + patient records + lab results
```

### Why Multi-Modal?

**1. More information:**
```
Single modality: Limited view
Multiple modalities: Complementary information

Example: Movie classification
  Video only: Visual scenes
  + Audio: Dialogue, music, sound effects
  + Text: Subtitles, metadata
  
Combined: Richer understanding
```

**2. Robustness:**
```
One modality may be noisy/missing
Others can compensate

Example: Video in dark (poor visual)
  Audio can still provide information
```

**3. Real-world scenarios:**
```
Humans use multiple senses
AI should too

Example: Autonomous driving
  Camera (vision) + Lidar (depth) + Radar (speed)
  Fusion improves safety
```

### Challenges

**1. Representation gap:**
```
Different modalities have different characteristics:
  Images: Spatial, grid structure
  Text: Sequential, discrete tokens
  Audio: Temporal, continuous waveform

Need unified representation
```

**2. Alignment:**
```
How to align different modalities?

Example: Image + Caption
  Which part of image corresponds to which words?
  
Solution: Attention mechanisms
```

**3. Fusion:**
```
When to combine modalities?
  Early fusion: Combine raw features
  Late fusion: Combine predictions
  Hybrid: Combine at multiple stages
```

**4. Missing modalities:**
```
Training: All modalities available
Inference: One modality missing (sensor failure)

Need robustness to missing modalities
```

### Approach 1: Early Fusion

**Idea:** Combine features from different modalities early

```
Image → CNN → Features (2048-dim)
Text → BERT → Features (768-dim)
    ↓
Concatenate: [2048 + 768 = 2816-dim]
    ↓
Dense layers (shared)
    ↓
Prediction
```

**Pros:**
```
- Simple
- Modalities can interact early
- Learns joint representation
```

**Cons:**
```
- One modality can dominate
- Hard to handle missing modalities
- High-dimensional concatenation
```

### Approach 2: Late Fusion

**Idea:** Train separate models, combine predictions

```
Image → CNN → Prediction 1
Text → BERT → Prediction 2
    ↓
Average/Weighted sum
    ↓
Final prediction
```

**Pros:**
```
- Easy to handle missing modalities (use available)
- Modular (train separately)
- Interpretable (see individual contributions)
```

**Cons:**
```
- No interaction between modalities
- May miss complementary information
- Limited joint learning
```

### Approach 3: Cross-Modal Attention

**Idea:** Let modalities attend to each other

```
Image features: [N × 2048] (N regions)
Text features: [M × 768] (M words)

Cross-attention:
For each word, attend to relevant image regions
For each image region, attend to relevant words

Used in:
- CLIP (OpenAI): Image-text matching
- DALL-E: Text-to-image generation
- Flamingo: Vision-language understanding
```

**Example (Visual Question Answering):**
```
Image: Photo of dog in park
Question: "What color is the dog?"

Process:
1. Extract image features (regions)
2. Extract question features (words)
3. Cross-attention: 
   - "color" attends to dog region
   - "dog" attends to animal in image
4. Fused features → Answer: "Brown"
```

### Approach 4: Contrastive Learning (CLIP)

**CLIP:** Contrastive Language-Image Pre-training

**Idea:** Learn shared embedding space

```
Image → Image Encoder → Embedding (512-dim)
Text → Text Encoder → Embedding (512-dim)

Training:
  Match image-text pairs (positive examples)
  Separate non-matching pairs (negative examples)

Loss: Contrastive loss
  Pull positive pairs together
  Push negative pairs apart

Result:
  Similar images and descriptions have similar embeddings
```

**Applications:**
```
Zero-shot classification:
  Text: "A photo of a cat"
  Text: "A photo of a dog"
  Image: [cat photo]
  
  Compute similarity with both texts
  Higher similarity with "cat" → Classify as cat
  
  No training needed! (zero-shot)
```

### Multi-Modal Applications

**1. Image Captioning:**
```
Input: Image
Output: Text description

Architecture:
  CNN (image) → Features
  LSTM/Transformer (decoder) generates caption
  Attention: Attend to relevant image regions while generating each word

Example:
  Image: Beach scene
  Caption: "A person walking on the beach at sunset"
```

**2. Visual Question Answering (VQA):**
```
Input: Image + Question (text)
Output: Answer (text)

Example:
  Image: Living room
  Question: "How many chairs are there?"
  Answer: "Three"

Uses: Cross-modal attention
```

**3. Video Understanding:**
```
Input: Video (visual + audio)
Output: Action label, caption, or answer

Combines:
  - CNN for frames (visual)
  - Audio features (sound)
  - LSTM/Transformer for temporal modeling

Applications: Sports analysis, surveillance, content moderation
```

**4. Medical Diagnosis:**
```
Input: X-ray image + patient history (text) + lab results
Output: Diagnosis, risk score

Multi-modal fusion captures holistic view
Better than image-only or text-only
```

**5. Autonomous Driving:**
```
Inputs: 
  - Camera (RGB images)
  - Lidar (3D point clouds)
  - Radar (object detection)
  - GPS (location)

Fusion for:
  - Object detection (pedestrians, vehicles)
  - Lane detection
  - Path planning

Safety critical: Redundancy from multiple sensors
```

### Interview Decision Framework

**Question:** "Design a system to detect inappropriate content in videos"

**Analysis:**
```
Modalities available:
- Video frames (visual)
- Audio track (sound)
- Subtitles/captions (text)

Challenges:
- Inappropriate content can be in any modality
  • Visual: Explicit images
  • Audio: Offensive language, violent sounds
  • Text: Hate speech in subtitles
  
- Need to detect combinations
  • Video + audio together (context)
```

**Recommended architecture:**
```
Multi-modal fusion:

1. Visual branch:
   - CNN (ResNet-50) on sampled frames (1 FPS)
   - Detect explicit content, violence

2. Audio branch:
   - Spectrogram → CNN
   - Detect screams, gunshots, offensive audio

3. Text branch (if subtitles available):
   - BERT
   - Detect hate speech, profanity

4. Fusion:
   - Cross-modal attention
   - Late fusion (max/average scores)
   
5. Temporal modeling:
   - LSTM over time
   - Detect patterns (violence escalation)

Output:
  - Per-frame scores
  - Overall video score
  - Timestamp of violations

Threshold-based decision:
  Low risk: Allow
  Medium risk: Flag for review
  High risk: Block
```

**Why multi-modal:**
```
Single modality misses context:
  - Violence in horror movie (intentional) vs real violence
  - Profanity in educational context vs hate speech
  
Multi-modal provides context for better decisions
```

---

## 7. Interview Focus: When to Apply Specialized Architectures

### Decision Framework

**Question:** "Given problem X, which architecture?"

**Step 1: Understand Data Structure**
```
Structured (tabular): XGBoost, LightGBM
Images: CNNs (ResNet, EfficientNet)
Text: Transformers (BERT, GPT)
Time series: LSTM, Prophet, ARIMA
Graphs: GNNs
Multi-modal: Fusion architectures
```

**Step 2: Problem Type**
```
Classification: Standard supervised learning
Generation: Autoregressive models (GPT, VAE)
Recommendation: Collaborative filtering, Neural CF
Anomaly detection: Autoencoders, Isolation Forest
Sequential decisions: Reinforcement learning
```

**Step 3: Constraints**
```
Data size: <10K → Simple models, >100K → Deep learning
Latency: <10ms → Linear models, <100ms → Neural networks
Interpretability: Required → Tree models, Optional → Neural networks
Resources: Limited → Smaller models, Abundant → Large models
```

**Step 4: Business Requirements**
```
Accuracy critical: Ensemble, large models
Cost critical: Simple models, caching
Real-time: Optimized architectures
Offline batch: Complex models acceptable
```

### Example Interview Questions

**Q1: "Predict customer churn. 100K customers, 50 features."**

**Analysis:**
```
Data: Structured (tabular)
Problem: Binary classification
Size: Medium (100K rows)
Latency: Not critical (batch predictions)
Interpretability: Important (understand why churning)
```

**Answer:**
```
Recommended: XGBoost

Why:
- Best for tabular data (structured features)
- Interpretable (feature importance, SHAP)
- Fast training (<10 minutes)
- Handles missing values
- State-of-the-art accuracy for this size

Alternative: Neural network
- If >1M samples
- If features have complex interactions
- But loses interpretability

Start simple: Logistic regression baseline
  → XGBoost (main model)
  → Neural network (if XGBoost insufficient)
```

**Q2: "Real-time product recommendations on e-commerce site."**

**Analysis:**
```
Data: User-item interactions (graph structure)
Problem: Recommendation
Size: 1M users, 100K products
Latency: <100ms (real-time)
Cold start: Frequent new products
```

**Answer:**
```
Recommended: Two-stage hybrid

Stage 1: Candidate generation (retrieve 1000)
  - Collaborative filtering (ALS)
  - Content-based (for new products)
  - Popular items (fallback)

Stage 2: Ranking (top 10)
  - XGBoost with features:
    • User: Purchase history, demographics
    • Item: Category, price, popularity
    • Context: Time, device, search query

Why two-stage:
- Stage 1 fast retrieval (ANN, <10ms)
- Stage 2 precise ranking (<50ms)
- Total latency: <100ms ✓

Why not GNN:
- Inference too slow (>100ms)
- Complexity not needed
- Collaborative filtering sufficient

Why not deep learning end-to-end:
- Slower than XGBoost
- Harder to maintain
- XGBoost accuracy sufficient for e-commerce
```

**Q3: "Detect defects in manufacturing from sensor data."**

**Analysis:**
```
Data: Time series (temperature, pressure, vibration)
Problem: Anomaly detection
Size: 1M readings/day
Imbalance: 0.01% defects
Real-time: <1 second detection needed
```

**Answer:**
```
Recommended: LSTM Autoencoder

Why:
- Time series data (temporal patterns)
- Unsupervised (few defect examples)
- Captures normal patterns
- Reconstruction error for anomalies

Architecture:
  Input: 100-step window (past 100 readings)
  Encoder: 2-layer LSTM (64 units)
  Decoder: 2-layer LSTM (64 units)
  Output: Reconstruct input

Training:
  - Train on normal data only
  - Learn to reconstruct normal patterns

Detection:
  - Reconstruction error > threshold → Defect
  - Threshold: 99th percentile on validation

Alternative: Isolation Forest
  - If real-time critical (faster inference)
  - If temporal patterns not complex
  - Simpler, easier to deploy

Start: Statistical methods (Z-score)
  → Isolation Forest (quick prototype)
  → LSTM Autoencoder (if accuracy insufficient)
```

---

## Summary: Key Takeaways

**1. Time Series:**
- **ARIMA:** Simple, univariate, linear (< 1K points)
- **Prophet:** Business forecasting, multiple seasonalities (1K-100K points)
- **LSTM:** Complex patterns, multivariate (>10K points)
- **Decision:** Start simple (Prophet), use LSTM only if needed

**2. Recommendation Systems:**
- **Collaborative Filtering:** User-item interactions, no features needed
- **Content-Based:** Item features, explainable, no cold start for items
- **Hybrid:** Best of both, production standard
- **Modern:** Two-stage (retrieval + ranking), neural networks

**3. Anomaly Detection:**
- **Statistical:** Simple, fast, interpretable (Z-score, Gaussian)
- **Isolation Forest:** High-dimensional, fast, no assumptions
- **Autoencoders:** Complex patterns, unsupervised
- **Decision:** Isolation Forest for most cases, autoencoders for complex

**4. Reinforcement Learning:**
- **Q-Learning:** Small discrete spaces, simple
- **DQN:** Large state spaces (images), discrete actions
- **Policy Gradients:** Continuous actions, robotics
- **When:** Sequential decisions, delayed rewards, no labels

**5. Graph Neural Networks:**
- **Use when:** Data has graph structure, relationships matter
- **Applications:** Social networks, molecules, fraud detection
- **Don't use:** No graph structure, complete graphs

**6. Multi-Modal Learning:**
- **Early fusion:** Combine features early
- **Late fusion:** Combine predictions
- **Cross-attention:** Let modalities interact (modern)
- **When:** Multiple data types, complementary information

**For Interviews:**
- **Always ask clarifying questions** (data size, latency, interpretability)
- **Start simple, add complexity** when justified
- **Know trade-offs** (accuracy vs speed vs interpretability)
- **Explain decision rationale** (why this architecture for this problem)
- **Consider production constraints** (not just accuracy)

**Critical Interview Theme:**
"Which architecture?" depends on:
1. Data structure (tabular, image, text, time series, graph)
2. Problem type (classification, generation, recommendation)
3. Constraints (data size, latency, interpretability, resources)
4. Business requirements (accuracy, cost, real-time)

**Always justify architecture choice with these factors!**

---

**END OF DAY 7**
