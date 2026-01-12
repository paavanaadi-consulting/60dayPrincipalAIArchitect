# Day 7: Advanced ML Topics - Interview Questions

## Overview

These questions assess deep understanding of specialized ML techniques and architectural decisions. Focus on **when and why** to apply each technique, not just how they work.

**Interview Strategy:**
- Ask clarifying questions about data, constraints, and business requirements
- Start with simple approaches, justify complexity
- Explain trade-offs and decision rationale
- Consider production constraints beyond just accuracy

---

## Section 1: Time Series Forecasting

### Basic Level (3-5 years experience)

**Q1: Explain why traditional ML algorithms (like Random Forest) don't work well for time series forecasting.**

**Expected Answer:**
- **IID Assumption Violation:** Traditional ML assumes samples are independent and identically distributed. In time series, observations are temporally dependent (today's sales affect tomorrow's).
- **Temporal Order Loss:** Random Forest can shuffle samples during training, losing crucial temporal ordering.
- **No Temporal Features:** Traditional ML doesn't naturally capture trends, seasonality, or autocorrelation.
- **Example:** Predicting stock prices - knowing the sequence of past prices is crucial, but Random Forest treats each day as independent.

**Q2: You need to forecast daily sales for a retail store. You have 2 years of data. Walk me through your approach.**

**Expected Answer:**
- **Data Exploration:**
  - Check for trends (growing sales over time?)
  - Identify seasonality (weekly patterns, holidays)
  - Look for outliers (Black Friday spikes)
  - Missing data handling
- **Approach Selection:**
  - Start with **Prophet** (designed for business forecasting)
  - Handles multiple seasonalities automatically
  - Can incorporate holidays/promotions
  - Quick prototype in hours, not days
- **Validation Strategy:**
  - Time-based splits (train on first 18 months, validate on last 6)
  - Walk-forward validation (retrain monthly)
- **Success Criteria:** MAPE < 15% for retail typically acceptable

**Q3: Compare ARIMA, Prophet, and LSTM for time series forecasting. When would you choose each?**

**Expected Answer:**

| Method | Data Size | Complexity | Best Use Case |
|--------|-----------|------------|---------------|
| **ARIMA** | <1K points | Linear trends | Short-term, univariate, need interpretability |
| **Prophet** | 1K-100K | Additive seasonality | Business metrics, multiple seasonalities |
| **LSTM** | >10K points | Non-linear | Complex patterns, multivariate, high accuracy needs |

- **ARIMA:** Academic research, small datasets, need confidence intervals
- **Prophet:** E-commerce sales, website traffic, revenue forecasting  
- **LSTM:** High-frequency trading, sensor data with many features

### Advanced Level (5+ years experience)

**Q4: Your Prophet model is overfitting to holiday effects. The model predicts huge spikes on holidays that don't materialize. How do you debug and fix this?**

**Expected Answer:**
- **Diagnosis:**
  - Check holiday coefficients: `model.params['holidays']`
  - Plot holiday effects: `plot_components(forecast)`
  - Validate on recent holidays
- **Solutions:**
  1. **Regularize holiday effects:** Increase `holidays_prior_scale` (default 10 → 0.1)
  2. **Reduce holiday impact:** Set custom prior scales per holiday
  3. **Holiday grouping:** Group similar holidays (all federal holidays together)
  4. **Cross-validation:** Use `cross_validation()` to tune hyperparameters
- **Code Example:**
```python
model = Prophet(
    holidays_prior_scale=0.1,  # Regularize holidays
    yearly_seasonality=True,
    weekly_seasonality=True
)
model.add_country_holidays('US')
```

**Q5: Design a real-time anomaly detection system for website traffic that can distinguish between normal spikes (viral content) and actual issues (DDoS attacks).**

**Expected Answer:**
- **Multi-layered approach:**
  1. **Statistical baseline:** Simple z-score for obvious anomalies
  2. **Time series model:** Prophet for expected traffic patterns
  3. **Feature engineering:** Additional signals
     - Traffic source (direct vs external)
     - Geographic distribution
     - User agent patterns
     - Page access patterns
- **Architecture:**
```python
# Real-time pipeline
traffic_data → Feature extraction → [Prophet forecast, Z-score, Context features] 
             → Ensemble model → [Normal, Viral, Attack] classification
```
- **Distinguishing signals:**
  - **Viral:** Gradual increase, social media referrers, normal user behavior
  - **DDoS:** Sudden spike, suspicious IPs, repetitive requests, single pages

**Q6: You have multiple time series (sales across 1000 stores) that are related. How would you model this?**

**Expected Answer:**
- **Hierarchical forecasting approaches:**
  1. **Bottom-up:** Forecast each store individually, sum for total
  2. **Top-down:** Forecast total, distribute to stores
  3. **Middle-out:** Forecast at cluster level (geographic regions)
- **Advanced techniques:**
  - **Vector Autoregression (VAR):** Model cross-series dependencies
  - **Prophet with regressors:** Use aggregate signals as features
  - **Deep learning:** LSTM with shared embeddings across stores
- **Practical recommendation:**
```python
# Hierarchical with Prophet
# Level 1: Total sales (all stores)
# Level 2: Regional sales (by geography)  
# Level 3: Individual stores
# Reconcile forecasts to ensure consistency
```
- **Benefits:** Better accuracy for small stores, maintains business logic

---

## Section 2: Recommendation Systems

### Basic Level (3-5 years experience)

**Q7: Explain the cold start problem in recommendation systems. How would you handle new users and new items?**

**Expected Answer:**
- **Cold Start Definition:**
  - **New User:** No interaction history to base recommendations on
  - **New Item:** No user ratings/interactions to understand item preferences
- **Solutions for New Users:**
  1. **Popularity-based:** Recommend most popular items globally
  2. **Demographic-based:** Use age, gender, location for similar users
  3. **Onboarding quiz:** Ask preferences during signup
  4. **Content-based:** Use item features, not user history
- **Solutions for New Items:**
  1. **Content-based filtering:** Use item features (genre, price, category)
  2. **Hybrid approach:** Combine with collaborative filtering
  3. **Exploration bonus:** Boost new items in rankings temporarily
- **Production strategy:** Start with content-based, switch to collaborative as data accumulates

**Q8: Design a recommendation system for a music streaming service with 100M users and 50M songs.**

**Expected Answer:**
- **Two-stage architecture:**
  
  **Stage 1: Candidate Generation (100M songs → 1000 candidates)**
  - **Collaborative filtering:** ALS matrix factorization for user-song interactions
  - **Content-based:** Song features (genre, artist, tempo) for new songs
  - **Sequential:** Recently played songs → similar songs
  - **Popular:** Trending songs globally/regionally
  
  **Stage 2: Ranking (1000 → 10 recommendations)**
  - **LightGBM/XGBoost** with features:
    - User: Age, location, subscription type, listening history
    - Song: Popularity, genre, release date, acoustic features
    - Context: Time of day, device, playlist context
    - Interaction: Skip rate, replay rate for similar songs

- **Special considerations:**
  - **Real-time updates:** User skips song → update preferences immediately
  - **Diversity:** Not all same artist/genre (MMR - Maximal Marginal Relevance)
  - **Exploration:** 10% random songs for discovery

**Q9: How would you evaluate a recommendation system? What metrics would you use?**

**Expected Answer:**
- **Offline Metrics (Historical data):**
  - **Precision@K:** Of top K recommendations, how many were relevant?
  - **Recall@K:** Of all relevant items, how many were in top K?
  - **NDCG@K:** Normalized Discounted Cumulative Gain (ranking quality)
  - **Coverage:** What percentage of items can be recommended?

- **Online Metrics (A/B testing):**
  - **Click-through rate (CTR):** Users click on recommendations
  - **Conversion rate:** Users purchase/consume recommended items
  - **Engagement:** Time spent with recommended content
  - **User retention:** Do better recommendations keep users?

- **Business Metrics:**
  - **Revenue impact:** Do recommendations increase sales?
  - **User satisfaction:** Surveys, ratings of recommendation quality
  - **Diversity:** User explores new content vs. echo chamber

### Advanced Level (5+ years experience)

**Q10: Your collaborative filtering model has a popularity bias - it only recommends mainstream items. How do you increase diversity while maintaining relevance?**

**Expected Answer:**
- **Root cause:** Popular items have more interactions → higher scores → recommended more
- **Solutions:**
  1. **Inverse popularity weighting:** Boost scores of less popular items
  2. **MMR (Maximal Marginal Relevance):** Balance relevance vs diversity
     ```python
     # Select next item that maximizes:
     # λ * relevance_score(item) - (1-λ) * max_similarity(item, selected_items)
     ```
  3. **Regularization:** Add diversity term to loss function
  4. **Post-processing:** Ensure top 10 has items from different categories
  5. **Exploration:** Reserve 20% of recommendations for long-tail items

- **Business trade-off:** Slight accuracy decrease for better user experience and discovery

**Q11: Design a session-based recommendation system for e-commerce where you need to recommend products based on the current shopping session (no user login).**

**Expected Answer:**
- **Challenge:** No user history, only current session behavior
- **Architecture:**
  
  **Session modeling:**
  - **Sequential patterns:** User viewed Phone → Case → Screen Protector
  - **RNN/GRU:** Learn session progression patterns
  - **Session embeddings:** Aggregate item interactions in session
  
  **Real-time inference:**
  ```python
  current_session = [item1, item2, item3]  # User's current items
  session_embedding = RNN(item_embeddings)  # Encode session
  candidates = similar_sessions(session_embedding)  # Find similar patterns
  recommendations = rank(candidates, current_context)
  ```

- **Features:**
  - **Item sequence:** Order of items viewed/added
  - **Time spent:** Dwell time on each product
  - **Context:** Device, location, time of day, season
  - **Item attributes:** Category, price range, brand

- **Fallbacks:**
  - **Co-occurrence:** "Users who viewed X also viewed Y"
  - **Category trends:** Popular in current category
  - **Seasonal/trending:** Current popular items

**Q12: How would you handle the feedback loop problem in recommendation systems where your recommendations influence user behavior, which then influences future recommendations?**

**Expected Answer:**
- **Problem:** Recommendations → User clicks → More similar recommendations → Filter bubble
- **Example:** User clicks action movies → Only see action movies → Never discover other genres

- **Solutions:**
  1. **Exploration policies:**
     - **ε-greedy:** 90% exploitation (best recommendations), 10% exploration (random)
     - **Thompson sampling:** Probabilistic exploration based on uncertainty
  
  2. **Unbiased evaluation:**
     - **Inverse Propensity Scoring:** Weight training data by recommendation probability
     - **Doubly robust estimation:** Combine model-based and propensity-based methods
  
  3. **Counterfactual evaluation:**
     - **A/B testing:** Compare recommendation policies
     - **Causal inference:** Estimate effect of recommendations on user preferences
  
  4. **Diversification:**
     - **Fairness constraints:** Ensure representation across categories/artists
     - **Temporal diversity:** Vary recommendations over time

---

## Section 3: Anomaly Detection

### Basic Level (3-5 years experience)

**Q13: You're building a fraud detection system for credit card transactions. You have 1 million transactions per day with 0.1% fraud rate. What's your approach?**

**Expected Answer:**
- **Challenge:** Extreme imbalance (0.1% fraud), real-time requirements (<100ms)

- **Feature engineering:**
  - **Transaction:** Amount, merchant category, location
  - **User behavior:** Spending patterns, typical locations, time patterns
  - **Aggregated:** Spending last 24h, number of transactions, velocity
  - **Network:** Merchant risk score, IP reputation

- **Model approach:**
  1. **Isolation Forest:** Real-time scoring, handles high dimensions
  2. **Ensemble:** Combine multiple weak signals
  3. **Threshold tuning:** Balance false positives vs fraud detection

- **Evaluation:**
  - **Precision/Recall trade-off:** Customer experience vs fraud loss
  - **Cost-sensitive:** False negative (missed fraud) costs more than false positive

- **Production pipeline:**
```
Transaction → Feature extraction → Model scoring → [Block, Review, Allow]
```

**Q14: Compare Isolation Forest vs Autoencoder for anomaly detection. When would you use each?**

**Expected Answer:**

| Aspect | Isolation Forest | Autoencoder |
|--------|------------------|-------------|
| **Training data** | Normal + anomalies | Normal data only |
| **Data size** | 1K-1M samples | >10K samples |
| **Features** | High-dimensional | Any dimensionality |
| **Speed** | Fast (linear) | Slow (neural network) |
| **Interpretability** | Low | Very low |
| **Complex patterns** | Simple | Complex non-linear |

- **Use Isolation Forest when:**
  - Mixed normal/abnormal training data
  - Need fast real-time detection
  - High-dimensional sparse data
  - Limited training data

- **Use Autoencoder when:**
  - Only normal data for training
  - Complex patterns (images, sequences)
  - Accuracy more important than speed
  - Large dataset available

**Q15: Your anomaly detection model has high false positive rate in production. How do you reduce it while maintaining detection capability?**

**Expected Answer:**
- **Root cause analysis:**
  1. **Data drift:** Production data differs from training data
  2. **Threshold too strict:** Need to relax threshold
  3. **Missing context:** Model lacks important features
  4. **Seasonality:** Normal patterns change over time

- **Solutions:**
  1. **Adaptive thresholds:** Adjust based on recent performance
  2. **Ensemble approach:** Multiple models vote (reduce single model errors)
  3. **Human feedback loop:** Learn from false positive corrections
  4. **Context features:** Add time of day, day of week, seasonality
  5. **Staged approach:**
     ```
     High confidence anomaly → Immediate action
     Medium confidence → Human review
     Low confidence → Log for analysis
     ```

### Advanced Level (5+ years experience)

**Q16: Design an anomaly detection system for a manufacturing plant with 10,000 sensors generating data every second. How do you handle the scale and complexity?**

**Expected Answer:**
- **Architecture challenges:**
  - **Volume:** 10K sensors × 86,400 seconds = 864M data points/day
  - **Velocity:** Real-time detection needed (1-second latency)
  - **Variety:** Different sensor types (temperature, pressure, vibration)

- **Hierarchical approach:**
  1. **Edge level:** Simple statistical checks per sensor (z-score)
  2. **Local level:** Correlation analysis within sensor groups
  3. **Global level:** Plant-wide pattern recognition

- **Technical architecture:**
```
Sensors → Stream processing (Kafka/Kinesis) → 
         [Real-time rules, Statistical models, ML models] → 
         Alert system → Human operators
```

- **Models by urgency:**
  - **Immediate (<1s):** Rule-based thresholds
  - **Short-term (<10s):** Isolation Forest on recent windows
  - **Medium-term (<1min):** LSTM Autoencoder for temporal patterns
  - **Long-term (batch):** Deep analysis for root cause

**Q17: How would you detect coordinated inauthentic behavior (bot networks) on a social media platform?**

**Expected Answer:**
- **Multi-layered detection:**

  **Individual account level:**
  - **Profile features:** Default profile pics, similar usernames, creation dates
  - **Behavior patterns:** Posting frequency, interaction patterns, content similarity
  - **Network features:** Friend/follower relationships, interaction timing

  **Network level (Graph analysis):**
  - **Community detection:** Unusual clustering of accounts
  - **Synchronous behavior:** Accounts acting in coordination
  - **Graph patterns:** Star topology (one account → many followers)

  **Content level:**
  - **Text similarity:** Copy-paste content, template messages
  - **Temporal patterns:** Simultaneous posting across accounts
  - **Hashtag coordination:** Artificial trending attempts

- **Technical approach:**
```python
# Multi-modal detection
account_features → Isolation Forest → Individual anomaly scores
network_graph → Graph Neural Network → Network anomaly scores
content_text → BERT embeddings → Content similarity scores
time_series → LSTM → Temporal anomaly scores

# Ensemble all scores → Final bot probability
```

---

## Section 4: Reinforcement Learning

### Basic Level (3-5 years experience)

**Q18: Explain the difference between supervised learning and reinforcement learning. When would you choose RL over supervised learning?**

**Expected Answer:**
- **Key differences:**

| Aspect | Supervised Learning | Reinforcement Learning |
|--------|-------------------|----------------------|
| **Data** | Input-output pairs (x,y) | Environment interactions |
| **Feedback** | Immediate correct answer | Delayed rewards |
| **Goal** | Learn mapping function | Learn optimal policy |
| **Examples** | Image classification | Game playing |

- **Use RL when:**
  1. **Sequential decisions:** Action affects future states (games, robotics)
  2. **No labeled data:** Only know final outcome (win/lose)
  3. **Need exploration:** Discover optimal strategy through trial and error
  4. **Dynamic environment:** Strategy must adapt to changing conditions

- **Use Supervised when:**
  1. **Static decisions:** Each prediction independent
  2. **Labeled data available:** Clear input-output pairs
  3. **Known optimal behavior:** Can learn from expert demonstrations

**Q19: You're building an AI for a simple game (like Tic-Tac-toe). Walk through your RL approach.**

**Expected Answer:**
- **MDP formulation:**
  - **State:** Current board configuration
  - **Actions:** Available moves (empty squares)
  - **Reward:** +1 for win, -1 for loss, 0 for draw, -0.01 per move (encourage shorter games)
  - **Policy:** Which move to choose in each state

- **Approach - Q-Learning:**
  1. **Q-table:** State-action values Q(s,a)
  2. **Exploration:** ε-greedy (start ε=1.0, decay to 0.1)
  3. **Training:** Self-play (agent vs itself)
  4. **Update rule:** Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

- **Training process:**
```python
for episode in range(10000):
    state = initial_board()
    while not terminal(state):
        action = epsilon_greedy(Q, state)
        next_state, reward = environment.step(action)
        update_Q(state, action, reward, next_state)
        state = next_state
```

**Q20: What is the exploration vs exploitation trade-off? How do you balance it?**

**Expected Answer:**
- **Trade-off:**
  - **Exploration:** Try new actions to learn about environment
  - **Exploitation:** Choose best known action to maximize reward
  - **Dilemma:** Need to explore to find better actions, but exploring might give lower rewards

- **Strategies:**
  1. **ε-greedy:** Random action with probability ε
  2. **Decaying ε:** Start high (explore), decrease over time (exploit)
  3. **UCB (Upper Confidence Bound):** Choose actions with high uncertainty
  4. **Thompson Sampling:** Probabilistic approach based on action value uncertainty

- **Practical example:**
```python
# Decaying epsilon
epsilon = max(0.1, 1.0 * (0.995 ** episode))
if random.random() < epsilon:
    action = random_action()  # Explore
else:
    action = best_action(Q)   # Exploit
```

### Advanced Level (5+ years experience)

**Q21: You're building an RL system for dynamic pricing in e-commerce. How would you approach this problem?**

**Expected Answer:**
- **Problem formulation:**
  - **State:** Current inventory, competitor prices, demand signals, time features
  - **Action:** Price adjustment (discrete: increase/decrease/maintain)
  - **Reward:** Revenue - lost demand penalty
  - **Constraints:** Price bounds, inventory constraints

- **Challenges:**
  - **Sparse rewards:** Sales might happen hours/days later
  - **Non-stationary:** Market conditions change
  - **Safety:** Can't experiment with extreme prices (business risk)

- **Approach:**
  1. **Start with supervised learning:** Learn from historical pricing data
  2. **Offline RL:** Train on historical data before online deployment
  3. **Conservative policy:** Small price adjustments initially
  4. **A/B testing framework:** Test RL policy on small traffic percentage

- **Technical implementation:**
```python
# Actor-Critic with constrained actions
state = [inventory, competitor_prices, demand_features, time_features]
action = policy_network(state)  # Price adjustment
constrained_action = clip(action, min_price, max_price)  # Safety
reward = calculate_revenue(new_price, actual_demand)
```

**Q22: How would you handle the partial observability problem in RL? Give a practical example.**

**Expected Answer:**
- **Partial Observability:** Agent doesn't see complete state of environment

- **Example - Autonomous driving:**
  - **Observable:** Camera images, sensor readings, GPS
  - **Hidden:** Other drivers' intentions, road conditions ahead, pedestrian behavior

- **Solutions:**
  1. **Recurrent Neural Networks:** LSTM to maintain internal state/memory
  2. **Frame stacking:** Use multiple recent observations
  3. **Belief states:** Maintain probability distribution over possible states
  4. **Attention mechanisms:** Focus on relevant parts of observation history

- **Technical approach:**
```python
# LSTM-based policy for partial observability
class PartialObsPolicy(nn.Module):
    def __init__(self):
        self.lstm = nn.LSTM(obs_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, obs_sequence, hidden_state):
        lstm_out, new_hidden = self.lstm(obs_sequence, hidden_state)
        action_probs = self.policy_head(lstm_out[-1])  # Use last output
        return action_probs, new_hidden
```

---

## Section 5: Graph Neural Networks

### Basic Level (3-5 years experience)

**Q23: What makes graph data different from images or text? Why can't we use CNNs or RNNs directly on graphs?**

**Expected Answer:**
- **Graph structure differences:**
  - **Images:** Regular grid structure, fixed neighborhood size
  - **Text:** Sequential structure, clear ordering
  - **Graphs:** Irregular structure, variable neighborhood sizes

- **Why CNNs don't work:**
  - CNNs use fixed-size kernels (3x3, 5x5)
  - Graphs have variable node degrees (some nodes have 2 neighbors, others have 100)
  - No spatial locality assumption

- **Why RNNs don't work:**
  - RNNs assume sequential ordering
  - Graphs have no natural ordering of nodes
  - Multiple valid traversal orders exist

- **Solution - GNNs:**
  - **Message passing:** Each node aggregates information from neighbors
  - **Permutation invariant:** Order of neighbors doesn't matter
  - **Scalable:** Can handle variable neighborhood sizes

**Q24: Explain message passing in GNNs with a concrete example.**

**Expected Answer:**
- **Social network example:**
  - **Nodes:** Users (Alice, Bob, Carol)
  - **Edges:** Friendships
  - **Goal:** Predict user interests

- **Message passing process:**
  
  **Layer 1:**
  ```
  Alice's neighbors: [Bob, Carol]
  Messages: [Bob's features, Carol's features]
  Aggregation: mean([Bob_features, Carol_features])
  Update: Alice_new = MLP([Alice_old, aggregated_message])
  ```

  **Layer 2:**
  ```
  Alice now has information from 2-hop neighbors
  (Bob's friends, Carol's friends through Bob and Carol)
  ```

- **Mathematical formulation:**
```python
# For each node v
messages = []
for neighbor u in neighbors(v):
    messages.append(u.features)

aggregated = aggregate(messages)  # mean, sum, max
v.features_new = update(v.features_old, aggregated)
```

**Q25: When would you use GNNs vs traditional ML? Give specific use cases.**

**Expected Answer:**
- **Use GNNs when:**
  1. **Data has graph structure:** Social networks, molecules, road networks
  2. **Relationships matter:** Fraud detection (fraudsters connected), drug discovery (molecular structure)
  3. **Need to propagate information:** Semi-supervised learning (few labels, many unlabeled)

- **Specific use cases:**
  - **Fraud detection:** Credit card fraud often involves connected accounts (family, organized crime)
  - **Drug discovery:** Molecular properties depend on atomic bonds and structure
  - **Recommendation:** User-item interactions form bipartite graph
  - **Traffic prediction:** Road networks where traffic propagates

- **Don't use GNNs when:**
  - **No natural graph structure:** Independent samples (image classification)
  - **Complete graphs:** Every node connected to every other (use regular neural networks)
  - **Simple relationships:** Can be captured by features (no need for graph structure)

### Advanced Level (5+ years experience)

**Q26: Design a GNN system to detect money laundering in financial transactions. What are the key challenges?**

**Expected Answer:**
- **Graph construction:**
  - **Nodes:** Bank accounts, individuals, companies
  - **Edges:** Transactions (weighted by amount, frequency)
  - **Node features:** Account type, balance, location, activity patterns
  - **Edge features:** Transaction amount, timing, frequency

- **Money laundering patterns:**
  - **Smurfing:** Many small transactions from multiple accounts → single account
  - **Layering:** Complex chains of transactions to obscure origin
  - **Shell companies:** Multiple layers of fake companies

- **GNN approach:**
```python
# Multi-relational GNN
class MoneyLaunderingGNN(nn.Module):
    def __init__(self):
        self.account_conv = GraphConv(account_features)
        self.transaction_conv = GraphConv(transaction_features)
        self.temporal_conv = TemporalConv()  # Handle time sequences
    
    def forward(self, graph, time_window):
        # Learn account representations
        account_emb = self.account_conv(graph.nodes, graph.edges)
        
        # Learn transaction patterns
        transaction_emb = self.transaction_conv(graph.transactions)
        
        # Combine with temporal patterns
        risk_score = self.classify(account_emb, transaction_emb)
        return risk_score
```

- **Challenges:**
  1. **Scale:** Millions of accounts, billions of transactions
  2. **Temporal dynamics:** Patterns change over time
  3. **Adversarial:** Criminals adapt to detection methods
  4. **Imbalanced data:** Very few actual money laundering cases
  5. **Privacy:** Sensitive financial data

**Q27: How would you handle the scalability problem when applying GNNs to graphs with billions of nodes (like Facebook's social graph)?**

**Expected Answer:**
- **Sampling strategies:**
  1. **Node sampling:** Sample subset of nodes per mini-batch
  2. **Subgraph sampling:** Extract connected subgraphs
  3. **Layer sampling:** Sample different neighborhoods for each GNN layer

- **Specific techniques:**
  - **GraphSAINT:** Sample connected subgraphs for training
  - **FastGCN:** Sample nodes independently for each layer
  - **GraphSAGE:** Sample fixed number of neighbors per node

- **Distributed training:**
```python
# Partition graph across multiple GPUs/machines
# Each partition handles subset of nodes
# Message passing across partitions handled by communication layer

class DistributedGNN:
    def forward_pass(self, local_nodes):
        # Compute on local nodes
        local_embeddings = self.local_gnn(local_nodes)
        
        # Exchange boundary node information
        boundary_embeddings = self.communicate_boundaries()
        
        # Combine local and boundary information
        final_embeddings = self.combine(local_embeddings, boundary_embeddings)
        return final_embeddings
```

- **Engineering optimizations:**
  - **Sparse matrices:** Efficient storage and computation
  - **Batch processing:** Group similar operations
  - **Caching:** Store frequently accessed embeddings

---

## Section 6: Multi-Modal Learning

### Basic Level (3-5 years experience)

**Q28: What are the main challenges in multi-modal learning? How do you address the representation gap between different modalities?**

**Expected Answer:**
- **Key challenges:**
  1. **Representation gap:** Different modalities have different characteristics
     - Images: 2D spatial, continuous values
     - Text: Sequential, discrete tokens  
     - Audio: Temporal, continuous waveforms

  2. **Alignment:** How to connect corresponding elements across modalities
     - Which image region corresponds to which text phrase?

  3. **Fusion:** When and how to combine modalities
     - Early: Combine raw features (simple but may lose modality-specific patterns)
     - Late: Combine predictions (modular but may miss interactions)

- **Solutions:**
  - **Shared embedding space:** Train encoders to map different modalities to same vector space (like CLIP)
  - **Cross-attention:** Let modalities attend to each other
  - **Contrastive learning:** Pull matching pairs together, push non-matching apart

**Q29: Design a system for automatic image captioning. Walk through your architecture and training approach.**

**Expected Answer:**
- **Architecture:**
```
Image → CNN (ResNet-50) → Visual features (2048-dim)
       ↓
   Attention mechanism
       ↓
   LSTM Decoder → Word sequence

# At each timestep:
# - LSTM generates next word
# - Attention decides which image regions are relevant
# - Combine attention + LSTM state → predict word
```

- **Training:**
  1. **Dataset:** MS COCO (images with 5 human-written captions each)
  2. **Loss:** Cross-entropy on next word prediction
  3. **Teacher forcing:** During training, use ground-truth previous word
  4. **Evaluation:** BLEU, METEOR, CIDEr scores

- **Attention mechanism:**
```python
# Visual attention
attention_weights = softmax(MLP(concat(lstm_state, visual_features)))
attended_features = sum(attention_weights * visual_features)
next_word_logits = MLP(concat(lstm_state, attended_features))
```

**Q30: How would you handle missing modalities in a multi-modal system during inference?**

**Expected Answer:**
- **Problem:** Training with all modalities, but inference with some missing (audio broken, image blurry)

- **Solutions:**
  1. **Modular architecture:** Train separate encoders, robust to missing inputs
  ```python
  if image_available:
      image_features = image_encoder(image)
  else:
      image_features = zero_vector()  # or learned default
      
  if text_available:
      text_features = text_encoder(text)
  else:
      text_features = zero_vector()
      
  combined = fusion_layer(image_features, text_features)
  ```

  2. **Uncertainty modeling:** Express confidence based on available modalities
  3. **Graceful degradation:** Performance degrades smoothly with fewer modalities
  4. **Cross-modal generation:** Use available modalities to infer missing ones

### Advanced Level (5+ years experience)

**Q31: Design a multi-modal search system that can find relevant videos using text queries, example images, or audio clips. How do you create a unified search experience?**

**Expected Answer:**
- **Unified embedding space:**
```
Text query → Text encoder → 512-dim embedding
Image → Vision encoder → 512-dim embedding  
Audio → Audio encoder → 512-dim embedding
Video → [Vision + Audio] → 512-dim embedding

All modalities mapped to same space for comparison
```

- **Architecture:**
  1. **Video preprocessing:**
     - Extract frames (1 FPS) → Vision features
     - Extract audio → Audio features
     - Combine with temporal modeling (LSTM/Transformer)

  2. **Query processing:**
     - Text: BERT/RoBERTa encoder
     - Image: ResNet/ViT encoder  
     - Audio: Mel-spectrogram → CNN

  3. **Similarity computation:**
     - Cosine similarity in shared embedding space
     - Rank videos by similarity score

- **Training:**
  - **Contrastive learning:** Video-caption pairs, video-audio synchronization
  - **Hard negative mining:** Find challenging negative examples
  - **Multi-task learning:** Optimize for all query types simultaneously

- **Advanced features:**
  - **Temporal localization:** Find exact timestamp in video
  - **Multi-modal query composition:** "Video of dogs" + [image of park] = "Dogs playing in parks"

---

## Section 7: Architecture Decision Framework

### System Design Questions (5+ years experience)

**Q32: You're the ML lead for a startup. You need to build a real-time personalization system for an e-commerce app. You have 100K users, 50K products, limited budget, and need to launch in 3 months. What's your approach?**

**Expected Answer:**
- **Constraints analysis:**
  - **Scale:** Medium (100K users) → Simple solutions work
  - **Timeline:** 3 months → Need proven techniques, not research
  - **Budget:** Limited → Cloud costs matter, prefer simpler models
  - **Real-time:** <100ms response needed

- **Recommended architecture:**
  1. **Start simple:** Popularity + content-based filtering
  2. **Phase 2:** Add collaborative filtering (matrix factorization)
  3. **Phase 3:** Optimize with deep learning if needed

- **Technical stack:**
```
Stage 1 (Month 1): 
  - Popular items by category
  - Content-based using product features
  - A/B test framework

Stage 2 (Month 2):
  - Matrix factorization (Surprise library)
  - Hybrid with content-based
  - Real-time inference API

Stage 3 (Month 3):
  - Neural collaborative filtering (if needed)
  - Advanced features (time, context)
  - Performance optimization
```

- **Success metrics:**
  - CTR improvement: >5% vs random
  - Revenue impact: >2% increase
  - Latency: <100ms p95

**Q33: Compare your approach for these three scenarios: (a) Real-time fraud detection, (b) Customer churn prediction, (c) Dynamic pricing. How do your architecture choices differ?**

**Expected Answer:**

| Aspect | Fraud Detection | Churn Prediction | Dynamic Pricing |
|--------|-----------------|------------------|-----------------|
| **Latency** | <50ms (real-time) | Hours/days (batch) | Minutes (semi-real-time) |
| **Features** | Transaction details | Historical behavior | Market conditions |
| **Model** | Isolation Forest/XGBoost | XGBoost/Neural Net | Reinforcement Learning |
| **Evaluation** | Precision/Recall | Precision@K | Revenue impact |
| **Infrastructure** | Streaming (Kafka) | Batch (Spark) | Hybrid |

**Architecture differences:**

**Fraud Detection:**
- **Real-time streaming:** Kafka → Model server → Decision API
- **Feature store:** Pre-computed user profiles for fast lookup
- **Model:** Fast inference (XGBoost, 10ms latency)
- **Fallback:** Rule-based system if model fails

**Churn Prediction:**
- **Batch processing:** Daily/weekly model scoring
- **Feature engineering:** Complex aggregations (3 months purchase history)
- **Model:** Accuracy-focused (ensemble, neural networks)
- **Output:** Risk scores for marketing campaigns

**Dynamic Pricing:**
- **Semi-real-time:** Hourly price updates based on inventory/competition
- **RL framework:** Online learning from price experiments
- **Safety constraints:** Price bounds, business rules
- **A/B testing:** Careful experimentation framework

---

## Section 8: Trade-offs and Decision Making

### Senior Architect Questions (7+ years experience)

**Q34: You have a client who insists on using deep learning for a problem with 1000 samples and 10 features (customer churn). How do you handle this conversation?**

**Expected Answer:**
- **Technical reality:**
  - Deep learning needs large datasets (>10K samples typically)
  - 1000 samples → High overfitting risk
  - 10 features → Traditional ML sufficient
  - Simple models will outperform complex ones

- **Conversation approach:**
  1. **Acknowledge their perspective:** "I understand the appeal of deep learning..."
  2. **Explain the problem:** "With 1000 samples, deep learning will likely overfit..."
  3. **Propose alternative:** "Let's start with XGBoost, which is ideal for this data size..."
  4. **Show evidence:** "Here's a quick prototype comparing both approaches..."
  5. **Future path:** "Once we collect 10K+ samples, we can revisit deep learning"

- **Compromise solution:**
  - Build both prototypes
  - Demonstrate XGBoost superiority with cross-validation
  - Document when to transition to deep learning
  - Focus on business value, not technology choices

**Q35: Your company wants to implement "AI-driven decision making" across all business functions. As the Principal ML Engineer, how do you approach this broad mandate?**

**Expected Answer:**
- **Assessment phase:**
  1. **Inventory current processes:** Which decisions are made how?
  2. **Identify data availability:** What data exists for each decision?
  3. **Estimate business impact:** Which decisions have highest value?
  4. **Assess technical feasibility:** Which problems are ML-suitable?

- **Prioritization framework:**
```
Impact vs Feasibility Matrix:
High Impact + High Feasibility → Quick wins (start here)
High Impact + Low Feasibility → Long-term research
Low Impact + High Feasibility → Maybe later
Low Impact + Low Feasibility → Don't do
```

- **Implementation strategy:**
  1. **Start small:** Pilot with 1-2 high-value, low-risk decisions
  2. **Build infrastructure:** Data pipelines, ML platforms, monitoring
  3. **Demonstrate value:** Show ROI and build organizational buy-in
  4. **Scale gradually:** Expand to more complex decisions
  5. **Change management:** Train teams, adjust processes

- **Common pitfalls to avoid:**
  - Trying to automate everything at once
  - Ignoring human expertise and judgment
  - Underestimating data quality requirements
  - Not measuring business impact

---

## Evaluation Rubric

### Junior Level (3-5 years)
- **Basic understanding** of specialized techniques (time series, recommendations, anomaly detection)
- **Can explain** when to use different approaches with guidance
- **Recognizes** common patterns and standard solutions
- **Implements** existing algorithms with minor modifications

### Mid Level (5-7 years)  
- **Deep technical knowledge** of multiple advanced areas
- **Makes informed decisions** about technique selection based on constraints
- **Handles trade-offs** between accuracy, interpretability, and performance
- **Debugs and optimizes** complex models in production

### Senior Level (7+ years)
- **Architected end-to-end systems** using advanced ML techniques
- **Leads technical decisions** across multiple domains and teams  
- **Balances business requirements** with technical constraints
- **Mentors others** and drives technical strategy

### Principal/Staff Level (8+ years)
- **Designs organizational ML strategy** across multiple business units
- **Makes build-vs-buy decisions** for advanced ML capabilities
- **Handles complex trade-offs** involving technical debt, team skills, and business priorities
- **Drives innovation** while maintaining production system reliability
- **Influences product strategy** through deep ML expertise

### Key Evaluation Criteria
1. **Problem decomposition:** Breaking complex problems into manageable pieces
2. **Constraint handling:** Balancing accuracy, latency, cost, interpretability
3. **Decision rationale:** Clear reasoning for architectural choices
4. **Production awareness:** Understanding operational requirements
5. **Business context:** Connecting technical decisions to business value
