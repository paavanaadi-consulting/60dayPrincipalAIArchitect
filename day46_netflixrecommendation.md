# Day 46: Netflix Recommendation System

## Overview

Netflix's recommendation system is one of the most sophisticated ML systems in production, driving 80% of watch time. Understanding it is critical for Principal AI Architect interviews because:
- Real-world example of ML at massive scale (200M+ users)
- Covers full ML lifecycle (data, training, serving, experimentation)
- Demonstrates trade-offs between accuracy, diversity, and business goals
- Common interview question: "Design Netflix recommendations"

**Key Stats:**
- 200M+ subscribers globally
- 80% of viewing from recommendations
- Saves $1B/year in customer retention
- 100K+ titles in catalog
- Billions of recommendations daily

---

## 1. Architecture Deep Dive

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    NETFLIX RECOMMENDATION SYSTEM                 │
│                                                                   │
│  USER INTERACTION                                                │
│  (Views, Searches, Ratings, Hover time)                         │
│         ↓                                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  DATA PIPELINE (Kafka + Spark)                           │  │
│  │  Real-time + Batch processing                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│         ↓                                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  OFFLINE TRAINING                                         │  │
│  │  • Collaborative Filtering (Matrix Factorization)        │  │
│  │  • Content-Based (Metadata features)                     │  │
│  │  • Deep Learning (Neural Networks)                       │  │
│  │  • XGBoost Ranking Models                                │  │
│  └──────────────────────────────────────────────────────────┘  │
│         ↓                                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  MODEL SERVING                                            │  │
│  │  • Pre-computed recommendations (batch)                   │  │
│  │  • Real-time ranking (< 100ms)                           │  │
│  │  • Contextual bandits (exploration)                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│         ↓                                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  PRESENTATION LAYER                                       │  │
│  │  • Personalized rows (Top Picks, Trending, etc.)        │  │
│  │  • Artwork personalization                                │  │
│  │  • Search ranking                                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│         ↓                                                         │
│  USER SEES RECOMMENDATIONS → Feedback Loop                      │
└─────────────────────────────────────────────────────────────────┘
```

### Component 1: Data Collection & Processing

**Data Sources:**

```
Explicit Signals (Strong):
- Thumbs up/down ratings
- Title completion rate
- Rewatches
- Add to "My List"

Implicit Signals (Weak but Abundant):
- Play vs skip
- Time of day watching
- Device type (TV, mobile, laptop)
- Pause, rewind, fast-forward
- Browsing behavior (hover time on titles)
- Search queries
- Genre preferences over time
```

**Real-Time Stream Processing (Kafka + Flink):**

```
User watches Stranger Things Episode 1:
  ↓
Event: {
  user_id: "user_123",
  title_id: "stranger_things_s1e1",
  action: "watch",
  duration: 3420 seconds (57 minutes),
  completion_rate: 0.95,
  timestamp: "2024-01-15T20:30:00Z",
  device: "smart_tv",
  country: "US"
}
  ↓
Kafka topic: user-viewing-events (millions/sec)
  ↓
Flink streaming jobs:
  1. Update user viewing history (append to profile)
  2. Increment title popularity counters
  3. Update real-time trends (trending now)
  4. Feature computation (user_sci_fi_watch_count++)
  5. Write to data warehouse (S3) for batch processing
  ↓
Low latency: <1 second from action to feature update
```

**Batch Processing (Spark):**

```
Nightly Jobs (2 AM - 6 AM):
  
1. User Profile Update (30 min):
   - Aggregate last 90 days of viewing
   - Compute genre affinities
   - Viewing patterns (binge vs casual)
   - Preferred actors, directors, languages
   
2. Title Embeddings (60 min):
   - Matrix factorization on full interaction matrix
   - Deep learning embeddings
   - Content features (metadata, cast, genre)
   
3. Similarity Computation (45 min):
   - Title-to-title similarities (collaborative)
   - User-to-user similarities (for user-based CF)
   - Pre-compute top 1000 similar titles per title
   
4. Candidate Generation (90 min):
   - Generate personalized candidate pools
   - Per user: 10K-100K candidate titles
   - Store in distributed cache (Redis/Cassandra)
   
5. Model Training (120 min):
   - Train ranking models (XGBoost, Neural Networks)
   - A/B test new model variants
   - Validate on held-out data
```

**Data Volume:**

```
Daily:
- 1 billion+ viewing events
- 10 billion+ UI interactions (clicks, hovers)
- 5TB of compressed event data
- 100TB+ in data warehouse (S3)

User profiles:
- 200M users × 100KB per profile = 20TB
- Includes: viewing history, preferences, embeddings
```

### Component 2: Recommendation Algorithms

**Stage 1: Candidate Generation (Retrieve 10K titles)**

**Algorithm 1: Collaborative Filtering (Matrix Factorization)**

```
Problem: User-Item interaction matrix (200M × 100K) is sparse

Solution: Matrix factorization
  R ≈ U × V^T
  
  U: User embeddings (200M × 100 dimensions)
  V: Item embeddings (100K × 100 dimensions)
  
  User embedding: [0.2, -0.4, 0.7, ..., 0.1]
    Represents: Genres, styles, preferences (latent factors)
    
  Title embedding: [0.3, -0.3, 0.6, ..., 0.2]
    Represents: Genre, tone, complexity (latent factors)
    
  Prediction: u_i · v_j (dot product)
    High score → User likely to watch title

Training:
  Algorithm: Alternating Least Squares (ALS)
  Data: 90 days of viewing history
  Objective: Minimize prediction error on observed interactions
  Regularization: Prevent overfitting
  
  Runtime: 2 hours on Spark cluster (1000 nodes)
  
Output:
  Per user: Top 1000 titles by score
  Update: Daily
```

**Algorithm 2: Content-Based Filtering**

```
For users who watched Sci-Fi shows, recommend similar:

Title features:
  - Genre: [Sci-Fi: 1.0, Drama: 0.3, Thriller: 0.5]
  - Actors: [Millie Bobby Brown, Finn Wolfhard, ...]
  - Director: [Duffer Brothers]
  - Year: 2016
  - Tags: [supernatural, 1980s, coming-of-age, mystery]
  - Cinematography style: dark, moody
  - Maturity rating: TV-14
  
User profile (aggregate of watched titles):
  - Sci-Fi affinity: 0.85 (loves sci-fi)
  - Drama affinity: 0.60
  - Preferred era: 2010s-2020s
  - Liked actors: [...]
  
Similarity: Cosine(user_profile, title_features)

Output:
  Per user: Top 2000 titles by similarity
  
Advantage:
  - Works for new titles (no collaborative data yet)
  - Explainable: "Because you watched X"
```

**Algorithm 3: Popularity & Trending**

```
Baseline candidates for all users:

1. Global trending (last 24 hours):
   - Most-watched titles globally
   - Viral content (social media buzz)
   - New releases (first week)
   
2. Regional trending:
   - Country-specific popularity
   - Language preferences
   - Cultural relevance
   
3. Recently added:
   - New to Netflix (past 7 days)
   - Boost new content visibility

Output:
  500 titles (same for everyone, but personalized ranking later)
  
Why include:
  - Cold start (new users have no history)
  - Serendipity (discover outside comfort zone)
  - Business goals (promote Netflix originals)
```

**Algorithm 4: Personal Viewing History**

```
Continue watching:
  - Partially watched titles (completion < 100%)
  - Sort by recency
  - Top 10 per user
  
Watch again:
  - Titles user rated highly
  - Completed shows user might rewatch
  - Comfort content (Friends, The Office)
  
Because you watched X:
  - Title-to-title similarity
  - If user watched Stranger Things → Recommend Dark, The OA
  - Collaborative filtering: "Users who watched X also watched Y"
```

**Combined Candidate Pool:**

```
Per user: ~10,000 candidates from:
  - Collaborative filtering: 1,000
  - Content-based: 2,000
  - Trending: 500
  - Continue watching: 10
  - Watch again: 100
  - Similar titles: 3,000
  - Genre-specific: 2,000
  - New releases: 500
  - A/B test variants: 890

Diverse sources ensure variety and coverage
```

### Stage 2: Ranking (Narrow to Top 100)

**Why ranking needed:**

```
Problem: 10K candidates too many to display
Solution: Rank by "probability user will watch AND enjoy"

Not just: P(click)
But: P(watch) × P(enjoy | watch) × P(retain subscriber | enjoy)

Optimizing for long-term engagement, not just clicks
```

**Ranking Model Architecture:**

```
Input Features (1000+):

User features:
  - Demographics: Age, country, language, subscription tier
  - Viewing history: Genres watched, avg session length, time of day
  - Engagement: Thumbs up/down, completion rates, rewatch rate
  - Embeddings: 100-dim latent factors from collaborative filtering
  - Recency: Days since last watch, hours since signup
  - Device: TV (lean-back) vs mobile (lean-forward)

Title features:
  - Metadata: Genre, year, maturity rating, runtime
  - Quality signals: Avg rating, completion rate (global)
  - Popularity: View count (7d, 30d), trending score
  - Embeddings: 100-dim latent factors
  - Content tags: 1000s of tags (moody, witty, cerebral, ...)
  - Production: Budget, cast popularity, Netflix original

Context features:
  - Time of day: Evening (long content) vs morning (short)
  - Day of week: Weekend (movies) vs weekday (episodes)
  - Recent activity: Just finished show → recommend similar
  - Device: TV (high-quality visuals) vs mobile (portable)
  - Network: WiFi (HD) vs cellular (lower quality)

Interaction features:
  - User × Title: Has user watched similar genres?
  - User × Actor: Does user like this actor?
  - User × Director: Affinity for director's style
  - Temporal: Is this trending NOW in user's region?

Model: XGBoost Gradient Boosted Trees
  - 1000 trees
  - Max depth: 6
  - Learning rate: 0.05
  - 1000+ features
  
  Why XGBoost:
    - Handles complex feature interactions
    - Fast inference (<10ms for 10K candidates)
    - Interpretable (feature importance)
    - Robust to feature engineering
    
Alternative: Deep Neural Network
  - For even more complex patterns
  - Slower inference (50-100ms)
  - Used for specific rows (Top Picks)

Training:
  - Data: 30 days of viewing sessions (billions of examples)
  - Positive: User watched >70% of title
  - Negative: User skipped title (was shown but not clicked)
  - Loss: Binary cross-entropy (watch vs skip)
  - Validation: Next 7 days of data (temporal split)
  - Retrain: Weekly

Output:
  - Score per (user, title) pair
  - Rank candidates by score
  - Top 100 per user
```

**Multi-Objective Optimization:**

```
Not just maximizing P(watch), but multiple objectives:

Primary: P(watch AND complete)
  - Watch >70% of title
  
Secondary: Diversity
  - Not all same genre
  - Mix of familiar + new
  - Variety in tone, style
  
Tertiary: Business goals
  - Promote Netflix originals (higher weight)
  - Boost new releases (cold start)
  - Regional content (local productions)
  
Constraints:
  - Catalog coverage (don't ignore long-tail)
  - Fairness (avoid filter bubbles)
  - Rights management (licensing expiry soon → promote)

Combined score:
  final_score = 0.7 × P(watch) + 0.2 × diversity + 0.1 × business
  
Tuned via A/B testing
```

### Stage 3: Presentation (Homepage Layout)

**Personalized Rows:**

```
Netflix homepage: 20-40 rows, each with 20-40 titles

Row types:

1. Top Picks for [User]:
   - Highest-ranked titles overall
   - Personalized thumbnail (artwork personalization)
   - Position: Top of page (most valuable real estate)

2. Trending Now:
   - Global or regional trending
   - Updated hourly
   - Social proof ("Everyone's watching this")

3. Continue Watching:
   - Incomplete titles
   - Sorted by recency
   - High conversion (user already started)

4. Because You Watched [Title]:
   - Similar titles to user's favorites
   - Explainable recommendations
   - Builds trust

5. Genre rows (Personalized):
   - Action & Adventure (if user watches action)
   - Korean Dramas (if user watches K-dramas)
   - Order and selection personalized per user

6. New Releases:
   - Recently added to Netflix
   - Personalized subset (relevant to user)

7. Netflix Originals:
   - Business priority (owned content)
   - Higher margin, exclusive

8. Watch Again:
   - Rewatchable content
   - Comfort shows

9. Award Winners:
   - Oscar/Emmy winners
   - Quality signal

10. Critically Acclaimed:
    - High ratings, awards
    - For discerning viewers
```

**Row Ordering (Meta-Ranking):**

```
Which rows to show? In what order?

Factors:
  - User intent (just logged in vs browsing for 5 min)
  - Time of day (evening → movie rows higher)
  - User history (binge-watcher → show episodes)
  - Device (TV → cinematic content higher)
  - Churn risk (at-risk users → more engaging content)

Algorithm:
  - Reinforcement learning (contextual bandits)
  - State: User profile + context
  - Action: Row order
  - Reward: Engagement (watch time, session length)
  
  Learn optimal row ordering per user type

Result:
  - Different users see different row orders
  - Even row contents vary (personalized)
```

**Artwork Personalization:**

```
Same title, different thumbnail for different users

Stranger Things:
  - User who loves character-driven → Show characters
  - User who loves suspense → Show creepy imagery
  - User who loves sci-fi → Show supernatural elements

A/B tested artwork variants:
  - 10-20 variants per title
  - Select based on user profile
  
Impact:
  - 20-30% higher click-through rate
  - Significant engagement boost

Algorithm:
  - Contextual bandit (Thompson Sampling)
  - Explore: Try different artworks
  - Exploit: Show best-performing artwork
  - Learn user preferences (character-focused vs scene-focused)
```

### Stage 4: Real-Time Ranking & Serving

**Serving Infrastructure:**

```
Request: User loads homepage
  ↓
Load Balancer (Route to nearest data center)
  ↓
API Gateway (Authentication, rate limiting)
  ↓
Recommendation Service (Microservice)
  ↓
┌─────────────────────────────────────────┐
│ REAL-TIME RANKING PIPELINE              │
│                                          │
│ 1. User Profile Retrieval (5ms)        │
│    - Redis/Cassandra                    │
│    - User embeddings, recent history    │
│                                          │
│ 2. Candidate Retrieval (10ms)          │
│    - Pre-computed candidates (batch)    │
│    - 10K titles per user                │
│                                          │
│ 3. Context Enrichment (5ms)            │
│    - Time of day, device, location      │
│    - Recent activity (session state)    │
│                                          │
│ 4. Feature Generation (20ms)           │
│    - User × Title features              │
│    - Real-time signals                  │
│    - 1000+ features per candidate       │
│                                          │
│ 5. Model Inference (30ms)              │
│    - XGBoost ranking (CPU)              │
│    - Score 10K candidates               │
│    - Batch inference (optimized)        │
│                                          │
│ 6. Diversification (10ms)              │
│    - Re-rank for diversity              │
│    - Business rules (promote originals) │
│                                          │
│ 7. Row Assembly (10ms)                 │
│    - Group into rows                    │
│    - Artwork selection (personalized)   │
│                                          │
│ 8. Caching (5ms)                       │
│    - Cache final recommendations        │
│    - TTL: 5 minutes                     │
│                                          │
│ Total: ~95ms (within 100ms budget) ✓   │
└─────────────────────────────────────────┘
  ↓
Response: Personalized homepage JSON
  ↓
Client (Web/Mobile App) renders UI
```

**Caching Strategy:**

```
Three-tier caching:

L1 Cache (Client-side):
  - Homepage cached for 5 minutes
  - Refresh on user action (scroll to bottom)
  - Avoids unnecessary API calls

L2 Cache (CDN - CloudFront):
  - Popular queries cached at edge
  - Reduces latency (serve from nearest POP)
  - TTL: 1 minute (frequent updates)

L3 Cache (Application - Redis):
  - User profiles: 10 minutes
  - Candidates: 1 hour (batch updates)
  - Model scores: 5 minutes (dynamic)

Cache hit rates:
  - Homepage: 60% (5-min TTL)
  - User profile: 90% (frequently accessed)
  - Candidates: 80% (updated hourly)

Cost savings:
  - 70% reduction in compute (avoid re-ranking)
  - Billions of requests/day
  - Significant infrastructure savings
```

**Scalability:**

```
Load:
  - 200M users
  - Assume 50% daily active (100M)
  - Avg 2 sessions/day = 200M requests/day
  - Peak: 5× average (evenings) = 1M requests/minute = 16K RPS

Infrastructure:
  - Kubernetes clusters (1000+ nodes)
  - Auto-scaling (scale up during peak hours)
  - Multi-region (US, EU, Asia)
  - Failover (if one region down, route to another)

Latency targets:
  - P50: 50ms
  - P95: 100ms
  - P99: 200ms
  
  Above targets → degrade gracefully:
    - Serve cached recommendations (stale OK)
    - Fall back to popularity-based (no personalization)
    - Never show empty page
```

---

## 2. Personalization at Scale

### Challenge: 200M Unique Experiences

**Scale of Personalization:**

```
Users: 200M
Titles: 100K
User × Title combinations: 20 trillion

Cannot pre-compute all:
  20 trillion × 4 bytes (score) = 80 TB just for scores
  
Solution: Hybrid approach
  - Pre-compute: User embeddings, title embeddings, candidates
  - Real-time: Rank candidates on-demand
```

### Personalization Dimensions

**1. Content Recommendations:**

```
What titles to show?
  - Personalized per user
  - Different genres, styles, tones
  - Based on viewing history

Example:
  User A: Sci-fi enthusiast → Stranger Things, Dark, The OA
  User B: Rom-com lover → Emily in Paris, Love is Blind
```

**2. Artwork Personalization:**

```
Same title, different thumbnail

Stranger Things:
  User who likes ensemble casts → Group photo
  User who likes lead actors → Millie Bobby Brown close-up
  User who likes suspense → Dark, mysterious imagery

10-20 variants per title
Select via contextual bandit (A/B test)
```

**3. Row Ordering:**

```
Which rows to prioritize?

Binge-watcher (watches 3+ episodes/session):
  - Continue Watching: Row 1
  - Because You Watched: Row 2
  - Trending: Row 3

Casual viewer (1 episode/session):
  - Top Picks: Row 1
  - Trending: Row 2
  - Continue Watching: Row 3

Varies per user behavior
```

**4. Search Ranking:**

```
User searches "comedy"

User A (watches dark comedies):
  1. BoJack Horseman
  2. Russian Doll
  3. The End of the F***ing World

User B (watches light comedies):
  1. Schitt's Creek
  2. The Good Place
  3. Brooklyn Nine-Nine

Same query, different results (personalized)
```

**5. Title Metadata:**

```
Genre tags personalized:

Title: Inception
  User A (action fan): "Action, Thriller, Heist"
  User B (sci-fi fan): "Sci-Fi, Mind-Bending, Cerebral"
  
Emphasize different aspects per user preference
```

### Personalization Algorithms

**Contextual Bandits:**

```
Problem: Exploration vs exploitation at scale

Traditional A/B test:
  - 50% see variant A, 50% see variant B
  - Slow (need weeks to reach significance)
  
Contextual bandits:
  - Adapt per user (context)
  - Explore more with uncertain users
  - Exploit more with well-understood users

Algorithm: Thompson Sampling
  1. Maintain belief distribution per (context, action)
  2. Sample from distribution (stochastic)
  3. Take sampled action
  4. Observe reward (watch time)
  5. Update belief (Bayesian update)

Result:
  - Faster learning (days vs weeks)
  - Personalized exploration (not one-size-fits-all)

Example: Artwork selection
  New user: Try 5 different artworks (explore)
  Known user: Show best-performing artwork (exploit)
```

**Multi-Armed Bandits (Row Ordering):**

```
20 possible rows, show top 10

Which 10 to show? In what order?

State: User profile (demographics, history, session context)
Arms: Row combinations (factorial many)
Reward: Session engagement (watch time, titles played)

Algorithm: Upper Confidence Bound (UCB)
  - Balance exploration (uncertainty) vs exploitation (mean reward)
  - UCB score = mean_reward + sqrt(log(n) / n_arm)
  - Choose rows with highest UCB

Result:
  - Learn optimal row ordering per user segment
  - Adapt to trending content (dynamic)
```

### Cold Start Problem

**Challenge: New users have no history**

**Solutions:**

**1. Demographics-based:**
```
Ask during signup:
  - Favorite genres (action, comedy, drama)
  - Favorite titles (pick 3)
  - Preferred language

Use demographics to initialize profile:
  - Age + country → likely genres
  - Similar users → likely preferences
```

**2. Popularity fallback:**
```
First session: Show trending content
  - Globally popular
  - High engagement rate
  - Broad appeal

Collect initial signals:
  - What they watch
  - What they skip
  - How long they watch
```

**3. Fast learning:**
```
After 1 title watched:
  - Use content-based (genre, cast)
  - Similar titles to first watch

After 3 titles:
  - Basic collaborative filtering
  - Enough to segment user

After 10 titles:
  - Full personalization
  - Rich profile built

Goal: Personalize within first session
```

**New Title Cold Start:**

```
Challenge: New release has no collaborative data

Solutions:
  1. Content-based (metadata, cast, genre)
  2. Transfer learning (similar titles' patterns)
  3. Boost in "New Releases" row (exposure)
  4. A/B test different positions
  5. Fast learning (collect data quickly)

Within 24 hours:
  - 10K-100K views
  - Enough for collaborative filtering
  - Integrate into main recommendations
```

---

## 3. A/B Testing Infrastructure

### Why A/B Testing Critical

```
Small changes = big impact at scale

Example:
  - Improve click-through rate by 1%
  - 200M users × 2 sessions/day = 400M sessions
  - 1% improvement = 4M additional clicks/day
  - More engagement = lower churn = millions in revenue
```

### A/B Test Framework

**Experimentation Platform:**

```
┌───────────────────────────────────────────────┐
│      NETFLIX EXPERIMENTATION PLATFORM         │
│                                                │
│  1. Experiment Definition                     │
│     - Hypothesis                              │
│     - Treatment variants (A vs B vs C)        │
│     - Success metrics                         │
│     - Sample size calculation                 │
│     - Duration estimate                       │
│                                                │
│  2. Traffic Allocation                        │
│     - Random assignment (user_id hash)        │
│     - Stratification (by country, device)     │
│     - Ramp-up (1% → 10% → 50%)               │
│                                                │
│  3. Monitoring                                │
│     - Real-time metrics dashboard             │
│     - Statistical significance tracking       │
│     - Guardrail metrics (check for harm)      │
│     - Anomaly detection (quality issues)      │
│                                                │
│  4. Analysis                                  │
│     - Bayesian or Frequentist                 │
│     - Multiple hypothesis correction          │
│     - Segment analysis (power users vs casual)│
│     - Long-term effects (retention)           │
│                                                │
│  5. Decision & Rollout                        │
│     - Ship if: stat sig + metric improvement  │
│     - Rollback if: degradation                │
│     - Gradual rollout (10% → 100%)           │
└───────────────────────────────────────────────┘
```

### Example A/B Test: Ranking Algorithm

**Hypothesis:**
```
New neural network ranking model (vs current XGBoost) will improve engagement
```

**Test Design:**

```
Control (A): XGBoost ranking (current production)
Treatment (B): Neural network ranking (new model)

Traffic split: 95% control, 5% treatment (cautious start)

Success metrics (primary):
  - Watch time per session (hours)
  - Titles played per session
  - 28-day retention rate

Guardrail metrics:
  - Latency (P95 < 150ms, don't degrade UX)
  - Error rate (< 0.1%, model failures)
  - Diversity (Simpson's index > baseline)

Sample size:
  - Need: 10M users per variant
  - At 5% traffic: 200M × 0.05 = 10M ✓
  - Duration: 2 weeks (capture weekend patterns)

Statistical test:
  - Two-sample t-test (or Bayesian equivalent)
  - Significance level: α = 0.05
  - Minimum detectable effect: 2% (practical significance)
```

**Results Analysis:**

```
Scenario 1: Clear winner
  Treatment B: +5% watch time (p < 0.001)
  → Ship to 100%

Scenario 2: Mixed results
  Treatment B: +3% watch time (p = 0.02) ✓
  But: +10% latency, -2% diversity ✗
  → Don't ship, optimize further

Scenario 3: No difference
  Treatment B: +0.5% watch time (p = 0.4, not significant)
  → Don't ship, not worth complexity

Scenario 4: Segmented effects
  Treatment B: 
    - Power users: +8% watch time ✓
    - Casual users: -2% watch time ✗
  → Ship only to power users (targeted rollout)
```

### Interleaving Experiments

**Problem with A/B:**
```
User sees only one variant (A or B)
Cannot compare directly
Need large sample size
```

**Interleaving solution:**
```
Show both variants to same user, mixed together

Example (Search ranking):
  User searches "comedy"
  
  Results shown:
    Pos 1: From model A
    Pos 2: From model B
    Pos 3: From model A
    Pos 4: From model B
    ...

  Metric: Which model's results get more clicks?
  
Benefits:
  - Within-user comparison (more sensitive)
  - Smaller sample size needed
  - Faster results (days vs weeks)

Used for: Search ranking, row ordering
```

### Multi-Armed Bandit A/B Tests

**Problem:** A/B test wastes traffic on losing variant

```
Traditional A/B:
  Week 1-4: 50% control, 50% treatment
  Even if treatment clearly worse after week 1,
  still show to 50% for 3 more weeks (bad UX)
```

**Bandit solution:**
```
Dynamic traffic allocation based on performance

Week 1: 50% control, 50% treatment
  → Treatment performing better

Week 2: 30% control, 70% treatment
  → Shift traffic to winner

Week 3: 10% control, 90% treatment
  → Almost all traffic to winner

Result:
  - Faster learning
  - Less opportunity cost (fewer users see inferior variant)
  - Still maintain exploration (10% to control)

Used for: High-stakes experiments, continuous optimization
```

### Guardrail Metrics

**Always monitor these (prevent harm):**

```
User experience:
  - Latency (P95, P99)
  - Error rate
  - Crash rate (mobile apps)
  - Buffering rate (video quality)

Engagement:
  - Session length (don't decrease)
  - Bounce rate (don't increase)
  - Return rate (7-day)

Business:
  - Subscription cancellations (churn)
  - Customer support tickets
  - Revenue (lifetime value)

Fairness:
  - Diversity (content variety)
  - Catalog coverage (long-tail titles)
  - Regional equity (not US-centric)

Example:
  New algorithm improves engagement by 5%
  But increases latency by 50%
  → Don't ship! (UX matters more)
```

---

## 4. Challenges and Solutions

### Challenge 1: Data Sparsity

**Problem:**
```
Interaction matrix: 200M users × 100K titles = 20 trillion cells
Observed interactions: 10 billion (~0.05% density)
99.95% of matrix is missing!

Challenge: How to predict for unseen (user, title) pairs?
```

**Solutions:**

**1. Matrix Factorization:**
```
Assume: R ≈ U × V^T (low-rank approximation)

Learn: Latent factors (100-dim embeddings)
  - Generalize to unseen pairs
  - Dot product predicts for any (user, title)

Limitation: Linear interactions only
```

**2. Deep Learning:**
```
Neural Collaborative Filtering:
  - Non-linear interactions
  - Learns complex patterns
  - Better than matrix factorization (empirically)

Architecture:
  User ID → Embedding → Dense layers
  Title ID → Embedding → Dense layers
  → Concatenate → Dense → Prediction
```

**3. Hybrid (Collaborative + Content):**
```
Collaborative: Uses interactions (sparse)
Content: Uses metadata (always available)

For sparse users: Rely more on content
For dense users: Rely more on collaborative

Weighted combination:
  score = α × collaborative + (1-α) × content
  
  α = f(user_activity)
    New user: α = 0.2 (mostly content)
    Active user: α = 0.8 (mostly collaborative)
```

### Challenge 2: The Cold Start Problem

**Three types:**

**1. New Users:**
```
Problem: No viewing history → Cannot personalize

Solution:
  - Ask during onboarding (favorite genres)
  - Show popular content (broad appeal)
  - Fast learning (personalize within first session)
  - Transfer learning (use demographics)
```

**2. New Titles:**
```
Problem: No interaction data → Cannot recommend collaboratively

Solution:
  - Content-based (metadata, cast, genre)
  - Boost in "New Releases" row
  - Transfer learning (similar titles' patterns)
  - Fast data collection (promote heavily first week)
```

**3. New Regions:**
```
Problem: Launching in new country, no local data

Solution:
  - Global models (transfer from similar regions)
  - Local popular content (charts, trending)
  - Cultural adaptation (local languages, genres)
  - Rapid localization (collect data fast)
```

### Challenge 3: Popularity Bias

**Problem:**
```
Popular titles get more views → More data → Recommended more → Even more popular

Rich get richer (Matthew effect)
Long-tail titles ignored (even if high quality)

Impact:
  - Reduced diversity
  - User filter bubbles
  - Unfair to niche content
```

**Solutions:**

**1. Regularization:**
```
Penalize popular titles in ranking:

adjusted_score = raw_score - β × log(popularity)

β: Tuned via A/B test
  - Higher β: More diversity
  - Lower β: More popular (safer)

Balance: User satisfaction vs diversity
```

**2. Exploration bonus:**
```
Occasionally recommend titles with high uncertainty:

score = exploitation_score + c × sqrt(uncertainty)

c: Exploration coefficient
  - Try titles with few views (uncertain predictions)
  - Learn their true quality
  - Update beliefs

Thompson Sampling (Bayesian)
```

**3. Diversity re-ranking:**
```
After initial ranking, re-rank for diversity:

Greedy diversification:
  1. Select top-ranked title
  2. Penalize similar titles (same genre, cast, style)
  3. Select next top-ranked (after penalty)
  4. Repeat until N titles selected

Result: Diverse set while maintaining quality
```

**4. Row-level diversity:**
```
Each row has different theme:
  - Row 1: Top Picks (familiar)
  - Row 2: Trending (social proof)
  - Row 3: Because You Watched (similar)
  - Row 4: Hidden Gems (niche)
  
Ensures variety across page
```

### Challenge 4: Shilling Attacks (Manipulation)

**Problem:**
```
Bad actors try to game recommendations:
  - Create fake accounts
  - Watch specific titles repeatedly
  - Inflate popularity/ratings

Goal: Promote their content artificially
```

**Detection:**

```
Anomaly signals:
  - Account creation patterns (many at once)
  - Viewing patterns (same title, same time)
  - Rating patterns (all 5 stars for one title)
  - IP addresses (many accounts, same IP)
  - Device fingerprints (same device)

ML model:
  - Classify accounts: Real vs Fake
  - Features: Viewing patterns, network effects
  - Label: Human review + heuristics

Action:
  - Ban fake accounts
  - Downweight suspicious interactions
  - Alert fraud team
```

**Prevention:**

```
Robust algorithms:
  - Don't over-weight individual signals
  - Require diverse interactions (many users, many titles)
  - Use implicit feedback (harder to fake than ratings)
  - Monitor for sudden spikes (anomaly detection)

Example:
  Title gets 10K views in 1 hour (suspicious)
  vs gradually 10K over 1 week (organic)
  
  Flag sudden spikes for review
```

### Challenge 5: Feedback Loops

**Problem:**
```
Recommendations influence what users watch
  → Users watch recommended titles
  → Model learns from these watches
  → Recommends similar titles even more
  → Self-reinforcing loop

Result:
  - Filter bubbles (stuck in same genre)
  - Reduced serendipity
  - Miss diverse interests
```

**Solutions:**

**1. Exploration:**
```
Occasionally recommend outside comfort zone:
  - Random titles (low probability)
  - Trending titles (social exploration)
  - A/B test variants (systematic exploration)

ε-greedy:
  With probability ε: Random recommendation
  With probability 1-ε: Best recommendation
  
  ε = 0.1 (10% exploration)
```

**2. Causal inference:**
```
Problem: Observational data (user watches what we recommend)
Solution: Causal models (what WOULD user watch if shown different content?)

Inverse Propensity Scoring:
  - Weight observations by inverse of probability shown
  - Corrects for selection bias
  - Unbiased learning from biased data

Counterfactual learning:
  - Estimate effect of showing different titles
  - Learn from exploration (random recommendations)
```

**3. Diversification:**
```
Don't just maximize P(watch)
Also consider:
  - Genre diversity (Simpson's index)
  - Temporal diversity (variety over time)
  - Serendipity (surprising but relevant)

Multi-objective optimization:
  score = 0.6 × P(watch) + 0.3 × diversity + 0.1 × serendipity
```

### Challenge 6: Real-Time vs Batch Trade-Off

**Tension:**

```
Batch models (nightly):
  - Use all historical data
  - Complex models (hours to train)
  - High accuracy
  - Stale (12-24 hours old)

Real-time models:
  - Use recent data only
  - Simple models (milliseconds to train)
  - Lower accuracy
  - Fresh (seconds old)
```

**Netflix's Hybrid Approach:**

```
Batch (nightly):
  - Train complex models (XGBoost, Neural Networks)
  - Compute user embeddings, title embeddings
  - Pre-compute candidate pools (10K per user)
  - Update user profiles (aggregated stats)

Real-time (sub-second):
  - Update short-term signals (continue watching, last search)
  - Context features (time of day, device)
  - Re-rank pre-computed candidates
  - Simple models (linear, trees)

Combine:
  - Batch provides foundation (personalized candidates)
  - Real-time adds context (session state, trends)
  
Result: Best of both worlds
```

---

## 5. Key Learnings and Takeaways

### 1. Simple Baselines Are Strong

```
Before complex deep learning:
  - Collaborative filtering (matrix factorization)
  - Content-based filtering (metadata)
  - Popularity baseline

Insight: Simple methods cover 80% of cases
Complex methods: Last 20% improvement

Recommendation: Start simple, add complexity incrementally
```

### 2. Data > Algorithms

```
More viewing data > Better algorithm

Example:
  Good algorithm + 10M interactions: 75% accuracy
  Simple algorithm + 100M interactions: 80% accuracy

Netflix advantage: Massive data (billions of interactions)

Implication: Focus on data collection, quality, coverage
```

### 3. Context Matters

```
Same user, different recommendations based on:
  - Time of day (evening: movies, morning: short videos)
  - Device (TV: cinematic, mobile: portable)
  - Day of week (weekend: binge, weekday: single episode)
  - Recent activity (just finished show: recommend similar)

Insight: Static recommendations insufficient
Need: Dynamic, contextual recommendations
```

### 4. Diversity = Long-Term Engagement

```
Short-term: Maximize immediate clicks
Long-term: Maintain diverse interests

Trade-off:
  - Too narrow: Filter bubble → Boredom → Churn
  - Too diverse: Irrelevant → Frustration → Churn

Solution: Balance exploitation (familiar) + exploration (new)

Measured: User diversity over 30 days (not single session)
```

### 5. Explain != Accurate

```
Explainable recommendations:
  "Because you watched X" (content-based)
  
Accurate recommendations:
  Complex ensembles (black box)

Tension: Transparency vs Performance

Netflix choice: Hybrid
  - Most recommendations: Accurate (black box)
  - Some rows: Explainable ("Because you watched")
  - Build trust through explanations, deliver value through accuracy
```

### 6. Infrastructure = Competitive Advantage

```
Not just algorithms, but systems:
  - Data pipelines (real-time + batch)
  - Serving infrastructure (< 100ms)
  - A/B testing platform (iterate fast)
  - Monitoring (detect issues early)

Insight: Algorithm is 20%, infrastructure is 80%

Differentiation: Speed of iteration (experiment, learn, ship)
```

### 7. Business Constraints Shape Design

```
Not just maximize engagement, but:
  - Promote Netflix Originals (owned content)
  - Manage licensing costs (expiring content)
  - Regional content balance (local productions)
  - Diversity (avoid filter bubbles)
  - Fairness (long-tail titles)

ML objective ≠ Business objective

Solution: Multi-objective optimization, constrained ranking
```

---

## 6. How I Would Improve It

### Improvement 1: Longer-Term Optimization

**Current:** Optimize for session engagement (watch time)

**Problem:**
```
May recommend addictive but low-quality content
Short-term gain, long-term churn

Example: Recommend sensational reality TV
  → High immediate engagement
  → User regrets (low satisfaction)
  → Eventually cancels subscription
```

**Proposed:**
```
Optimize for 90-day retention (not session watch time)

Objective:
  Maximize: P(user still subscribed in 90 days | recommendations)

Implementation:
  - Causal inference (estimate long-term effect)
  - Delayed reward (reinforcement learning)
  - Cohort analysis (retention by recommendation strategy)

Challenge: Attribution is hard (many factors affect retention)

Solution:
  - Randomized controlled trials (long-term A/B tests)
  - Instrumental variables (find causal factors)
  - Synthetic control (compare to similar users)
```

### Improvement 2: Social Recommendations

**Current:** Mostly individual preferences

**Missing:** Social proof, friend recommendations

**Proposed:**
```
"Your friends are watching..."
"Trending in your city..."
"Watch with friends" feature (synchronized viewing)

Benefits:
  - Social motivation (FOMO)
  - Discovery through trusted sources (friends)
  - Engagement (discuss with friends)

Implementation:
  - Friend graph (opt-in social connections)
  - Privacy-preserving (don't reveal what friends watch without consent)
  - Aggregate trends (city-level, anonymous)

Challenge: Privacy concerns (viewing history is sensitive)

Solution:
  - Opt-in only (explicit consent)
  - Anonymous aggregates (no individual data shared)
  - User control (choose what to share)
```

### Improvement 3: Multi-Modal Understanding

**Current:** Mostly metadata (genre, cast, tags)

**Missing:** Visual, audio, narrative understanding

**Proposed:**
```
Deep content understanding:
  - Scene analysis (computer vision)
    • Cinematography style (dark, bright, saturated)
    • Visual pace (fast cuts vs slow)
    • Setting (urban, rural, period)
  
  - Audio analysis (music, dialogue)
    • Tone (cheerful, somber, intense)
    • Music genre (orchestral, electronic)
    • Dialogue complexity (literary vs casual)
  
  - Narrative analysis (NLP on subtitles)
    • Plot structure (linear, non-linear)
    • Themes (redemption, revenge, coming-of-age)
    • Character development

Match to user preferences at deeper level
```

**Example:**
```
User likes:
  - Visually dark cinematography
  - Slow-burn pacing
  - Psychological themes

Current: Recommend based on genre (thriller)
Proposed: Recommend based on style (even if different genre)
  → Recommend art-house drama (matches style)
```

**Implementation:**
```
Models:
  - Video encoder (3D CNN, video transformers)
  - Audio encoder (mel-spectrogram CNN)
  - Text encoder (BERT on subtitles)

Training:
  - Contrastive learning (similar content → similar embeddings)
  - Supervised (predict user engagement from content features)

Inference:
  - Pre-compute embeddings (batch)
  - Serve from feature store
```

### Improvement 4: Conversational Recommendations

**Current:** Passive browsing, searching

**Proposed:** Interactive dialogue

```
Netflix: "Looking for something to watch?"
User: "Yeah, something light and funny"
Netflix: "Comedy? Any preferences on style?"
User: "British humor, witty dialogue"
Netflix: "How about The Office (UK), Fleabag, or Derry Girls?"
User: "I've seen Fleabag. What else?"
Netflix: "Based on that, try Killing Eve (dark comedy) or The End of the F***ing World"

Benefits:
  - Natural interaction (like talking to friend)
  - Clarification (narrow down preferences)
  - Discovery (guided exploration)

Implementation:
  - LLM-based dialogue (GPT, Claude)
  - RAG (retrieve relevant titles)
  - Preference elicitation (ask clarifying questions)
  - Memory (remember conversation context)

Challenge: Latency (LLM inference slow)

Solution:
  - Pre-compute embeddings (fast retrieval)
  - Smaller models (distilled, fine-tuned)
  - Caching (common queries)
  - Streaming responses (progressive rendering)
```

### Improvement 5: Watchlist Intelligence

**Current:** "My List" is static (user manually adds)

**Proposed:** Smart watchlist management

```
Auto-add:
  - Titles user is likely to watch (>80% probability)
  - Expiring soon (licensing ends)
  - Trending (friends watching)

Auto-remove:
  - Titles user added but never watches (stale interest)
  - Titles user unlikely to watch anymore (preferences changed)

Prioritize:
  - Sort by: Likelihood to watch, expiring soon, mood match

Notify:
  - "3 titles expiring this week"
  - "New season of show you liked"

Benefits:
  - Reduce cognitive load (don't manage list manually)
  - Increase engagement (relevant suggestions)
  - FOMO (expiring content)
```

---

## 7. Practice Interview: "Design Netflix Recommendations from Scratch"

### Interview Framework (40-minute interview)

**Phase 1: Requirements Gathering (5 min)**

```
Clarify scope:
  Q: "What are we optimizing for?"
  A: User engagement (watch time), long-term retention

  Q: "Scale?"
  A: 200M users, 100K titles, billions of interactions

  Q: "Latency requirements?"
  A: Homepage load <100ms, real-time updates

  Q: "Cold start?"
  A: New users daily, new titles weekly

  Q: "Constraints?"
  A: Cost-effective, interpretable, fair (diversity)
```

**Phase 2: High-Level Design (10 min)**

```
1. Data Collection
   - Viewing events (Kafka)
   - User profiles (batch updates)
   - Real-time signals (session state)

2. Offline Training
   - Collaborative filtering (ALS)
   - Content-based (metadata)
   - Ranking models (XGBoost)

3. Online Serving
   - Pre-computed candidates (10K per user)
   - Real-time ranking (<100ms)
   - Personalized rows

4. Feedback Loop
   - A/B testing
   - Monitoring
   - Retraining (weekly)

[Draw architecture diagram on whiteboard]
```

**Phase 3: Deep Dive on Ranking (15 min)**

```
Interviewer: "How do you rank 10K candidates in <100ms?"

Answer:
  Two-stage funnel:

  Stage 1: Candidate Generation (batch, nightly)
    - Collaborative filtering: 1K titles
    - Content-based: 2K titles
    - Trending: 500 titles
    - Continue watching: 10 titles
    - Total: ~10K candidates per user
    - Pre-computed, stored in Redis

  Stage 2: Real-time Ranking (online, <100ms)
    - Fetch candidates from Redis (5ms)
    - Fetch user profile (5ms)
    - Context features (time, device) (5ms)
    - Feature generation (20ms)
    - Model inference (XGBoost, 10K scores) (30ms)
    - Diversification (10ms)
    - Total: ~75ms ✓

  Why XGBoost for ranking:
    - Fast inference (30ms for 10K candidates)
    - Handles complex interactions (1000+ features)
    - Interpretable (feature importance)
    - Robust (production-tested)

  Features (~1000):
    User: Demographics, viewing history, embeddings
    Title: Metadata, popularity, embeddings
    Context: Time, device, session state
    Interaction: User×Title (affinity scores)

  Training:
    - Positive: User watched >70% of title
    - Negative: User skipped (shown but not clicked)
    - Loss: Binary cross-entropy
    - Data: 30 days of interactions (billions)
    - Retrain: Weekly

  Alternative: Neural network
    - More accurate but slower (100ms)
    - Use for subset (Top Picks row only)
```

**Phase 4: Handling Challenges (5 min)**

```
Interviewer: "How do you handle cold start?"

Answer:
  Three scenarios:

  1. New user:
     - Ask during onboarding (favorite genres)
     - Show popular content (trending, high-rated)
     - Fast learning (personalize after 1 watch)

  2. New title:
     - Content-based (use metadata, cast, genre)
     - Boost in "New Releases" row
     - Fast data collection (promote heavily first week)

  3. New region (e.g., launching in India):
     - Transfer learning (global models)
     - Local popular content (Bollywood)
     - Cultural adaptation (language, genres)
     - Rapid localization (collect data fast)

Interviewer: "How do you ensure diversity?"

Answer:
  Multi-pronged:

  1. Regularization (penalize popular titles)
  2. Exploration bonus (try uncertain titles)
  3. Diversity re-ranking (greedy diversification)
  4. Row-level diversity (different themes per row)
  5. Monitoring (diversity metrics in A/B tests)

  Trade-off: Accuracy vs diversity
    - Measure: Simpson's diversity index
    - Target: >0.6 (60% of titles from different genres)
    - A/B test: Ensure diversity doesn't hurt engagement
```

**Phase 5: Scale and Trade-offs (5 min)**

```
Interviewer: "How do you scale to 200M users?"

Answer:
  Horizontal scaling:

  Data:
    - Kafka: Partitioned by user_id
    - Spark: 1000 nodes for batch processing
    - Redis: Sharded (100 shards, 10GB each)

  Serving:
    - Kubernetes: Auto-scaling (1000+ pods)
    - Multi-region: US, EU, Asia (latency)
    - CDN: Cache at edge (reduce backend calls)

  Caching:
    - User profiles: 10-min TTL (90% hit rate)
    - Candidates: 1-hour TTL (batch updates)
    - Rankings: 5-min TTL (dynamic)

  Optimization:
    - Batch inference (10K candidates together)
    - Model quantization (FP32 → FP16, 2× faster)
    - Pre-computation (candidates, embeddings)

Interviewer: "Accuracy vs latency trade-off?"

Answer:
  Tiered models:

  Fast path (<50ms):
    - XGBoost ranking
    - Pre-computed candidates
    - Most users (90%)

  Slow path (<200ms):
    - Neural network ranking
    - Dynamic candidate generation
    - High-value users (10%, at-risk of churn)

  Degrade gracefully:
    - If latency >100ms: Serve cached recommendations
    - If cache miss: Serve popularity-based (no personalization)
    - Never show empty page (always have fallback)

Cost vs accuracy:
  - Full personalization: $10/user/year
  - Hybrid (batch + real-time): $5/user/year
  - Popularity only: $0.50/user/year

  Choose: Hybrid (balances cost and quality)
```

---

## Summary: Key Interview Takeaways

**System Design Principles:**

1. **Two-stage funnel** (candidate generation + ranking)
   - Stage 1: Cast wide net (10K candidates, batch)
   - Stage 2: Precise ranking (top 100, real-time)

2. **Hybrid batch + real-time**
   - Batch: Complex models, full data (slow but accurate)
   - Real-time: Simple models, recent data (fast but noisy)

3. **Multi-algorithm ensemble**
   - Collaborative filtering (interaction data)
   - Content-based (metadata, cold start)
   - Popularity (fallback, new users)

4. **Personalization everywhere**
   - Content (what to show)
   - Artwork (how to show)
   - Rows (where to show)
   - Order (priority)

5. **A/B testing culture**
   - Experiment everything
   - Data-driven decisions
   - Iterate fast (weekly releases)

**Trade-offs to Discuss:**

- **Accuracy vs Latency:** Complex model (accurate) vs simple model (fast)
- **Accuracy vs Diversity:** Narrow (accurate) vs broad (diverse)
- **Personalization vs Privacy:** Granular (creepy) vs coarse (safe)
- **Exploitation vs Exploration:** Known (safe) vs unknown (risky)
- **Short-term vs Long-term:** Immediate engagement vs retention

**Red Flags to Avoid:**

- ❌ "Use deep learning for everything" (overkill)
- ❌ Ignore latency constraints (real-time matters)
- ❌ No cold start strategy (new users exist)
- ❌ No diversity (filter bubbles)
- ❌ No A/B testing (how do you know it works?)

**Strong Signals to Show:**

- ✅ Start simple, add complexity (baseline → advanced)
- ✅ Consider scale (200M users)
- ✅ Discuss trade-offs (pros/cons of each approach)
- ✅ Mention A/B testing (data-driven)
- ✅ Production concerns (latency, cost, monitoring)

---

**END OF DAY 46**
