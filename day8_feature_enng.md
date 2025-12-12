# Day 8: Feature Engineering at Scale

## Overview

Feature engineering is the **most impactful** part of ML systems. A mediocre model with great features beats a great model with mediocre features. At scale, feature engineering becomes a data engineering challenge involving pipelines, storage, versioning, and real-time computation.

---

## 1. The Feature Engineering Challenge at Scale

### What Makes It Hard?

**Volume:**
- Billions of raw events → Millions of features
- 10K predictions/second requires 10K feature retrievals/second
- Historical features for training: Petabytes of data

**Velocity:**
- Real-time features must compute in <50ms
- Batch features process billions of rows nightly
- Late-arriving data needs reconciliation

**Variety:**
- Streaming features (real-time aggregates)
- Batch features (complex computations)
- Static features (user profiles)
- External features (weather, stock prices)

**Consistency:**
- Training uses batch features
- Serving uses real-time features
- Train-serve skew causes performance degradation

**Versioning:**
- Features evolve over time
- Model trained on v1 features
- Production must use same v1 features
- Need to reproduce historical feature values

---

## 2. Feature Types and Computation Patterns

### 2.1 Static Features

**Definition:** Features that rarely change

**Examples:**
- User demographics (age, gender, location)
- Account metadata (account_type, signup_date)
- Product catalog (product_category, brand)

**Computation:**
- Batch: Loaded from database nightly
- Real-time: Retrieved from cache (Redis)
- Update frequency: Daily or on-demand

**Storage:**
- Offline: Data warehouse tables
- Online: Redis/DynamoDB (key-value)

### 2.2 Batch Features

**Definition:** Complex features computed periodically from historical data

**Examples:**
- User lifetime value (complex calculation)
- 90-day transaction patterns
- Merchant risk scores (requires global aggregation)
- User-merchant affinity (collaborative filtering)

**Computation:**
- Spark/Flink batch jobs (nightly/weekly)
- Expensive computations acceptable (offline)
- Can use complex algorithms (clustering, factorization)

**Storage:**
- Offline: Parquet files in S3
- Online: Pre-computed and cached in Redis

**Why Batch?**
- Too expensive for real-time (>100ms)
- Requires historical data not available in streaming
- Needs global context (all users, all merchants)

### 2.3 Streaming Features (Real-Time Aggregates)

**Definition:** Features computed from recent events

**Examples:**
- Transaction count (last 1h, 24h, 7d)
- Spending totals (last 24h, 7d, 30d)
- Distinct merchants visited (last 7d)
- Velocity metrics (transactions per hour)

**Computation:**
- Flink/Kafka Streams (continuous)
- Sliding/tumbling windows
- Incremental updates (<10ms)

**Storage:**
- Offline: Time-series tables
- Online: Redis with TTL

**Why Streaming?**
- Must be current (seconds old)
- Cannot wait for batch job
- User behavior changes rapidly

### 2.4 On-Demand Features

**Definition:** Features computed at prediction time from request data

**Examples:**
- Distance from home (requires current transaction location)
- Time since last transaction (requires current timestamp)
- Deviation from average (requires current amount)
- Day of week, hour of day (temporal features)

**Computation:**
- Computed during inference (<5ms)
- Simple calculations only
- No external data dependencies

**Storage:**
- Not stored (computed fresh each time)
- Ephemeral (exists only during prediction)

**Why On-Demand?**
- Depends on current request context
- Too simple to pre-compute
- Always fresh (no staleness)

---

## 3. Feature Store Architecture

### What is a Feature Store?

A **centralized system** for:
1. **Defining** features (code/config)
2. **Computing** features (pipelines)
3. **Storing** features (online + offline)
4. **Serving** features (low latency retrieval)
5. **Versioning** features (reproducibility)
6. **Monitoring** features (quality)

### Why Not Just Database + Code?

**Without Feature Store:**
- Features computed differently in training vs serving
- No versioning (cannot reproduce historical features)
- Duplicate feature logic across teams
- No monitoring or validation
- Manual consistency checks

**With Feature Store:**
- Single source of truth for feature definitions
- Automatic train-serve consistency
- Reusable features across models
- Built-in monitoring and validation
- Feature lineage tracking

---

## 4. Feature Store Components

### 4.1 Feature Registry (Metadata)

**Purpose:** Catalog of all features

**Contains:**
- Feature definitions (what it computes)
- Feature schema (data type, constraints)
- Compute logic (batch/streaming/on-demand)
- Ownership (who owns this feature)
- Dependencies (upstream features/data)
- Version history (changes over time)
- Usage tracking (which models use it)

**Example Entry:**
```
Feature: user_transaction_count_24h

Description: Number of transactions by user in last 24 hours
Type: Integer
Owner: fraud-detection-team
Compute: Streaming (Flink)
Source: transactions_stream (Kafka)
Online: Yes (Redis)
Offline: Yes (S3)
Created: 2024-01-15
Version: v2 (changed aggregation window from 1h to 24h)
Used by: fraud_model_v3, spending_insights_model
```

### 4.2 Offline Store (Historical Features)

**Purpose:** Training data and batch analytics

**Storage:**
- S3 (Parquet files) - most common
- Snowflake/BigQuery - for SQL access
- Delta Lake - for ACID properties

**Organization:**
```
s3://features/
├── user_features/
│   ├── date=2024-12-01/
│   │   └── features.parquet
│   ├── date=2024-12-02/
│   │   └── features.parquet
│   └── date=2024-12-03/
│       └── features.parquet
├── transaction_features/
│   └── date=2024-12-01/
│       └── features.parquet
└── aggregate_features/
    └── date=2024-12-01/
        └── features.parquet
```

**Access Pattern:**
- Point-in-time queries: "Get features as they were on 2024-06-15"
- Time travel: Reproduce historical feature values
- Batch retrieval: Get millions of feature sets for training

**Key Capability: Time Travel**
```
Query: Get user_123's features as of 2024-06-15

Feature Store retrieves:
- Static features: Latest before 2024-06-15
- Aggregates: Computed from data up to 2024-06-15
- No future data leakage

This is critical for training accuracy!
```

### 4.3 Online Store (Real-Time Features)

**Purpose:** Low-latency feature serving (<10ms)

**Storage:**
- Redis - most common (in-memory)
- DynamoDB - fully managed
- Cassandra - high availability

**Organization:**
```
Redis keys:
user:123:features → {age: 35, account_type: "premium", ...}
user:123:agg:24h → {txn_count: 5, total_spend: 420.50}
user:123:agg:7d → {txn_count: 18, total_spend: 1420.50}
merchant:456:features → {category: "restaurant", risk: 0.15}
```

**Access Pattern:**
- Key-value lookup by entity ID
- Sub-10ms latency required
- High throughput (10K+ requests/sec)

**Synchronization with Offline:**
- Batch sync: Nightly load from offline store
- Streaming sync: Real-time updates from Kafka/Flink
- Hybrid: Static features (batch), aggregates (streaming)

### 4.4 Feature Pipelines

**Purpose:** Compute features and populate stores

**Types:**

**Batch Pipeline (Spark/Airflow):**
```
Schedule: Daily 2 AM

Steps:
1. Read raw data from S3/warehouse
2. Compute complex features (joins, aggregations)
3. Write to offline store (S3)
4. Sync subset to online store (Redis)
5. Validate feature quality
6. Update feature registry

Duration: 1-2 hours
Data volume: Billions of rows
```

**Streaming Pipeline (Flink/Kafka Streams):**
```
Continuous processing

Steps:
1. Consume from Kafka topics
2. Compute windowed aggregates
3. Update online store (Redis) - incremental
4. Append to offline store (S3) - micro-batch
5. Emit data quality metrics

Latency: <1 second
Throughput: 10K events/sec
```

**On-Demand Pipeline:**
```
Triggered at prediction time

Steps:
1. Receive prediction request
2. Fetch stored features (Redis)
3. Compute derived features (distance, ratios)
4. Combine into feature vector
5. Return to model

Latency: <5ms
No persistent storage
```

---

## 5. Feature Store Solutions Comparison

### 5.1 Feast (Open Source)

**Architecture:**
- Feature registry: Git/S3 (feature definitions in YAML)
- Offline store: BigQuery/Snowflake/S3
- Online store: Redis/DynamoDB/Datastore
- SDK: Python/Go

**Strengths:**
- Simple to set up (lightweight)
- Cloud-agnostic
- Good documentation
- Active community

**Weaknesses:**
- No built-in feature computation (bring your own pipelines)
- Limited monitoring/observability
- Manual deployment of features
- No native streaming support

**Best For:**
- Small-to-medium teams
- Existing data pipelines
- Simple feature workflows
- Cost-conscious projects

**Example Use Case:**
Small fintech startup with 50K users, batch features only, running on AWS

### 5.2 Tecton (Enterprise SaaS)

**Architecture:**
- Fully managed platform
- Feature registry: Web UI + API
- Offline store: Tecton-managed (S3/Snowflake)
- Online store: Tecton-managed (DynamoDB)
- Compute: Spark/Flink (managed)

**Strengths:**
- Complete end-to-end platform
- Built-in streaming support (Flink)
- Excellent monitoring/alerting
- Feature serving infrastructure included
- Point-in-time correctness guaranteed
- Automatic train-serve consistency

**Weaknesses:**
- Expensive ($50K-500K+/year)
- Vendor lock-in
- Less customization
- Overkill for simple use cases

**Best For:**
- Large enterprises
- Real-time ML at scale
- Teams lacking infrastructure expertise
- Regulated industries (compliance features)

**Example Use Case:**
Large bank with 10M customers, real-time fraud detection, strict compliance

### 5.3 Hopsworks (Open Source + Enterprise)

**Architecture:**
- Feature registry: Python SDK + UI
- Offline store: Hive/S3/Snowflake
- Online store: RonDB (custom in-memory DB)
- Compute: Spark/Flink/Python

**Strengths:**
- Feature engineering in notebooks
- Strong versioning/lineage
- Built-in feature validation
- Good UI for exploration
- Can self-host (enterprise version)

**Weaknesses:**
- Smaller community than Feast
- RonDB learning curve
- Less mature than Tecton
- Documentation gaps

**Best For:**
- Data science teams (notebook-first)
- Organizations wanting self-hosted solution
- Teams prioritizing feature exploration

**Example Use Case:**
Mid-size e-commerce company, data scientists prefer notebooks, hybrid cloud

### 5.4 Custom-Built (Roll Your Own)

**When to Build:**
- Unique requirements not met by existing tools
- Very large scale (>100K predictions/sec)
- Tight integration with existing infrastructure
- Cost optimization (avoid SaaS fees)

**Components Needed:**
- Metadata registry (database + API)
- Batch compute (Spark/Airflow)
- Streaming compute (Flink/Kafka)
- Online store (Redis cluster)
- Offline store (S3 + Parquet)
- Feature serving API (FastAPI/gRPC)
- Monitoring (Prometheus/Grafana)

**Cost:**
- Development: 6-12 months, 2-4 engineers
- Maintenance: 1-2 engineers ongoing
- Infrastructure: $10K-50K/month

**Example:**
Large tech company (Google, Meta, Uber) with ML platform team

---

## 6. Online vs Offline Features - The Core Problem

### The Train-Serve Skew Problem

**Training (Offline):**
- Uses batch-computed features from data warehouse
- Features computed with complex SQL/Spark jobs
- Can use days/weeks of historical data
- No latency constraints (can take hours)

**Serving (Online):**
- Uses real-time features from cache/API
- Features must compute in <10ms
- Only recent data available
- Strict latency requirements

**The Problem:**
If features computed differently, model performance degrades!

**Example: Transaction Count Feature**

**Training (Offline - Spark):**
```
SELECT 
  user_id,
  COUNT(*) as transaction_count_24h
FROM transactions
WHERE timestamp >= (prediction_timestamp - INTERVAL 24 HOURS)
  AND timestamp < prediction_timestamp
GROUP BY user_id
```
Window: Exactly 24 hours before prediction

**Serving (Online - Flink):**
```
Sliding window: 24 hours, slide every 1 minute
Updates incrementally on each transaction
May include transactions up to 1 minute newer than prediction
```
Window: Approximately 24 hours, may be slightly different

**Result:**
- Training: transaction_count_24h = 12
- Serving: transaction_count_24h = 13 (one more transaction arrived)
- Model sees inconsistent input → degraded performance

### Solution: Feature Store Ensures Consistency

**Single Definition:**
```
Feature: user_transaction_count_24h
Computation:
  SELECT COUNT(*) FROM transactions
  WHERE user_id = {user_id}
  AND timestamp >= NOW() - INTERVAL 24 HOURS
  AND timestamp < NOW()
  
Offline Implementation:
  Spark job using exact SQL above
  
Online Implementation:
  Flink sliding window (24h, slide 1m)
  Materialized to Redis
  
Validation:
  Daily check: offline vs online values should match within 1%
```

**Feature Store Enforces:**
1. Same computation logic (shared code/config)
2. Same time windows
3. Same data sources
4. Same transformations
5. Validation that offline ≈ online

---

## 7. Feature Versioning and Lineage

### Why Version Features?

**Reproducibility:**
- Model trained on June 1 with feature set v1
- On August 1, need to retrain with same features
- Feature logic may have changed → need v1 definition

**A/B Testing:**
- Test model with old features vs new features
- Need both feature versions in production simultaneously

**Debugging:**
- Model performance degraded on July 15
- What features changed around that time?
- Need to trace feature lineage

**Compliance:**
- Regulators ask "How did you compute this prediction?"
- Need to reproduce exact features used for that prediction
- Audit trail required

### Versioning Strategy

**Approach 1: Schema Versioning**
```
Feature: user_avg_transaction_amount

v1 (2024-01-01 to 2024-06-30):
  Computation: AVG(amount) over all time
  
v2 (2024-07-01 onwards):
  Computation: AVG(amount) over last 90 days
  (Changed to be more responsive to recent behavior)

Storage:
  Offline: features/user/v1/date=2024-06-15/
  Offline: features/user/v2/date=2024-07-01/
  
  Online: user:123:features:v1
  Online: user:123:features:v2
```

**Approach 2: Semantic Versioning**
```
Feature: transaction_amount_deviation

v1.0.0 (Initial):
  (amount - user_avg) / user_std

v1.1.0 (Added bounds):
  CLIP((amount - user_avg) / user_std, -3, 3)
  
v2.0.0 (Breaking change):
  (amount - user_median) / user_iqr
  (Changed from mean/std to median/IQR for robustness)

Models must specify exact version:
  fraud_model_v5 uses transaction_amount_deviation==v1.1.0
```

**Approach 3: Immutable Features**
```
Philosophy: Never modify features, always create new ones

Old feature:
  user_transaction_count_7d
  
New feature (don't modify old one):
  user_transaction_count_7d_v2
  or
  user_transaction_count_7d_excluding_refunds

Benefits:
  - No breaking changes
  - Perfect reproducibility
  - Clear naming

Drawbacks:
  - Feature proliferation
  - More storage
  - Need cleanup process
```

### Feature Lineage Tracking

**What is Lineage?**
Tracking the full data flow: raw data → transformations → feature → model → prediction

**Example Lineage:**
```
Prediction: transaction_123 flagged as fraud

Model: fraud_model_v5 (trained 2024-11-01)
  ↓
Features used (point-in-time: 2024-12-12 10:30:45):
  - user_transaction_count_24h: 5
      ↓ (computed by Flink job: streaming_agg_v3)
      ↓ (data source: transactions_stream, Kafka topic)
      ↓ (raw events: txn_abc, txn_def, txn_ghi, txn_jkl, txn_mno)
      
  - user_avg_transaction_amount: $85.30
      ↓ (computed by Spark job: batch_features_v7, run 2024-12-12 02:00)
      ↓ (data source: transactions table, S3)
      ↓ (raw data: 342 historical transactions)
      
  - distance_from_home: 2,451 miles
      ↓ (computed on-demand at inference)
      ↓ (inputs: current_lat=34.05, current_lon=-118.24, 
                  home_lat=40.71, home_lon=-74.00)
      ↓ (home location from: user_profile table, updated 2024-12-10)

Can trace back to every raw event that contributed!
```

**Why This Matters:**
- Debugging: "Why did model predict fraud?"
- Compliance: "Show me exactly what data was used"
- Data quality: "Bad sensor caused wrong feature values"
- Impact analysis: "If I change this pipeline, what breaks?"

**How Feature Stores Track Lineage:**

**1. Metadata Logging:**
Every feature computation logs:
- Input data sources (tables, streams, files)
- Transformation logic (code version, config)
- Execution metadata (timestamp, job ID, cluster)
- Output locations (S3 paths, Redis keys)

**2. DAG Representation:**
Feature dependencies form directed acyclic graph (DAG):
```
raw_transactions → user_transaction_history → user_avg_amount → model
                ↘ user_transaction_count_24h → model
```

**3. Versioned Artifacts:**
- Feature definitions: Git commits
- Computed features: Immutable files (append-only)
- Pipelines: Docker images with tags

**4. Query Interface:**
```
feature_store.get_lineage(
  feature="user_avg_transaction_amount",
  entity_id="user_123",
  timestamp="2024-12-12T10:30:45Z"
)

Returns:
  - Feature value: $85.30
  - Computation: batch_features_v7
  - Executed: 2024-12-12T02:00:00Z
  - Data source: s3://transactions/date=2024-12-11/
  - Upstream features: None (leaf feature)
  - Downstream usage: fraud_model_v5, spending_model_v2
```

---

## 8. Feature Monitoring and Validation

### Why Monitor Features?

**Data Quality Issues:**
- Upstream data source fails → features missing/incorrect
- Pipeline bug → features computed wrong
- Data drift → feature distributions change

**Impact:**
- Silent model degradation
- Incorrect predictions
- Business impact before anyone notices

**Philosophy:**
Monitor features, not just model performance. Catch issues early!

### Monitoring Layers

**Layer 1: Availability**
- Are features being computed?
- Are pipelines running successfully?
- Is online store responding?

**Metrics:**
- Pipeline success rate: 100% (alerts if <99%)
- Feature freshness: Last update <10 minutes ago
- Online store latency: P95 <10ms

**Layer 2: Completeness**
- Are features present for all entities?
- Missing value rates acceptable?

**Metrics:**
- Coverage: % of users with features (target: >99%)
- Missing rate per feature: <1%
- Null count trends

**Layer 3: Validity**
- Are feature values in expected ranges?
- Data types correct?
- Constraints satisfied?

**Validation Rules:**
```
Feature: user_transaction_count_24h
  - Type: Integer
  - Range: [0, 1000] (more than 1000 in 24h is impossible)
  - Non-null: Required

Feature: transaction_amount
  - Type: Float
  - Range: [0, 100000] (transactions >$100K are rare)
  - Precision: 2 decimal places

Feature: user_age
  - Type: Integer
  - Range: [18, 120] (valid age range)
  - Non-negative: Yes
```

**Layer 4: Consistency**
- Offline vs online feature values match?
- Feature relationships maintained?

**Checks:**
```
Daily validation:
  For sample of 10,000 users:
    1. Compute features offline (batch)
    2. Fetch features online (Redis)
    3. Compare values
    4. Alert if >1% mismatch
    
Example discrepancy:
  user_123:
    Offline: transaction_count_24h = 12
    Online:  transaction_count_24h = 13
    Difference: 8.3%
    Verdict: Acceptable (likely timing difference)
```

**Layer 5: Distribution Monitoring**
- Statistical properties stable?
- Detecting data drift?

**Statistical Tests:**
```
For each feature, track:
  - Mean, median, std deviation
  - Min, max, percentiles (P5, P25, P50, P75, P95)
  - Distinct value count (cardinality)
  - Distribution shape (histogram)

Compare daily:
  - Today vs yesterday
  - Today vs last 7 days
  - Today vs same day last week (seasonality)

Drift Detection:
  - KS test (distribution change)
  - PSI (Population Stability Index)
  - Alert if PSI >0.25 (significant drift)
```

**Example Distribution Change:**
```
Feature: transaction_amount

Last 7 days:
  Mean: $67.50
  Std: $42.30
  P50: $55.00
  P95: $150.00

Today:
  Mean: $98.20 (↑ 45%)  ← Significant change!
  Std: $58.10
  P50: $82.00
  P95: $210.00

Root Cause: Black Friday sales (legitimate drift)
Action: Update baseline, do not alert
```

### Automated Data Quality Framework

**Great Expectations Integration:**

Many feature stores integrate with Great Expectations for validation.

**Example Expectation Suite:**
```
Feature: user_transaction_count_24h

Expectations:
  1. expect_column_values_to_be_between(min=0, max=1000)
  2. expect_column_values_to_not_be_null
  3. expect_column_mean_to_be_between(min=0.5, max=5.0)
  4. expect_column_stdev_to_be_between(min=0, max=10)
  5. expect_column_values_to_be_of_type(integer)

Validation Schedule: Every feature computation
Failure Action: Alert + block deployment (critical) or log (warning)
```

**Validation Results:**
```
Validation Run: 2024-12-12 02:30:15
Feature: user_transaction_count_24h
Records Validated: 30,000,000

Results:
  ✓ expect_column_values_to_be_between: 100% pass
  ✓ expect_column_values_to_not_be_null: 99.97% pass (10K nulls)
  ✗ expect_column_mean_to_be_between: FAILED
      Expected: [0.5, 5.0]
      Actual: 7.2 (↑ above threshold!)
  ✓ expect_column_stdev_to_be_between: 100% pass
  ✓ expect_column_values_to_be_of_type: 100% pass

Action: Alert data quality team, investigate spike in transaction volume
```

---

## 9. Real-Time Feature Computation

### The Real-Time Challenge

**Requirements:**
- Compute features in <50ms
- Handle 10K requests/second
- Use most recent data
- Maintain consistency

**Approaches:**

### 9.1 Pre-Compute + Cache (Most Common)

**Strategy:** Compute features ahead of time, cache in Redis

**For Aggregates:**
```
Flink continuously computes:
  user_transaction_count_24h
  user_total_spend_7d
  
Updates Redis incrementally (per transaction)

At prediction time:
  Simply fetch from Redis (<5ms)
```

**Pros:**
- Ultra-low latency (just cache lookup)
- Handles complex computations offline
- Scales horizontally (more Redis nodes)

**Cons:**
- Features can be slightly stale (<1 minute)
- Requires pre-computation pipeline
- Memory cost for caching

**Best For:** Aggregates, historical features, complex computations

### 9.2 Compute-on-Request (On-Demand)

**Strategy:** Calculate features during prediction request

**Examples:**
```
Features computed live:
  - distance_from_home (requires current location)
  - time_since_last_transaction (requires current timestamp)
  - amount_deviation (requires current amount)
  - hour_of_day, day_of_week (temporal features)

Computation time: <5ms (simple math)
```

**Pros:**
- Always fresh (zero staleness)
- No storage needed
- Adapts to request context

**Cons:**
- Adds latency to prediction
- Limited to simple computations
- Compute cost per request

**Best For:** Simple derived features, context-dependent features

### 9.3 Hybrid Pre-Compute + Enrich

**Strategy:** Cache most features, compute a few on-demand

**Example Pipeline:**
```
Prediction Request arrives:
  ↓
1. Fetch pre-computed features from Redis (5ms)
   - user_profile
   - user_aggregates_24h
   - merchant_metadata
   ↓
2. Compute on-demand features (3ms)
   - distance_from_home (using cached home_location)
   - amount_vs_avg_ratio (using cached avg_amount)
   - is_weekend (from current timestamp)
   ↓
3. Combine into feature vector (1ms)
   ↓
4. Send to model (20ms)
   ↓
Total: 29ms ✓ (under 50ms budget)
```

**Best Practice:** Pre-compute expensive, compute simple on-demand

### Real-Time Computation Technologies

**Option 1: Flink**
- Stream processing framework
- Stateful computations (windowing)
- Exactly-once semantics
- Complex event processing

**Use Case:** Windowed aggregates, sessionization, pattern detection

**Option 2: Kafka Streams**
- Lighter than Flink
- Embedded in application
- Simpler ops (no separate cluster)

**Use Case:** Simple aggregations, filtering, enrichment

**Option 3: Python Microservice**
- For on-demand computations
- FastAPI/Flask endpoint
- Stateless (scales easily)

**Use Case:** Simple derivations, lookups, formatting

**Option 4: Feature Server (Feast)**
- Purpose-built for feature serving
- Handles caching automatically
- SDK for easy integration

**Use Case:** Unified feature retrieval (online + on-demand)

---

## 10. Feature Store Architecture Example

### Complete System for Fraud Detection

**Components:**

**1. Feature Registry (PostgreSQL + API)**
```
Tables:
  - features: metadata about each feature
  - feature_versions: version history
  - feature_lineage: dependencies
  - feature_usage: which models use which features

API endpoints:
  - POST /features (register new feature)
  - GET /features/{name} (get definition)
  - GET /features/{name}/lineage (trace dependencies)
```

**2. Offline Store (S3 + Parquet)**
```
Organization:
  s3://features/
  ├── user_profiles/v1/date=2024-12-12/
  ├── transaction_aggregates/v2/date=2024-12-12/
  └── merchant_features/v1/date=2024-12-12/

Partitioning: By date (for time travel)
Format: Parquet (compression + columnar)
Retention: 2 years
```

**3. Online Store (Redis Cluster)**
```
5-node cluster, 400GB total memory

Keys:
  user:{user_id}:profile:v1
  user:{user_id}:agg:24h:v2
  merchant:{merchant_id}:features:v1

Update strategy:
  - Nightly batch sync (profiles)
  - Real-time incremental (aggregates)

TTL:
  - Profiles: No expiration
  - Aggregates: Based on window (30h, 10d, 35d)
```

**4. Batch Pipeline (Spark + Airflow)**
```
DAG: daily_feature_batch

Tasks:
  1. extract_user_profiles (30 min)
  2. compute_merchant_features (20 min)
  3. validate_features (10 min)
  4. write_to_offline_store (15 min)
  5. sync_to_online_store (20 min)

Schedule: Daily 2 AM
Duration: ~2 hours
```

**5. Streaming Pipeline (Flink)**
```
Jobs:
  - realtime_aggregates (continuous)
    Input: Kafka (transactions)
    Output: Redis (incremental updates)
    Latency: <1 second
    
  - feature_validation (continuous)
    Input: Kafka (transactions)
    Output: Metrics (Prometheus)
    Monitors: Distribution changes, anomalies
```

**6. Feature Serving API (FastAPI)**
```
Endpoints:
  GET /features/online/{entity_id}
    - Fetches from Redis
    - Computes on-demand features
    - Returns feature vector
    - Latency: <10ms
    
  POST /features/batch
    - Fetches from S3
    - Point-in-time correct
    - Returns feature sets for training
    - Latency: seconds to minutes

Authentication: API keys
Rate limiting: 10K requests/sec
```

**7. Monitoring (Prometheus + Grafana)**
```
Metrics:
  - feature_freshness (last update timestamp)
  - feature_missing_rate (% nulls)
  - feature_distribution_stats (mean, std, percentiles)
  - offline_online_consistency (% match)
  - serving_latency_p95
  - pipeline_success_rate

Alerts:
  - Feature freshness >10 minutes
  - Missing rate >1%
  - Offline-online mismatch >5%
  - Serving latency >20ms
  - Pipeline failure
```

### Data Flow Example

**Training Flow:**
```
Data Scientist wants to train fraud model:

1. Define feature list:
   features = [
     "user_transaction_count_24h",
     "user_avg_transaction_amount",
     "merchant_risk_score"
   ]

2. Request training data:
   feature_store.get_historical_features(
     entities=user_ids,
     features=features,
     start_date="2024-06-01",
     end_date="2024-11-30"
   )

3. Feature store executes:
   - Query offline store (S3)
   - Point-in-time join (no data leakage!)
   - Return DataFrame with features

4. Data scientist trains model:
   model.fit(X, y)

5. Model registry records:
   - Features used: ["user_transaction_count_24h:v2", ...]
   - Training date: 2024-12-01
   - Feature store version: v1.5.0
```

**Inference Flow:**
```
Transaction arrives for fraud check:

1. Extract entity IDs:
   user_id = "user_123"
   merchant_id = "merch_456"

2. Request features:
   feature_vector = feature_store.get_online_features(
     entities={
       "user_id": "user_123",
       "merchant_id": "merch_456"
     },
     features=[
       "user_transaction_count_24h",
       "user_avg_transaction_amount",
       "merchant_risk_score"
     ],
     feature_version="v2"  # Match training version!
   )

3. Feature store executes:
   - Fetch from Redis (pre-computed features)
   - Compute on-demand (derived features)
   - Combine into vector
   - Return in <10ms

4. Model predicts:
   fraud_probability = model.predict(feature_vector)

5. Log prediction:
   - Feature values used
   - Model version
   - Timestamp
   - For audit trail
```

---

## 11. Interview Focus: Architecture Decisions

### Common Interview Questions

**Q1: "Why use a feature store instead of just querying the database?"**

**Good Answer:**
"Feature stores solve four critical problems:

**1. Train-Serve Consistency:** Without a feature store, training computes features in Spark/SQL while serving computes in Python/API. Different code = different logic = train-serve skew. Feature store ensures single source of truth.

**2. Latency:** Database queries take 50-500ms. Feature serving needs <10ms. Feature stores use Redis/DynamoDB for low-latency retrieval.

**3. Point-in-Time Correctness:** Training needs features 'as they were' on historical dates without data leakage. Databases don't track historical feature values. Feature stores maintain time-travel capability.

**4. Reusability:** Without centralized catalog, every team recomputes same features. Feature stores enable sharing and discovery.

I'd choose feature store for any production ML system at scale (>1M predictions/day) or with real-time requirements (<100ms latency)."

---

**Q2: "Design a feature store for real-time fraud detection at 10K TPS"**

**Good Answer:**
"I'd design a hybrid architecture:

**Online Store (Redis Cluster):**
- 5-node cluster, 400GB memory
- Sub-10ms latency
- Store: user profiles, aggregates, merchant data
- Update: streaming (Flink) + nightly batch (Spark)

**Offline Store (S3 + Parquet):**
- Point-in-time feature sets for training
- Date-partitioned for time travel
- 2-year retention

**Streaming Pipeline (Flink):**
- Compute real-time aggregates (24h, 7d, 30d counts/sums)
- Incremental updates to Redis
- <1 second latency

**Batch Pipeline (Spark):**
- Complex features (merchant risk scores, user profiles)
- Daily refresh at 2 AM
- Sync to Redis

**Feature Serving API:**
- FastAPI service
- Fetch from Redis + compute on-demand features
- <50ms total latency
- 10K requests/sec capacity

**Key Design Choices:**
1. Redis over DynamoDB (lower latency, simpler ops)
2. Streaming for aggregates (must be current)
3. Batch for complex features (too expensive real-time)
4. Hybrid update strategy (streaming + batch)
5. Monitoring at every layer"

---

**Q3: "How do you handle feature versioning?"**

**Good Answer:**
"Use semantic versioning with backward compatibility:

**Versioning Strategy:**
- Features are versioned: `user_avg_amount:v1`, `user_avg_amount:v2`
- Models specify exact versions: `fraud_model_v5` uses `user_avg_amount:v1.2.0`
- Breaking changes require new major version

**Storage:**
- Offline: Separate directories per version `s3://features/user_avg_amount/v1/`, `v2/`
- Online: Separate Redis keys `user:123:avg_amount:v1`, `user:123:avg_amount:v2`

**Lifecycle:**
- New version: Deploy alongside old version (both live)
- Migration: Models gradually upgraded to new version
- Deprecation: After 6 months, delete old version (if no models use it)

**Metadata Tracking:**
- Feature registry records all versions
- Models record exact feature versions used
- Can reproduce any historical prediction

**Example:**
```
user_avg_amount:
  v1: AVG(all transactions) - 2024-01 to 2024-06
  v2: AVG(last 90 days) - 2024-07 onwards
  
fraud_model_v5: uses v1 (trained May 2024)
fraud_model_v6: uses v2 (trained August 2024)

Both models in production until v5 deprecated.
```

This ensures reproducibility and safe feature evolution."

---

**Q4: "Your model performance dropped 10%. How do you debug features?"**

**Good Answer:**
"Systematic approach in this order:

**1. Check Feature Availability (5 min):**
- Are features being computed? Pipeline failures?
- Redis healthy? Any keys missing?
- Quick dashboard check

**2. Compare Offline vs Online (30 min):**
- Sample 10K recent predictions
- Fetch same features from offline store
- Calculate mismatch rate
- If >5% mismatch → train-serve skew issue

**3. Distribution Analysis (1 hour):**
- Compare current feature distributions vs training time
- PSI calculation for each feature
- Identify drifted features (PSI >0.25)
- Example: `user_avg_amount` shifted from $67 → $98 (Black Friday)

**4. Feature Lineage Investigation (1 hour):**
- Trace drifted features to upstream data
- Check for data quality issues
- Recent pipeline changes?
- Schema changes in source data?

**5. Correlation Analysis:**
- Which features changed when performance dropped?
- Correlation between feature drift and prediction errors
- Identify culprit features

**Common Root Causes:**
1. Data drift (most common - 60%)
2. Pipeline bug (20%)
3. Train-serve skew (15%)
4. Upstream data quality (5%)

**Tools:**
- Feature store monitoring dashboards
- Great Expectations validation reports
- Feature lineage graphs
- A/B test framework (rollback to old features)

Real example: E-commerce fraud model degraded because merchant categories changed (data provider updated taxonomy). Feature store lineage traced issue to upstream data vendor."

---

**Q5: "When would you NOT use a feature store?"**

**Good Answer:**
"Feature stores add complexity. Skip them when:

**1. Prototype/POC:**
- Small dataset (<1M rows)
- No production deployment planned
- Just validating ML approach
- Use pandas/SQL directly

**2. Batch-Only Models:**
- Daily/weekly predictions (no real-time)
- Single model, single team
- Simple features (no complex pipelines)
- Airflow + S3 sufficient

**3. Very Simple Use Case:**
- <10 features
- No aggregates or windows
- Features from single database table
- Direct database query acceptable

**4. Resource Constrained:**
- Small team (<3 engineers)
- Limited budget
- Can't maintain infrastructure
- Simpler to use managed services (Vertex AI, SageMaker)

**5. Short-Lived Project:**
- Temporary analysis
- No need for reproducibility
- Not mission-critical

**When Feature Store IS Worth It:**
- Real-time serving (<100ms)
- Multiple models sharing features
- Train-serve consistency critical
- Large scale (>1M predictions/day)
- Regulated industry (need audit trail)
- Team >5 people

In my experience, threshold is around 5 models in production or 1M+ daily predictions. Below that, simpler approaches often sufficient."

---

## Summary: Key Takeaways

**1. Feature Store = ML Data Infrastructure**
- Centralized feature management
- Train-serve consistency
- Low-latency serving
- Versioning and lineage

**2. Two Storage Layers Always:**
- Offline (S3/warehouse): Training, analytics
- Online (Redis/DynamoDB): Real-time serving

**3. Three Computation Patterns:**
- Batch (Spark): Complex, expensive features
- Streaming (Flink): Real-time aggregates
- On-demand (API): Simple derivations

**4. Monitoring is Critical:**
- Feature availability
- Distribution drift
- Offline-online consistency
- Pipeline health

**5. Start Simple, Scale Up:**
- Begin with Feast (open source)
- Migrate to Tecton if scaling challenges
- Build custom only if very unique needs

**6. Architecture Principles:**
- Separate hot path (streaming) from cold path (batch)
- Pre-compute expensive, compute simple on-demand
- Version everything (features, models, pipelines)
- Monitor aggressively (catch issues early)
- Design for reproducibility (audit trail)

**For Interviews:**
- Understand trade-offs (latency vs accuracy, cost vs performance)
- Know when feature store is overkill vs necessary
- Explain with concrete examples (fraud detection, recommendations)
- Show systems thinking (not just ML, but data engineering)

---

**END OF DAY 8**
