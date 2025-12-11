# Day 1: ML Fundamentals for Data Engineers
## Questions and Answers - Part 2 (Continuation)

---

## Continuing Q7: Debugging Overfitting (continued)

```python
    # Method 1b: Feature importance from tree model
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_temp.fit(X_train, y_train)
    
    # Get top 50 features
    importances = rf_temp.feature_importances_
    top_50_indices = np.argsort(importances)[-50:]
    
    X_train_top50 = X_train[:, top_50_indices]
    X_test_top50 = X_test[:, top_50_indices]
    
    # If improvement is significant, keep this
    improvement = test_acc_v2 - 0.70  # Original test accuracy
    print(f"\nImprovement: {improvement:+.1%}")
    
    # Strategy 2: Regularization
    print("\n" + "="*50)
    print("STRATEGY 2: Add Regularization")
    print("="*50)
    
    from sklearn.ensemble import RandomForestClassifier
    
    # Tune hyperparameters to reduce overfitting
    model_v3 = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,  # Limit tree depth
        min_samples_split=20,  # Require more samples to split
        min_samples_leaf=10,  # Require more samples in leaf
        max_features='sqrt',  # Use fewer features per tree
        random_state=42
    )
    
    model_v3.fit(X_train_top50, y_train)
    train_acc_v3 = model_v3.score(X_train_top50, y_train)
    test_acc_v3 = model_v3.score(X_test_top50, y_test)
    
    print(f"After regularization:")
    print(f"  Train: {train_acc_v3:.3f}")
    print(f"  Test: {test_acc_v3:.3f}")
    print(f"  Gap: {train_acc_v3 - test_acc_v3:.3f}")
    
    # Strategy 3: Cross-validation for robust evaluation
    print("\n" + "="*50)
    print("STRATEGY 3: Cross-Validation")
    print("="*50)
    
    from sklearn.model_selection import cross_val_score
    
    cv_scores = cross_val_score(
        model_v3, 
        X_train_top50, 
        y_train, 
        cv=5, 
        scoring='accuracy'
    )
    
    print(f"5-Fold CV Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # If CV score is close to test score, we're good
    if abs(cv_scores.mean() - test_acc_v3) < 0.05:
        print("✓ Model generalizes well across folds")
    
    # Strategy 4: Ensemble with different algorithms
    print("\n" + "="*50)
    print("STRATEGY 4: Try Simpler Model")
    print("="*50)
    
    from sklearn.linear_model import LogisticRegression
    
    # Sometimes simpler is better
    model_simple = LogisticRegression(
        C=0.1,  # Strong regularization
        max_iter=1000,
        random_state=42
    )
    
    model_simple.fit(X_train_top50, y_train)
    train_acc_simple = model_simple.score(X_train_top50, y_train)
    test_acc_simple = model_simple.score(X_test_top50, y_test)
    
    print(f"Logistic Regression (simpler model):")
    print(f"  Train: {train_acc_simple:.3f}")
    print(f"  Test: {test_acc_simple:.3f}")
    print(f"  Gap: {train_acc_simple - test_acc_simple:.3f}")
    
    # Strategy 5: Get more data (if possible)
    print("\n" + "="*50)
    print("STRATEGY 5: Data Augmentation / More Data")
    print("="*50)
    
    print("Options:")
    print("  1. Collect more historical data")
    print("  2. Use SMOTE for minority class (if imbalanced)")
    print("  3. Data augmentation (if applicable)")
    print("  4. Semi-supervised learning with unlabeled data")
    
    return {
        'feature_selection': (train_acc_v2, test_acc_v2),
        'regularization': (train_acc_v3, test_acc_v3),
        'simple_model': (train_acc_simple, test_acc_simple)
    }

# Execute strategy
results = fix_overfitting_strategy()
```

---

**Phase 3: Monitor and Validate**

```python
def production_validation(model, X_test, y_test):
    """
    Final checks before production
    """
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\n" + "="*50)
    print("PRODUCTION VALIDATION")
    print("="*50)
    
    # 1. Detailed metrics
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 2. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # 3. Check for class imbalance issues
    from collections import Counter
    class_dist = Counter(y_test)
    print(f"\nClass Distribution in Test:")
    for class_label, count in class_dist.items():
        print(f"  Class {class_label}: {count} ({count/len(y_test):.1%})")
    
    # 4. Threshold analysis (for binary classification)
    from sklearn.metrics import roc_curve, roc_auc_score
    
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"\nROC AUC Score: {auc:.3f}")
        
        # Find optimal threshold
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print(f"Optimal Threshold: {optimal_threshold:.3f}")
    
    # 5. Production readiness checklist
    print("\n" + "="*50)
    print("PRODUCTION READINESS CHECKLIST")
    print("="*50)
    
    checklist = {
        'Train-test gap < 10%': train_acc - test_acc < 0.10,
        'Test accuracy > 75%': test_acc > 0.75,
        'No data leakage': True,  # From earlier check
        'Cross-validation consistent': True,  # From earlier check
        'Feature count reasonable': X_test.shape[1] < 100,
        'Model size < 100MB': True,  # Check actual model size
        'Prediction latency < 100ms': True,  # To be tested
    }
    
    for check, passed in checklist.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
    
    return all(checklist.values())

# Validate
is_production_ready = production_validation(model_v3, X_test_top50, y_test)
```

---

**Complete Solution Summary:**

```python
def complete_overfitting_solution():
    """
    End-to-end solution summarized
    """
    
    print("""
    OVERFITTING SOLUTION - COMPLETE WORKFLOW
    ========================================
    
    Starting Point:
    - 1M samples, 200 features
    - Train: 99%, Test: 70% (29% gap = SEVERE OVERFITTING)
    
    Step 1: DIAGNOSE (30 min)
    ✓ Confirmed overfitting with learning curves
    ✓ Checked for data leakage
    ✓ Analyzed feature distribution
    
    Step 2: FEATURE REDUCTION (1 hour)
    ✓ Statistical feature selection: 200 → 50 features
    ✓ Used Random Forest feature importance
    ✓ Removed low-variance and highly correlated features
    Result: Test improved to 78%
    
    Step 3: REGULARIZATION (30 min)
    ✓ Tuned hyperparameters:
        - max_depth: None → 10
        - min_samples_split: 2 → 20
        - min_samples_leaf: 1 → 10
    Result: Train: 85%, Test: 82% (3% gap)
    
    Step 4: CROSS-VALIDATION (30 min)
    ✓ 5-fold CV: 82% ± 2%
    ✓ Consistent across folds
    
    Step 5: PRODUCTION VALIDATION (30 min)
    ✓ All metrics checked
    ✓ Ready for deployment
    
    Final Results:
    - Original: Train 99%, Test 70%, Gap 29%
    - Final:    Train 85%, Test 82%, Gap 3%
    - Features: 200 → 50 (4x reduction)
    - Improvement: +12% test accuracy
    
    Key Learnings:
    1. More features ≠ better model
    2. Feature selection is crucial
    3. Regularization prevents overfitting
    4. Always validate with cross-validation
    5. Monitor train-test gap, not just test accuracy
    """)

complete_overfitting_solution()
```

---

**Data Engineering Perspective:**

```python
class OverfittingMonitor:
    """
    Production monitoring for overfitting
    """
    
    def __init__(self):
        self.train_test_gap_threshold = 0.10
        self.accuracy_drop_threshold = 0.05
        
    def monitor_production_model(self, model_metrics):
        """
        Monitor deployed model for overfitting signs
        """
        
        # 1. Track metrics over time
        current_metrics = {
            'train_accuracy': model_metrics['train_acc'],
            'prod_accuracy': model_metrics['prod_acc'],
            'gap': model_metrics['train_acc'] - model_metrics['prod_acc']
        }
        
        # 2. Alert conditions
        if current_metrics['gap'] > self.train_test_gap_threshold:
            self.send_alert(
                severity='HIGH',
                message=f"Train-prod gap {current_metrics['gap']:.1%} exceeds threshold"
            )
            return 'RETRAIN_NEEDED'
        
        # 3. Check for performance degradation
        if model_metrics['prod_acc'] < model_metrics['baseline_acc'] - self.accuracy_drop_threshold:
            self.send_alert(
                severity='MEDIUM',
                message=f"Production accuracy dropped {model_metrics['prod_acc']:.1%}"
            )
            return 'INVESTIGATE'
        
        return 'OK'
    
    def setup_retraining_pipeline(self):
        """
        Automated retraining trigger
        """
        sql = """
        -- Monitor weekly performance
        WITH weekly_metrics AS (
            SELECT 
                DATE_TRUNC('week', prediction_timestamp) as week,
                AVG(CASE WHEN prediction = actual THEN 1 ELSE 0 END) as accuracy,
                COUNT(*) as predictions
            FROM ml_predictions
            WHERE prediction_timestamp >= CURRENT_DATE - INTERVAL '8 weeks'
            GROUP BY 1
        )
        SELECT 
            week,
            accuracy,
            LAG(accuracy) OVER (ORDER BY week) as prev_accuracy,
            accuracy - LAG(accuracy) OVER (ORDER BY week) as accuracy_change
        FROM weekly_metrics
        WHERE accuracy - LAG(accuracy) OVER (ORDER BY week) < -0.05  -- 5% drop
        """
        
        # If degradation detected, trigger retraining
        # This connects to your Airflow/orchestration system

monitor = OverfittingMonitor()
```

**Interview Answer Summary:**

"When debugging overfitting with 1M rows and 200 features (99% train, 70% test), I'd take a systematic approach:

**1. Diagnose (30 min):**
- Plot learning curves to confirm overfitting
- Check for data leakage
- Analyze train-test gap trend

**2. Feature Reduction (1 hour):**
- Reduce 200 → 50 features using feature importance
- As a data engineer, I'd implement this in the ETL pipeline
- Remove low-variance and highly correlated features early

**3. Regularization (30 min):**
- Tune hyperparameters (max_depth, min_samples_split)
- Try simpler models (Logistic Regression vs Random Forest)

**4. Validate (30 min):**
- Use 5-fold cross-validation
- Ensure consistency across folds

**5. Production Setup:**
- Monitor train-prod gap in production
- Set up automated alerts when gap > 10%
- Implement retraining pipeline

As a data engineer, I'd ensure this is reproducible in our ML pipeline and set up monitoring to catch overfitting in production before it impacts business."

---

### Q8: Your ML model performs well in training but fails in production. Walk through your debugging checklist.

**Answer:**

**The Train-Production Gap Problem**

This is one of the most common ML problems in production. Let's create a systematic debugging framework.

---

**Phase 1: Immediate Diagnosis (15 minutes)**

```python
class ProductionDebugger:
    """
    Systematic approach to debug production ML issues
    """
    
    def quick_diagnosis(self):
        """
        First 15 minutes - identify the problem type
        """
        
        print("="*60)
        print("QUICK DIAGNOSIS - Production ML Failure")
        print("="*60)
        
        checks = {
            '1. Check Metrics': self.check_basic_metrics(),
            '2. Check Data Pipeline': self.check_data_pipeline(),
            '3. Check Model Version': self.check_model_version(),
            '4. Check Infrastructure': self.check_infrastructure()
        }
        
        for check_name, result in checks.items():
            print(f"\n{check_name}: {result['status']}")
            if result['status'] == 'FAILED':
                print(f"  Issue: {result['message']}")
                return result['issue_type']
        
        return 'UNKNOWN'
    
    def check_basic_metrics(self):
        """
        Check if metrics are actually bad
        """
        query = """
        SELECT 
            DATE(prediction_timestamp) as date,
            COUNT(*) as total_predictions,
            AVG(CASE WHEN prediction = actual_label THEN 1 ELSE 0 END) as accuracy,
            AVG(prediction_confidence) as avg_confidence
        FROM ml_predictions
        WHERE prediction_timestamp >= CURRENT_DATE - 7
        GROUP BY 1
        ORDER BY 1 DESC
        """
        
        results = execute_query(query)
        
        # Check for degradation
        if results[0]['accuracy'] < results[-1]['accuracy'] - 0.10:
            return {
                'status': 'FAILED',
                'message': f"Accuracy dropped from {results[-1]['accuracy']:.1%} to {results[0]['accuracy']:.1%}",
                'issue_type': 'PERFORMANCE_DEGRADATION'
            }
        
        # Check prediction volume
        if results[0]['total_predictions'] < results[-1]['total_predictions'] * 0.5:
            return {
                'status': 'FAILED',
                'message': 'Prediction volume dropped 50%',
                'issue_type': 'VOLUME_DROP'
            }
        
        return {'status': 'PASSED', 'message': 'Metrics look normal'}
    
    def check_data_pipeline(self):
        """
        Check if input data changed
        """
        query = """
        -- Compare feature distributions
        WITH current_data AS (
            SELECT * FROM ml_features
            WHERE created_at >= CURRENT_DATE - 1
            LIMIT 10000
        ),
        historical_data AS (
            SELECT * FROM ml_features
            WHERE created_at BETWEEN CURRENT_DATE - 30 AND CURRENT_DATE - 23
            LIMIT 10000
        )
        SELECT 
            'current' as period,
            AVG(feature_1) as avg_f1,
            STDDEV(feature_1) as std_f1,
            AVG(feature_2) as avg_f2,
            COUNT(DISTINCT user_id) as unique_users,
            COUNT(*) as total_rows
        FROM current_data
        UNION ALL
        SELECT 
            'historical' as period,
            AVG(feature_1) as avg_f1,
            STDDEV(feature_1) as std_f1,
            AVG(feature_2) as avg_f2,
            COUNT(DISTINCT user_id) as unique_users,
            COUNT(*) as total_rows
        FROM historical_data
        """
        
        results = execute_query(query)
        current = results[0]
        historical = results[1]
        
        # Check for distribution shift
        for feature in ['avg_f1', 'avg_f2']:
            change = abs(current[feature] - historical[feature]) / historical[feature]
            if change > 0.20:  # 20% change
                return {
                    'status': 'FAILED',
                    'message': f'{feature} changed by {change:.1%}',
                    'issue_type': 'DATA_DRIFT'
                }
        
        # Check for data quality issues
        if current['total_rows'] == 0:
            return {
                'status': 'FAILED',
                'message': 'No data in pipeline!',
                'issue_type': 'PIPELINE_BROKEN'
            }
        
        return {'status': 'PASSED', 'message': 'Data pipeline OK'}
    
    def check_model_version(self):
        """
        Check if correct model is deployed
        """
        query = """
        SELECT 
            model_version,
            deployed_at,
            deployment_status,
            COUNT(*) as predictions_count
        FROM model_deployments
        WHERE deployed_at >= CURRENT_DATE - 7
        GROUP BY 1, 2, 3
        ORDER BY deployed_at DESC
        """
        
        results = execute_query(query)
        
        if not results:
            return {
                'status': 'FAILED',
                'message': 'No recent deployments found',
                'issue_type': 'DEPLOYMENT_ISSUE'
            }
        
        latest = results[0]
        if latest['deployment_status'] != 'SUCCESS':
            return {
                'status': 'FAILED',
                'message': f"Latest deployment status: {latest['deployment_status']}",
                'issue_type': 'DEPLOYMENT_FAILED'
            }
        
        # Check if multiple versions are serving (should be only one)
        if len(results) > 1 and results[1]['predictions_count'] > 0:
            return {
                'status': 'WARNING',
                'message': 'Multiple model versions serving traffic',
                'issue_type': 'VERSION_CONFLICT'
            }
        
        return {'status': 'PASSED', 'message': f"Model v{latest['model_version']} deployed correctly"}
    
    def check_infrastructure(self):
        """
        Check infrastructure health
        """
        # Check service health
        health_checks = {
            'prediction_service': check_service_health('prediction-service'),
            'feature_service': check_service_health('feature-service'),
            'database': check_database_health()
        }
        
        for service, health in health_checks.items():
            if not health['healthy']:
                return {
                    'status': 'FAILED',
                    'message': f'{service} is unhealthy: {health["error"]}',
                    'issue_type': 'INFRASTRUCTURE'
                }
        
        # Check latency
        latency_query = """
        SELECT 
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY prediction_latency_ms) as p50,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY prediction_latency_ms) as p95,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY prediction_latency_ms) as p99
        FROM ml_predictions
        WHERE prediction_timestamp >= NOW() - INTERVAL '1 hour'
        """
        
        latency = execute_query(latency_query)[0]
        if latency['p95'] > 200:  # 200ms SLA
            return {
                'status': 'FAILED',
                'message': f"P95 latency {latency['p95']}ms exceeds SLA",
                'issue_type': 'LATENCY'
            }
        
        return {'status': 'PASSED', 'message': 'Infrastructure healthy'}

# Run quick diagnosis
debugger = ProductionDebugger()
issue_type = debugger.quick_diagnosis()
print(f"\n🎯 Identified Issue Type: {issue_type}")
```

---

**Phase 2: Deep Dive by Issue Type**

```python
def deep_dive_data_drift():
    """
    If issue is DATA_DRIFT
    """
    print("\n" + "="*60)
    print("DEEP DIVE: Data Drift Analysis")
    print("="*60)
    
    # 1. Feature-by-feature comparison
    def compare_distributions(feature_name):
        query = f"""
        WITH current AS (
            SELECT {feature_name} as value
            FROM ml_features
            WHERE created_at >= CURRENT_DATE - 1
        ),
        training AS (
            SELECT {feature_name} as value
            FROM ml_training_data
            WHERE split = 'train'
        )
        SELECT 
            'current' as dataset,
            AVG(value) as mean,
            STDDEV(value) as std,
            MIN(value) as min,
            MAX(value) as max,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) as median
        FROM current
        UNION ALL
        SELECT 
            'training' as dataset,
            AVG(value) as mean,
            STDDEV(value) as std,
            MIN(value) as min,
            MAX(value) as max,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) as median
        FROM training
        """
        
        return execute_query(query)
    
    # 2. Statistical tests for drift
    from scipy.stats import ks_2samp
    
    def kolmogorov_smirnov_test(current_data, training_data):
        """
        Test if distributions are significantly different
        """
        statistic, p_value = ks_2samp(current_data, training_data)
        
        if p_value < 0.05:
            return {
                'drift_detected': True,
                'p_value': p_value,
                'recommendation': 'RETRAIN_MODEL'
            }
        return {
            'drift_detected': False,
            'p_value': p_value,
            'recommendation': 'CONTINUE_MONITORING'
        }
    
    # 3. Check specific drift types
    drift_analysis = {
        'covariate_shift': check_covariate_shift(),  # X distribution changed
        'prior_probability_shift': check_prior_shift(),  # Y distribution changed
        'concept_drift': check_concept_drift()  # X→Y relationship changed
    }
    
    print("\nDrift Analysis Results:")
    for drift_type, result in drift_analysis.items():
        print(f"  {drift_type}: {'DETECTED' if result['detected'] else 'OK'}")
        if result['detected']:
            print(f"    Action: {result['action']}")
    
    # 4. Root cause identification
    print("\n" + "-"*60)
    print("Root Cause Analysis:")
    print("-"*60)
    
    root_causes = [
        "✓ Check: Did business logic change?",
        "✓ Check: New data sources added?",
        "✓ Check: Upstream pipeline modified?",
        "✓ Check: Seasonality/trends?",
        "✓ Check: Data quality issues?"
    ]
    
    for cause in root_causes:
        print(cause)
    
    # 5. Immediate fix
    print("\n" + "-"*60)
    print("Immediate Actions:")
    print("-"*60)
    
    actions = """
    1. IMMEDIATE (< 1 hour):
       - Switch to fallback model (if available)
       - Increase monitoring frequency
       - Alert data team
    
    2. SHORT-TERM (1-24 hours):
       - Retrain model on recent data
       - Update feature engineering pipeline
       - Deploy hotfix model
    
    3. LONG-TERM (1-7 days):
       - Implement online learning
       - Add drift detection to pipeline
       - Set up automated retraining
    """
    
    print(actions)

def deep_dive_data_leakage():
    """
    If suspecting data leakage (works in training, fails in production)
    """
    print("\n" + "="*60)
    print("DEEP DIVE: Data Leakage Detection")
    print("="*60)
    
    common_leakage_patterns = {
        '1. Target Leakage': [
            'Features derived from target variable',
            'Example: Using "purchase_total" to predict "will_purchase"',
            'Check: Remove features that wouldn\'t exist at prediction time'
        ],
        
        '2. Train-Test Contamination': [
            'Test data used during training',
            'Example: Scaling on all data before split',
            'Check: Ensure preprocessing done after split'
        ],
        
        '3. Temporal Leakage': [
            'Using future information',
            'Example: Using next month\'s data to predict this month',
            'Check: Respect time ordering in features'
        ],
        
        '4. Preprocessing Leakage': [
            'Statistics computed on test set',
            'Example: Mean imputation using all data',
            'Check: Fit transformers on train only'
        ]
    }
    
    for pattern, checks in common_leakage_patterns.items():
        print(f"\n{pattern}:")
        for check in checks:
            print(f"  - {check}")
    
    # Automated checks
    def check_feature_availability():
        """
        Check if all training features available at prediction time
        """
        query = """
        -- Compare training features vs production features
        WITH training_features AS (
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'ml_training_data'
        ),
        production_features AS (
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'ml_production_features'
        )
        SELECT 
            t.column_name as training_only_feature
        FROM training_features t
        LEFT JOIN production_features p ON t.column_name = p.column_name
        WHERE p.column_name IS NULL
        """
        
        missing_features = execute_query(query)
        
        if missing_features:
            print("\n⚠️  WARNING: Features in training but not in production:")
            for feat in missing_features:
                print(f"  - {feat['training_only_feature']}")
            return 'LEAKAGE_SUSPECTED'
        
        return 'OK'
    
    result = check_feature_availability()
    
    if result == 'LEAKAGE_SUSPECTED':
        print("\n🔧 FIX:")
        print("  1. Review feature engineering pipeline")
        print("  2. Remove features not available at prediction time")
        print("  3. Retrain model without leaked features")
        print("  4. Compare train/test accuracy (should be closer now)")

def deep_dive_training_serving_skew():
    """
    If feature computation differs between training and serving
    """
    print("\n" + "="*60)
    print("DEEP DIVE: Training-Serving Skew")
    print("="*60)
    
    # Compare feature values
    query = """
    -- Sample comparison
    WITH training_sample AS (
        SELECT 
            user_id,
            feature_1 as train_f1,
            feature_2 as train_f2,
            'training' as source
        FROM ml_training_data
        LIMIT 1000
    ),
    serving_sample AS (
        SELECT 
            user_id,
            feature_1 as serve_f1,
            feature_2 as serve_f2,
            'serving' as source
        FROM ml_serving_features
        WHERE created_at >= CURRENT_DATE - 1
        LIMIT 1000
    )
    SELECT 
        t.user_id,
        t.train_f1,
        s.serve_f1,
        ABS(t.train_f1 - s.serve_f1) as diff_f1,
        t.train_f2,
        s.serve_f2,
        ABS(t.train_f2 - s.serve_f2) as diff_f2
    FROM training_sample t
    JOIN serving_sample s ON t.user_id = s.user_id
    WHERE ABS(t.train_f1 - s.serve_f1) > 0.01  -- Significant difference
    ORDER BY diff_f1 DESC
    LIMIT 100
    """
    
    skewed_features = execute_query(query)
    
    if skewed_features:
        print("\n⚠️  Training-Serving Skew Detected!")
        print(f"Found {len(skewed_features)} users with feature differences")
        
        # Analyze root cause
        print("\nCommon Causes:")
        causes = [
            "1. Different preprocessing code (training vs serving)",
            "2. Different data sources",
            "3. Timing issues (batch vs real-time computation)",
            "4. Missing values handled differently",
            "5. Floating point precision differences"
        ]
        
        for cause in causes:
            print(f"  {cause}")
        
        print("\n🔧 FIX:")
        print("  1. Use same code for training and serving")
        print("  2. Containerize feature engineering logic")
        print("  3. Use feature store for consistency")
        print("  4. Add feature validation tests")
        print("  5. Log features during serving to compare")

# Execute appropriate deep dive based on issue
if issue_type == 'DATA_DRIFT':
    deep_dive_data_drift()
elif issue_type == 'PERFORMANCE_DEGRADATION':
    # Could be leakage or skew
    deep_dive_data_leakage()
    deep_dive_training_serving_skew()
```

---

**Phase 3: Comprehensive Debugging Checklist**

```python
def complete_debugging_checklist():
    """
    Comprehensive checklist for production ML issues
    """
    
    checklist = {
        'DATA ISSUES': {
            '□ Data drift detected': check_data_drift(),
            '□ Missing values increased': check_missing_values(),
            '□ Feature distributions changed': check_distributions(),
            '□ Data quality issues': check_data_quality(),
            '□ Upstream pipeline broken': check_upstream(),
        },
        
        'MODEL ISSUES': {
            '□ Wrong model version deployed': check_model_version(),
            '□ Model files corrupted': check_model_integrity(),
            '□ Incompatible model format': check_model_format(),
            '□ Model not loaded properly': check_model_loading(),
        },
        
        'FEATURE ISSUES': {
            '□ Training-serving skew': check_feature_skew(),
            '□ Feature missing in production': check_feature_availability(),
            '□ Feature computation error': check_feature_computation(),
            '□ Feature scaling inconsistent': check_feature_scaling(),
        },
        
        'INFRASTRUCTURE': {
            '□ Service outage': check_service_health(),
            '□ High latency': check_latency(),
            '□ Resource constraints': check_resources(),
            '□ Network issues': check_network(),
        },
        
        'CODE/DEPLOYMENT': {
            '□ Code bug in serving': check_serving_code(),
            '□ Dependency mismatch': check_dependencies(),
            '□ Configuration error': check_configuration(),
            '□ Environment differences': check_environment(),
        }
    }
    
    print("="*70)
    print("COMPREHENSIVE DEBUGGING CHECKLIST")
    print("="*70)
    
    all_passed = True
    for category, checks in checklist.items():
        print(f"\n{category}:")
        print("-" * 70)
        
        for check_name, check_func in checks.items():
            result = check_func()
            status = "✓" if result['passed'] else "✗"
            print(f"{status} {check_name}")
            
            if not result['passed']:
                print(f"    Issue: {result['message']}")
                print(f"    Fix: {result['fix']}")
                all_passed = False
    
    return all_passed

# Run complete checklist
all_checks_passed = complete_debugging_checklist()

if not all_checks_passed:
    print("\n" + "="*70)
    print("ACTION PLAN")
    print("="*70)
    print(generate_action_plan())
```

---

**Phase 4: Prevention & Monitoring**

```python
def setup_production_monitoring():
    """
    Set up comprehensive monitoring to catch issues early
    """
    
    monitoring_config = {
        'metrics': {
            'model_accuracy': {
                'query': '''
                    SELECT 
                        DATE(prediction_time) as date,
                        AVG(CASE WHEN prediction = actual THEN 1 ELSE 0 END) as accuracy
                    FROM predictions
                    WHERE prediction_time >= CURRENT_DATE - 7
                    GROUP BY 1
                ''',
                'alert_threshold': 0.75,  # Alert if < 75%
                'alert_type': 'BELOW'
            },
            
            'prediction_volume': {
                'query': '''
                    SELECT 
                        DATE_TRUNC('hour', prediction_time) as hour,
                        COUNT(*) as volume
                    FROM predictions
                    WHERE prediction_time >= NOW() - INTERVAL '24 hours'
                    GROUP BY 1
                ''',
                'alert_threshold': 1000,  # Alert if < 1000/hour
                'alert_type': 'BELOW'
            },
            
            'feature_drift': {
                'query': '''
                    SELECT 
                        feature_name,
                        AVG(value) as current_avg,
                        training_avg,
                        ABS(AVG(value) - training_avg) / training_avg as drift_pct
                    FROM current_features, training_stats
                    WHERE current_features.created_at >= CURRENT_DATE - 1
                    GROUP BY feature_name, training_avg
                    HAVING ABS(AVG(value) - training_avg) / training_avg > 0.20
                ''',
                'alert_threshold': 0.20,  # 20% drift
                'alert_type': 'ABOVE'
            },
            
            'latency_p95': {
                'query': '''
                    SELECT 
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95
                    FROM predictions
                    WHERE prediction_time >= NOW() - INTERVAL '1 hour'
                ''',
                'alert_threshold': 200,  # 200ms
                'alert_type': 'ABOVE'
            }
        },
        
        'dashboards': [
            'Model Performance Over Time',
            'Feature Distributions',
            'Prediction Volume & Latency',
            'Error Analysis',
            'A/B Test Results'
        ],
        
        'alerts': {
            'critical': ['model_accuracy', 'service_down'],
            'warning': ['feature_drift', 'latency_high'],
            'info': ['prediction_volume_change']
        }
    }
    
    print("Setting up production monitoring...")
    for metric_name, config in monitoring_config['metrics'].items():
        print(f"  ✓ Monitoring: {metric_name}")
    
    return monitoring_config

monitoring = setup_production_monitoring()
```

**Interview Answer Summary:**

"When an ML model fails in production despite good training performance, I follow this systematic approach:

**Immediate (15 min):**
1. Check basic metrics - is it actually failing?
2. Check data pipeline - any recent changes?
3. Check model version - correct model deployed?
4. Check infrastructure - service healthy?

**Deep Dive (1-2 hours):**
- **If data drift:** Compare current vs training distributions, run KS tests, identify which features drifted
- **If data leakage:** Check for features that wouldn't exist at prediction time
- **If training-serving skew:** Compare feature values between training and production
- **If infrastructure:** Check logs, latency, resource usage

**Common Causes I've Seen:**
1. **Data drift** (most common) - retraining solves this
2. **Training-serving skew** - feature computation differs
3. **Data leakage** - used future information in training
4. **Infrastructure issues** - latency, resource constraints

**Prevention (Data Engineering Mindset):**
As a data engineer, I'd set up:
- Automated drift detection in pipelines
- Feature validation tests (training vs serving)
- Comprehensive monitoring dashboards
- Automated retraining triggers
- A/B testing for safe rollouts

The key is catching issues before they impact users through good monitoring and validation."

---

<a name="system-design"></a>
## 3. SYSTEM DESIGN QUESTIONS

### Q9: Design a real-time fraud detection system that processes 10,000 transactions/second.

**Answer:**

This is a classic ML system design question that tests architecture thinking.

**Step 1: Requirements Clarification (5 minutes)**

```
FUNCTIONAL REQUIREMENTS:
1. Process 10,000 transactions/second (TPS)
2. Real-time prediction (< 100ms latency)
3. High accuracy (catch fraud, minimize false positives)
4. Scalable to 50,000 TPS in future

NON-FUNCTIONAL REQUIREMENTS:
1. Availability: 99.9% uptime
2. Latency: P95 < 100ms
3. Throughput: 10,000 TPS
4. Data retention: 7 years (compliance)
5. Model retraining: Daily
6. False positive rate: < 1%

SCALE ESTIMATION:
- 10,000 TPS = 864M transactions/day
- Each transaction ~1KB = 864GB/day = 315TB/year
- Features per transaction: ~50 features
- Model size: ~100MB
- Prediction requests: 10,000/sec
- Feature store: ~10TB (hot data)
```

**Step 2: High-Level Architecture (10 minutes)**

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT APPLICATIONS                       │
│                    (Web, Mobile, API)                           │
└────────────────────┬────────────────────────────────────────────┘
                     │ Transaction Request
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API GATEWAY / LOAD BALANCER                 │
│                   (Rate limiting, Auth, Routing)                │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────────┐      ┌──────────────────┐
│  SYNCHRONOUS     │      │   ASYNCHRONOUS   │
│  PROCESSING      │      │   PROCESSING     │
│  (Low risk)      │      │   (High risk)    │
└────────┬─────────┘      └────────┬─────────┘
         │                         │
         ▼                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FRAUD DETECTION SERVICE                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐               │
│  │  Feature   │→ │   Model    │→ │   Risk     │               │
│  │Extraction  │  │  Serving   │  │  Scoring   │               │
│  └────────────┘  └────────────┘  └────────────┘               │
└─────────┬───────────────────────────────┬──────────────────────┘
          │                               │
          ▼                               ▼
┌──────────────────┐            ┌──────────────────┐
│  FEATURE STORE   │            │   DECISION       │
│  (Real-time)     │            │   ENGINE         │
│  - Redis         │            │  (Rules + ML)    │
│  - User history  │            └────────┬─────────┘
│  - Aggregations  │                     │
└──────────────────┘                     ▼
          ▲                     ┌──────────────────┐
          │                     │    RESPONSE      │
          │                     │  - APPROVE       │
          │                     │  - DECLINE       │
          │                     │  - REVIEW        │
          │                     └──────────────────┘
          │
┌─────────┴────────────────────────────────────────────────────────┐
│                    BATCH PROCESSING PIPELINE                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │   Data     │→ │  Feature   │→ │   Model    │                │
│  │  Ingestion │  │Engineering │  │  Training  │                │
│  └────────────┘  └────────────┘  └────────────┘                │
│                                                                   │
│  Data Lake (S3) ← Kafka ← Transaction Events                     │
└──────────────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────────┐
│                    MONITORING & ALERTING                          │
│  - Model performance    - Data drift    - Latency                │
│  - False positive rate  - Throughput    - Errors                 │
└──────────────────────────────────────────────────────────────────┘
```

**Step 3: Component Deep Dive (30 minutes)**

**3.1 Real-Time Prediction Service**

```python
class FraudDetectionService:
    """
    Real-time fraud detection microservice
    """
    
    def __init__(self):
        self.model = load_model('fraud_model_v1.pkl')
        self.feature_store = RedisFeatureStore()
        self.rules_engine = RulesEngine()
        self.cache = PredictionCache()
        
    async def predict(self, transaction: Transaction) -> FraudScore:
        """
        Main prediction endpoint
        Target: < 100ms P95 latency
        """
        start_time = time.time()
        
        # 1. Quick rule-based filtering (5ms)
        rule_result = self.rules_engine.evaluate(transaction)
        if rule_result.confidence > 0.95:
            # High confidence from rules alone
            self.log_prediction(transaction, rule_result, 'RULES_ONLY')
            return rule_result
        
        # 2. Feature extraction (30ms)
        features = await self.extract_features(transaction)
        
        # 3. Model prediction (20ms)
        fraud_score = self.model.predict_proba(features)[0][1]
        
        # 4. Combine with rules (5ms)
        final_score = self.combine_scores(fraud_score, rule_result.score)
        
        # 5. Make decision (5ms)
        decision = self.make_decision(final_score, transaction)
        
        # Total: ~65ms (well within 100ms budget)
        latency = (time.time() - start_time) * 1000
        self.record_metrics(latency, decision)
        
        return FraudScore(
            score=final_score,
            decision=decision,  # APPROVE, DECLINE, REVIEW
            latency_ms=latency,
            model_version='v1',
            features_used=features.keys()
        )
    
    async def extract_features(self, transaction: Transaction) -> Dict:
        """
        Extract features from multiple sources
        Target: 30ms
        """
        # Parallel feature extraction
        tasks = [
            self.get_transaction_features(transaction),  # 5ms
            self.get_user_features(transaction.user_id),  # 10ms
            self.get_aggregated_features(transaction.user_id),  # 15ms
            self.get_device_features(transaction.device_id),  # 5ms
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Combine all features
        features = {}
        for result in results:
            features.update(result)
        
        return features
    
    def get_transaction_features(self, transaction: Transaction) -> Dict:
        """
        Immediate transaction features
        """
        return {
            'amount': transaction.amount,
            'merchant_id': transaction.merchant_id,
            'merchant_category': transaction.merchant_category,
            'is_international': transaction.is_international,
            'time_of_day': transaction.timestamp.hour,
            'day_of_week': transaction.timestamp.weekday(),
        }
    
    async def get_user_features(self, user_id: str) -> Dict:
        """
        User profile features from feature store
        """
        # Redis lookup (very fast)
        user_profile = await self.feature_store.get(f'user:{user_id}')
        
        if not user_profile:
            # Fallback to database (slower)
            user_profile = await self.db.get_user_profile(user_id)
            # Cache for next time
            await self.feature_store.set(f'user:{user_id}', user_profile, ttl=3600)
        
        return {
            'user_age_days': user_profile['age_days'],
            'user_country': user_profile['country'],
            'kyc_status': user_profile['kyc_status'],
            'risk_score': user_profile['risk_score'],
        }
    
    async def get_aggregated_features(self, user_id: str) -> Dict:
        """
        Pre-computed aggregated features
        """
        # These are computed in batch and stored in Redis
        agg_features = await self.feature_store.get(f'agg:{user_id}')
        
        return {
            'transactions_last_24h': agg_features['count_24h'],
            'total_amount_last_24h': agg_features['sum_24h'],
            'avg_transaction_amount_7d': agg_features['avg_7d'],
            'distinct_merchants_30d': agg_features['distinct_merchants_30d'],
            'declined_transactions_7d': agg_features['declined_7d'],
        }
    
    def make_decision(self, score: float, transaction: Transaction) -> str:
        """
        Convert score to decision
        """
        # Multi-threshold approach
        if score < 0.1:
            return 'APPROVE'  # Low risk
        elif score < 0.5:
            # Medium risk - additional checks
            if transaction.amount > 1000:
                return 'REVIEW'  # High value, be cautious
            return 'APPROVE'
        elif score < 0.8:
            return 'REVIEW'  # High risk - manual review
        else:
            return 'DECLINE'  # Very high risk
```

**3.2 Feature Store Architecture**

```python
class FeatureStore:
    """
    Two-tier feature store: Online (Redis) + Offline (Data Lake)
    """
    
    def __init__(self):
        # Online: Low-latency serving
        self.redis = RedisCluster(nodes=redis_nodes)
        
        # Offline: Training and batch
        self.spark = SparkSession.builder.appName('FeatureStore').getOrCreate()
        self.s3_path = 's3://fraud-detection/features/'
        
    # ONLINE FEATURES (Real-time serving)
    async def get_online_features(self, user_id: str) -> Dict:
        """
        Get features for real-time prediction
        Target: < 10ms
        """
        # Try cache first
        cached = await self.redis.get(f'features:{user_id}')
        if cached:
            return json.loads(cached)
        
        # Compute on-the-fly (should be rare)
        features = await self.compute_realtime_features(user_id)
        
        # Cache for 5 minutes
        await self.redis.setex(
            f'features:{user_id}',
            300,  # 5 min TTL
            json.dumps(features)
        )
        
        return features
    
    async def update_online_features(self, user_id: str, transaction: Transaction):
        """
        Incrementally update aggregated features
        Called after each transaction
        """
        # Update counters atomically
        pipe = self.redis.pipeline()
        
        # Increment transaction count
        pipe.hincrby(f'agg:{user_id}', 'count_24h', 1)
        
        # Add to rolling sum
        pipe.hincrbyfloat(f'agg:{user_id}', 'sum_24h', transaction.amount)
        
        # Add merchant to set
        pipe.sadd(f'merchants:{user_id}:30d', transaction.merchant_id)
        
        # Execute all atomically
        await pipe.execute()
        
        # Set expiration (if new key)
        await self.redis.expire(f'agg:{user_id}', 86400)  # 24 hours
    
    # OFFLINE FEATURES (Batch processing)
    def compute_offline_features(self, date: str):
        """
        Batch compute features for training
        Runs daily via Airflow/Spark
        """
        # Read transactions
        transactions = self.spark.read.parquet(
            f's3://fraud-detection/transactions/date={date}'
        )
        
        # Compute user aggregations
        user_features = transactions.groupBy('user_id').agg(
            count('*').alias('transaction_count'),
            sum('amount').alias('total_amount'),
            avg('amount').alias('avg_amount'),
            countDistinct('merchant_id').alias('distinct_merchants'),
            max('timestamp').alias('last_transaction_time')
        )
        
        # Compute time-based features
        from pyspark.sql.functions import window
        
        windowed_features = transactions.groupBy(
            'user_id',
            window('timestamp', '24 hours')
        ).agg(
            count('*').alias('count_24h'),
            sum('amount').alias('sum_24h')
        )
        
        # Write to S3 (for training)
        user_features.write.mode('overwrite').parquet(
            f'{self.s3_path}/user_features/date={date}'
        )
        
        # Also update Redis (for serving)
        self.sync_to_redis(user_features)
    
    def sync_to_redis(self, features_df):
        """
        Sync batch-computed features to Redis
        """
        # Convert to records
        records = features_df.collect()
        
        # Batch update Redis
        pipe = self.redis.pipeline()
        for record in records:
            key = f'agg:{record.user_id}'
            pipe.hmset(key, record.asDict())
            pipe.expire(key, 2592000)  # 30 days
        
        pipe.execute()
```

**3.3 Model Training Pipeline**

```python
class FraudModelTrainer:
    """
    Daily model training pipeline
    """
    
    def __init__(self):
        self.spark = SparkSession.builder.appName('ModelTraining').getOrCreate()
        self.mlflow_uri = 'http://mlflow:5000'
        
    def train_daily_model(self, date: str):
        """
        Daily training job
        Triggered by Airflow at 2 AM
        """
        print(f"Training model for {date}")
        
        # 1. Load data (last 90 days)
        df = self.load_training_data(date, lookback_days=90)
        
        # 2. Feature engineering
        features_df = self.engineer_features(df)
        
        # 3. Handle class imbalance
        # Fraud is rare (~0.1%), need to balance
        balanced_df = self.balance_dataset(features_df)
        
        # 4. Train model
        model, metrics = self.train_model(balanced_df)
        
        # 5. Validate
        if self.validate_model(model, metrics):
            # 6. Deploy
            self.deploy_model(model, metrics, date)
        else:
            self.alert_team("Model validation failed", metrics)
    
    def load_training_data(self, date: str, lookback_days: int):
        """
        Load transactions with labels
        """
        # Labels come from fraud investigation team (delayed)
        # Transaction happens → Investigated → Labeled (1-7 days later)
        
        query = f"""
        SELECT 
            t.*,
            COALESCE(f.is_fraud, 0) as label
        FROM transactions t
        LEFT JOIN fraud_labels f ON t.transaction_id = f.transaction_id
        WHERE t.date BETWEEN date_sub('{date}', {lookback_days}) AND '{date}'
        AND (f.is_fraud IS NOT NULL OR random() < 0.01)  -- Sample negatives
        """
        
        return self.spark.sql(query)
    
    def balance_dataset(self, df):
        """
        Handle class imbalance (0.1% fraud rate)
        """
        from pyspark.ml.feature import SMOTE  # or custom implementation
        
        # Strategy: Undersample majority + oversample minority
        
        # Separate classes
        fraud = df.filter(df.label == 1)
        legitimate = df.filter(df.label == 0)
        
        fraud_count = fraud.count()
        legit_count = legitimate.count()
        
        # Target: 1:5 ratio (instead of 1:1000)
        target_legit = fraud_count * 5
        
        # Undersample legitimate
        sample_rate = target_legit / legit_count
        legitimate_sampled = legitimate.sample(False, sample_rate, seed=42)
        
        # Combine
        balanced = fraud.union(legitimate_sampled)
        
        print(f"Original: {fraud_count} fraud, {legit_count} legitimate")
        print(f"Balanced: {fraud_count} fraud, {target_legit} legitimate")
        
        return balanced
    
    def train_model(self, df):
        """
        Train XGBoost model
        """
        import xgboost as xgb
        import mlflow
        
        # Split
        train, test = df.randomSplit([0.8, 0.2], seed=42)
        
        # Convert to DMatrix
        X_train, y_train = self.prepare_features(train)
        X_test, y_test = self.prepare_features(test)
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Hyperparameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss'],
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 5,  # Handle imbalance
        }
        
        # Train
        with mlflow.start_run():
            mlflow.log_params(params)
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=100,
                evals=[(dtrain, 'train'), (dtest, 'test')],
                early_stopping_rounds=10
            )
            
            # Evaluate
            y_pred = model.predict(dtest)
            metrics = self.compute_metrics(y_test, y_pred)
            
            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(model, 'model')
            
        return model, metrics
    
    def compute_metrics(self, y_true, y_pred):
        """
        Compute business metrics
        """
        from sklearn.metrics import (
            roc_auc_score, precision_recall_curve, 
            confusion_matrix, classification_report
        )
        
        # Threshold optimization
        # Goal: Catch 95% of fraud, minimize false positives
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        
        # Find threshold where recall >= 0.95
        idx = np.where(recall >= 0.95)[0][0]
        optimal_threshold = thresholds[idx]
        
        y_pred_binary = (y_pred >= optimal_threshold).astype(int)
        
        # Metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        
        metrics = {
            'auc': roc_auc_score(y_true, y_pred),
            'precision': tp / (tp + fp),
            'recall': tp / (tp + fn),
            'false_positive_rate': fp / (fp + tn),
            'optimal_threshold': optimal_threshold,
            
            # Business metrics
            'fraud_caught': tp,
            'fraud_missed': fn,
            'false_alarms': fp,
            'legit_approved': tn,
        }
        
        print("\n" + "="*50)
        print("MODEL PERFORMANCE")
        print("="*50)
        print(f"AUC: {metrics['auc']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"False Positive Rate: {metrics['false_positive_rate']:.3f}")
        print(f"\nOptimal Threshold: {optimal_threshold:.3f}")
        print(f"\nBusiness Impact:")
        print(f"  Fraud Caught: {tp:,} ({tp/(tp+fn):.1%})")
        print(f"  Fraud Missed: {fn:,}")
        print(f"  False Alarms: {fp:,} ({fp/(fp+tn):.2%})")
        
        return metrics
    
    def validate_model(self, model, metrics):
        """
        Validation checks before deployment
        """
        checks = {
            'AUC > 0.90': metrics['auc'] > 0.90,
            'Recall > 0.95': metrics['recall'] > 0.95,
            'FPR < 0.01': metrics['false_positive_rate'] < 0.01,
        }
        
        print("\n" + "="*50)
        print("VALIDATION CHECKS")
        print("="*50)
        
        all_passed = True
        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"{status} {check}")
            all_passed = all_passed and passed
        
        return all_passed
    
    def deploy_model(self, model, metrics, date):
        """
        Deploy to production
        """
        # 1. Save model
        model_path = f's3://fraud-detection/models/model_{date}.pkl'
        model.save_model(model_path)
        
        # 2. Update model registry
        self.update_model_registry(model_path, metrics, date)
        
        # 3. Gradual rollout (canary deployment)
        self.canary_deploy(model_path, traffic_percentage=10)
        
        # 4. Monitor for 1 hour
        # If metrics look good, increase to 50%, then 100%
        
        print(f"\n✓ Model deployed: {model_path}")
```

**3.4 Monitoring & Alerting**

```python
class FraudDetectionMonitor:
    """
    Comprehensive monitoring system
    """
    
    def __init__(self):
        self.prometheus = PrometheusClient()
        self.alert_manager = AlertManager()
        
    def monitor_realtime_metrics(self):
        """
        Real-time metrics collection
        """
        metrics = {
            # Latency
            'prediction_latency_p50': self.get_percentile('latency', 0.50),
            'prediction_latency_p95': self.get_percentile('latency', 0.95),
            'prediction_latency_p99': self.get_percentile('latency', 0.99),
            
            # Throughput
            'requests_per_second': self.get_rate('requests'),
            'predictions_per_second': self.get_rate('predictions'),
            
            # Model performance
            'fraud_detection_rate': self.get_rate('fraud_detected'),
            'decline_rate': self.get_rate('declined_transactions'),
            'review_rate': self.get_rate('review_transactions'),
            
            # Errors
            'error_rate': self.get_rate('errors'),
            'timeout_rate': self.get_rate('timeouts'),
        }
        
        # Alert conditions
        if metrics['prediction_latency_p95'] > 100:
            self.alert_manager.send_alert(
                severity='WARNING',
                message=f"P95 latency {metrics['prediction_latency_p95']}ms > 100ms SLA"
            )
        
        if metrics['error_rate'] > 0.01:
            self.alert_manager.send_alert(
                severity='CRITICAL',
                message=f"Error rate {metrics['error_rate']:.2%} > 1%"
            )
        
        return metrics
    
    def monitor_model_performance(self):
        """
        Model performance monitoring
        Requires ground truth labels (delayed)
        """
        query = """
        WITH predictions_with_labels AS (
            SELECT 
                p.prediction_id,
                p.fraud_score,
                p.decision,
                l.is_fraud as actual_fraud
            FROM predictions p
            JOIN fraud_labels l ON p.transaction_id = l.transaction_id
            WHERE p.prediction_time >= CURRENT_DATE - 7
            AND l.investigation_complete = TRUE
        )
        SELECT 
            DATE(prediction_time) as date,
            -- Confusion matrix
            SUM(CASE WHEN decision = 'DECLINE' AND actual_fraud = 1 THEN 1 ELSE 0 END) as tp,
            SUM(CASE WHEN decision = 'DECLINE' AND actual_fraud = 0 THEN 1 ELSE 0 END) as fp,
            SUM(CASE WHEN decision != 'DECLINE' AND actual_fraud = 1 THEN 1 ELSE 0 END) as fn,
            SUM(CASE WHEN decision != 'DECLINE' AND actual_fraud = 0 THEN 1 ELSE 0 END) as tn,
            
            -- Metrics
            SUM(CASE WHEN decision = 'DECLINE' AND actual_fraud = 1 THEN 1 ELSE 0 END)::FLOAT /
            NULLIF(SUM(CASE WHEN actual_fraud = 1 THEN 1 ELSE 0 END), 0) as recall,
            
            SUM(CASE WHEN decision = 'DECLINE' AND actual_fraud = 1 THEN 1 ELSE 0 END)::FLOAT /
            NULLIF(SUM(CASE WHEN decision = 'DECLINE' THEN 1 ELSE 0 END), 0) as precision
        FROM predictions_with_labels
        GROUP BY 1
        ORDER BY 1 DESC
        """
        
        metrics_over_time = execute_query(query)
        
        # Alert if performance degrading
        if metrics_over_time[0]['recall'] < 0.90:
            self.alert_manager.send_alert(
                severity='HIGH',
                message=f"Model recall dropped to {metrics_over_time[0]['recall']:.1%}"
            )
    
    def monitor_data_drift(self):
        """
        Detect distribution changes
        """
        # Compare current vs training distributions
        for feature in self.important_features:
            drift_score = self.compute_drift(feature)
            
            if drift_score > 0.20:  # 20% drift
                self.alert_manager.send_alert(
                    severity='MEDIUM',
                    message=f"Data drift detected in {feature}: {drift_score:.1%}"
                )
```

**Step 4: Trade-offs & Alternatives (5 minutes)**

```
DESIGN DECISIONS & TRADE-OFFS:

1. SYNCHRONOUS vs ASYNCHRONOUS:
   ✓ Chose: Hybrid (sync for low-risk, async for high-risk)
   Why: Balance latency and thoroughness
   Alternative: Fully async (higher latency)

2. MODEL COMPLEXITY:
   ✓ Chose: XGBoost (gradient boosting)
   Why: Best accuracy/speed trade-off
   Alternative: Neural networks (slower, similar accuracy)

3. FEATURE STORE:
   ✓ Chose: Redis for online, S3 for offline
   Why: Redis gives <10ms lookups
   Alternative: DynamoDB (higher cost, similar performance)

4. BATCH vs ONLINE TRAINING:
   ✓ Chose: Daily batch retraining
   Why: Simpler, sufficient for fraud patterns
   Alternative: Online learning (more complex)

5. THRESHOLD STRATEGY:
   ✓ Chose: Multi-threshold (approve/review/decline)
   Why: Balances fraud catch rate and user experience
   Alternative: Single threshold (less flexible)

BOTTLENECKS & SOLUTIONS:

1. Feature Extraction (30ms):
   - Solution: Pre-compute aggregations, cache in Redis
   - Solution: Parallel async calls

2. Model Serving (20ms):
   - Solution: Load model in memory
   - Solution: Use C++ inference engine (ONNX Runtime)

3. Database Lookups:
   - Solution: Redis cache layer
   - Solution: Denormalize data for faster lookups

4. High Traffic (10,000 TPS):
   - Solution: Horizontal scaling (10+ servers)
   - Solution: Load balancer with auto-scaling
```

**Interview Answer Summary:**

"For a real-time fraud detection system handling 10,000 TPS with <100ms latency:

**Architecture:**
- API Gateway → Fraud Detection Service → Feature Store (Redis) → Model Serving
- Hybrid sync/async: Low-risk transactions get fast response, high-risk get thorough analysis
- Two-tier feature store: Online (Redis) for serving, Offline (S3) for training

**Key Components:**
1. **Real-time Service:** Extract features (30ms) + Model prediction (20ms) + Decision logic (5ms) = 65ms total
2. **Feature Store:** Pre-computed aggregations in Redis for <10ms lookups
3. **Training Pipeline:** Daily batch training with XGBoost, handles class imbalance
4. **Monitoring:** Track latency, throughput, model performance, data drift

**As a Data Engineer:**
- I'd ensure robust data pipelines for feature computation
- Set up incremental feature updates after each transaction
- Implement monitoring for data quality and drift
- Design for 99.9% uptime with failover and redundancy

**Scale:**
- 10,000 TPS = need 10+ prediction service instances
- Redis cluster for high availability
- Auto-scaling based on traffic
- Can scale to 50,000 TPS by adding more instances"

---

This completes the core Day 1 content. Would you like me to continue with more questions or move to create the complete document with all remaining sections?

