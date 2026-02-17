## MLOps & Related Concepts

Production machine learning: deploying, monitoring, and maintaining ML systems.

---

### Model Deployment

**Definition:** The process of integrating a trained machine learning model into a production environment where it can make predictions on new, real-world data.

**Explanation:** Training a model is only half the battle—deployment is where it provides actual business value. Deployment means taking your model from a Jupyter notebook or training script and making it available to serve predictions at scale, reliably, and securely.

**Deployment Stages:**

**1. Model Export:**
- Save trained model (pickle, ONNX, TensorFlow SavedModel)
- Include preprocessing pipeline
- Version the model

**2. Environment Setup:**
- Package dependencies
- Configure infrastructure (servers, containers)
- Set up monitoring

**3. Integration:**
- Expose via API (REST, gRPC)
- Connect to data sources
- Integrate with existing systems

**4. Testing:**
- Verify predictions match expected format
- Load testing (can handle traffic?)
- A/B testing (compare to baseline)

**Deployment Patterns:**

**Batch Prediction:**
- Process data in batches (hourly, daily)
- Not real-time
- **Example:** Nightly customer churn predictions
- **Pros:** Simple, efficient for large volumes
- **Cons:** Stale predictions, not interactive

**Online/Real-time Prediction:**
- Predict on-demand as requests arrive
- Low latency required
- **Example:** Fraud detection during transaction
- **Pros:** Fresh predictions, interactive
- **Cons:** Complex infrastructure, latency constraints

**Edge Deployment:**
- Model runs on device (phone, IoT)
- No network required
- **Example:** Face unlock on smartphone
- **Pros:** Fast, private, works offline
- **Cons:** Limited compute, model size constraints

**Model as a Service:**
- Host model behind API endpoint
- Clients send requests, receive predictions
- **Example:** POST /predict with JSON data
- **Pros:** Language-agnostic, scalable
- **Cons:** Network latency, security considerations

**Infrastructure Options:**

**Cloud Platforms:**
- AWS SageMaker, Google AI Platform, Azure ML
- Managed infrastructure, autoscaling
- Pay per use

**Containers:**
- Docker containers with model + dependencies
- Kubernetes for orchestration
- Portable, reproducible

**Serverless:**
- AWS Lambda, Google Cloud Functions
- Scale to zero when idle
- Cost-effective for sporadic traffic

**On-Premise:**
- Own servers/hardware
- Full control, regulatory compliance
- Higher upfront cost, maintenance burden

**Example: Fraud Detection Deployment**

**Training:** Model achieves 95% accuracy on test set
**Export:** Save as model.pkl with preprocessing pipeline
**Containerize:** Create Docker image with Flask API
**Deploy:** Kubernetes cluster on AWS
**Integrate:** Payment processor calls API during transactions
**Monitor:** Track latency, error rates, prediction distribution

**Challenges:**

1. **Latency:** Predictions must be fast enough
2. **Scalability:** Handle traffic spikes
3. **Reliability:** High uptime requirements
4. **Security:** Protect model and data
5. **Versioning:** Manage multiple model versions
6. **Rollback:** Revert if new model performs poorly

**Key Metrics:**

- **Latency:** Time from request to response (P50, P95, P99)
- **Throughput:** Requests per second
- **Availability:** Uptime percentage (99.9% = "three nines")
- **Error Rate:** Failed predictions/total predictions

**Best Practices:**

1. **Version everything:** Code, model, data, config
2. **Gradual rollout:** Start with small traffic percentage
3. **Canary deployment:** Test on subset before full rollout
4. **Blue-green deployment:** Switch between old/new instantly
5. **Health checks:** Monitor model is responding correctly
6. **Graceful degradation:** Fallback when model fails

**Key Insight:** Deployment is where ML meets software engineering. A 90% accurate model in production beats a 95% accurate model on your laptop. Focus on reliability, speed, and maintainability—not just accuracy.

---

### Model Monitoring

**Definition:** Continuous tracking of a deployed model's performance, behavior, and health to detect issues before they impact users.

**Explanation:** Once deployed, models don't "just work" forever. Data changes, user behavior evolves, bugs emerge, and performance degrades. Monitoring detects these problems early so you can fix them before major damage. It's like a health checkup for your ML system.

**What to Monitor:**

**1. Performance Metrics:**

Track the actual accuracy, precision, recall, etc. in production

**Challenge:** Often don't have ground truth labels immediately
**Solutions:**
- Sample data for manual labeling
- Wait for delayed labels (fraud discovered weeks later)
- Use proxy metrics (user satisfaction, click rates)

**Example:** Fraud model
- **During training:** 95% recall on test set
- **In production:** Monitor daily recall on labeled fraud cases
- **Alert:** If recall drops below 90%, investigate

**2. Prediction Distribution:**

Monitor what the model is predicting

**Healthy:** Distribution matches training/validation
**Unhealthy:** Sudden shifts suggest problems

**Example:** Churn prediction
- **Normal:** 20% customers predicted high churn risk
- **Alert:** Suddenly 60% high risk → investigate model or data issue

**3. Input Distribution:**

Monitor incoming feature distributions

**Data Drift:** Input features change over time
**Alert:** If features shift significantly from training distribution

**Example:** Loan approval model
- **Training:** Average income $50k
- **Production:** Average income now $40k → retrain needed

**4. System Health:**

Infrastructure and operational metrics

- **Latency:** Response time per prediction
- **Throughput:** Predictions per second
- **Error rate:** Failed requests / total requests
- **Resource usage:** CPU, memory, GPU utilization
- **Uptime:** System availability

**5. Data Quality:**

Incoming data sanity checks

- **Missing values:** More nulls than expected?
- **Outliers:** Extreme values that shouldn't exist
- **Schema changes:** New/removed features
- **Invalid values:** Negative ages, future dates

**Monitoring Techniques:**

**Statistical Tests:**
- Compare production data distribution to training data
- Kolmogorov-Smirnov test, Chi-square test
- Alert if distributions diverge significantly

**Thresholds:**
- Set acceptable ranges for metrics
- Alert if outside range
- **Example:** Latency must be < 100ms at P95

**Trend Analysis:**
- Track metrics over time
- Detect gradual degradation
- **Example:** Accuracy slowly dropping 1% per week

**Anomaly Detection:**
- Flag unusual patterns
- **Example:** Prediction distribution spike at 3am

**Tools:**

**Open Source:**
- Prometheus + Grafana (metrics + dashboards)
- ELK Stack (logging)
- Evidently AI (ML-specific monitoring)

**Commercial:**
- Datadog, New Relic (APM)
- Arize AI, Fiddler (ML monitoring)
- AWS CloudWatch, GCP Monitoring

**Example Monitoring Dashboard:**

```
┌─────────────────────────────────────┐
│ Fraud Detection Model - Live Status │
├─────────────────────────────────────┤
│ Latency (P95):  78ms  ✓             │
│ Error Rate:     0.2%  ✓             │
│ Predictions/s:  1,200 ✓             │
│ Fraud %:        2.1%  ⚠ (was 1.8%)  │
│ Avg Amount:     $450  ⚠ (was $320)  │
└─────────────────────────────────────┘

📊 Graphs: latency, throughput, prediction dist
🚨 Alerts: 2 active warnings
```

**Alert Strategies:**

**Severity Levels:**
- **Critical:** Model down, very high error rate
- **Warning:** Performance degrading, data drift
- **Info:** Minor changes, worth noting

**Alert Fatigue:**
- Too many alerts → ignored
- Set thresholds carefully
- Aggregate related alerts

**On-Call Rotation:**
- Assign responsibility
- Escalation procedures
- Runbooks for common issues

**When to Retrain:**

Monitoring tells you when model needs refresh:

1. **Performance drop:** Accuracy declining
2. **Data drift:** Input distribution changed
3. **Concept drift:** Relationship between X and Y changed
4. **New patterns:** World evolved (COVID changed behavior)

**Key Insight:** "Deploy and forget" doesn't work. Models degrade over time. Good monitoring detects problems early, enabling proactive fixes rather than reactive firefighting after users complain.

---

### Model Drift

**Definition:** The phenomenon where a deployed model's performance degrades over time because the statistical properties of the data or the underlying relationships have changed.

**Explanation:** The world is not static. User behavior evolves, markets change, new trends emerge, and the data your model sees in production starts to differ from training data. This causes "drift," and your once-accurate model becomes increasingly wrong. Drift is why ML systems need continuous monitoring and retraining.

**Types of Drift:**

### Data Drift (Covariate Shift)

**Definition:** The distribution of input features changes, but the relationship between features and target stays the same.

**Example: E-commerce Recommendation**

**Training:** Users mostly browse electronics (2020)
**Production:** COVID-19 → users browse home goods (2021)
**Result:** Features shifted, model trained on electronics data performs poorly

**P(X) changed, but P(Y|X) stayed the same**

**Detection:**
- Compare feature distributions (training vs production)
- Statistical tests (KS test, Chi-square)
- Visual inspection (histograms)

**Solution:**
- Retrain with recent data
- Update feature distributions
- May not need to change model architecture

### Concept Drift

**Definition:** The relationship between features and target changes—what used to predict Y no longer does.

**Example: Fraud Detection**

**Training:** Fraudsters use pattern A (stolen cards at gas stations)
**Production:** Fraudsters adapt, now use pattern B (online purchases)
**Result:** Old patterns no longer indicate fraud, new patterns do

**P(Y|X) changed** - the mapping from features to labels evolved

**Types:**

**Sudden Drift:** Abrupt change (new regulation, system update)
**Gradual Drift:** Slow evolution (changing user preferences)
**Recurring Drift:** Seasonal patterns (holiday shopping)
**Incremental Drift:** Continuous small changes

**Detection:**
- Monitor model performance over time
- Track prediction accuracy on labeled samples
- Compare predictions to delayed ground truth

**Solution:**
- Retrain with recent labeled data
- May need new features to capture new patterns
- Consider online learning (continuous updates)

### Label Drift (Prior Probability Shift)

**Definition:** The distribution of target labels changes, but features and their relationship to labels stay the same.

**Example: Spam Detection**

**Training:** 30% emails are spam
**Production:** Spam filters improve globally → only 10% spam
**Result:** Model over-predicts spam (calibrated for 30%)

**P(Y) changed, but P(X|Y) stayed the same**

**Solution:**
- Recalibrate model thresholds
- May need to retrain
- Adjust decision boundaries

**Real-World Examples:**

**1. COVID-19 Impact:**

**Before:** Travel booking model trained on 2019 data
**Pandemic:** Travel patterns completely changed
**Result:** Catastrophic drift, model useless

**2. Seasonal Drift:**

**Model:** Retail demand forecasting
**Summer:** High demand for swimwear
**Winter:** Same model predicts swimwear demand (wrong!)
**Result:** Predictable recurring drift

**3. Adversarial Drift:**

**Model:** Credit card fraud detection
**Initial:** Catches 90% of fraud
**Adaptation:** Fraudsters learn patterns to avoid detection
**Result:** Gradual performance decline as adversaries adapt

**Detecting Drift:**

**Method 1: Performance Monitoring**
- Track accuracy, precision, recall over time
- **Limitation:** Requires labeled data

**Method 2: Statistical Tests**
- Compare distributions (training vs recent production)
- KS test, PSI (Population Stability Index)
- **Advantage:** Works without labels

**Method 3: Prediction Monitoring**
- Track prediction distribution
- Alert if shifts significantly
- **Example:** Sudden spike in "high risk" predictions

**Method 4: Model Confidence**
- Monitor prediction confidence/entropy
- Low confidence → model uncertain (possible drift)

**Handling Drift:**

**1. Scheduled Retraining:**
- Retrain weekly, monthly, quarterly
- **Pros:** Simple, predictable
- **Cons:** May retrain unnecessarily or not enough

**2. Triggered Retraining:**
- Retrain when drift detected
- **Pros:** Adaptive, efficient
- **Cons:** Need good drift detection

**3. Online Learning:**
- Continuously update model with new data
- **Pros:** Always current
- **Cons:** Complex, risk of catastrophic forgetting

**4. Ensemble Approaches:**
- Combine old and new models
- Weight recent models higher
- **Pros:** Smooth transitions
- **Cons:** More complex serving

**Prevention Strategies:**

1. **Feature robustness:** Use stable features less prone to drift
2. **Regular retraining:** Don't wait for catastrophic failure
3. **Data recency:** Weight recent data higher
4. **Monitoring:** Detect drift early
5. **Feedback loops:** Collect labels for continuous improvement

**Key Insight:** Drift is inevitable in real-world ML systems. The question isn't "will my model drift?" but "when and how fast?" Build systems that assume drift will happen and handle it gracefully through monitoring, alerting, and automated retraining.

---

### A/B Testing

**Definition:** An experimental method for comparing two versions (A and B) to determine which performs better, commonly used to validate new ML models before full deployment.

**Explanation:** Before replacing your production model, prove the new one is actually better. A/B testing splits traffic between old model (A) and new model (B), measures real-world performance, and uses statistics to determine which is superior. It's the gold standard for model validation in production.

**How It Works:**

1. **Control (A):** Existing model serves 50% of traffic
2. **Treatment (B):** New model serves other 50% of traffic
3. **Random assignment:** Users randomly assigned to A or B
4. **Measurement:** Track key metrics for each group
5. **Analysis:** Statistical test determines if difference is significant

**Example: Recommendation System**

**Model A (Current):** Collaborative filtering, 3% click-through rate (CTR)
**Model B (New):** Neural network, claims 3.5% CTR in offline tests

**A/B Test:**
- 50% users see recommendations from Model A
- 50% users see recommendations from Model B
- Run for 2 weeks
- Measure actual CTR

**Results:**
- Model A: 3.0% CTR (10,000 impressions, 300 clicks)
- Model B: 3.6% CTR (10,000 impressions, 360 clicks)
- Statistical significance: p=0.01 (significant!)
- **Decision:** Deploy Model B to 100% traffic

**Key Concepts:**

**Statistical Significance:**
- Is difference real or due to chance?
- p-value < 0.05 typically considered significant
- Confidence intervals show range of true effect

**Sample Size:**
- Too small → can't detect real differences
- Too large → wastes time and resources
- Calculate required size based on expected effect

**Duration:**
- Long enough to capture patterns (weekday/weekend)
- Account for seasonality
- Typical: 1-4 weeks

**Metrics:**

**Business Metrics:** What actually matters
- Revenue, conversions, retention, engagement
- **Example:** For fraud model, measure $ of fraud caught vs false positives

**Model Metrics:** Technical performance
- Accuracy, latency, error rate
- Support business metrics but not the goal

**Guardrail Metrics:** Must not get worse
- User experience, system load, cost
- **Example:** Latency must stay < 100ms

**Variations:**

**A/B/C Testing:**
- Compare multiple variants (A vs B vs C)
- **Example:** Test 3 different models simultaneously

**Canary Deployment:**
- Start with 5% traffic to B, gradually increase
- **Safer:** Limit blast radius if B is worse

**Champion/Challenger:**
- A is current champion
- B is challenger trying to dethrone it
- Only replace if B clearly wins

**Multi-Armed Bandit:**
- Dynamically allocate more traffic to better model
- **Advantage:** Minimize cost of inferior model
- **Disadvantage:** Harder to get clean statistical test

**Common Pitfalls:**

**1. Not Running Long Enough:**
- Stop early → false conclusions
- Need sufficient data for statistical power

**2. Peeking:**
- Checking results repeatedly → inflated false positives
- **Fix:** Pre-commit to sample size and duration

**3. Multiple Comparisons:**
- Testing 20 metrics → 1 will be significant by chance
- **Fix:** Bonferroni correction or pre-specify primary metric

**4. Selection Bias:**
- Non-random assignment → groups not comparable
- **Fix:** Proper randomization

**5. Novelty Effect:**
- New model gets attention → short-term boost
- **Fix:** Run longer to capture steady-state behavior

**Example: Email Spam Filter**

**Scenario:** New deep learning spam filter

**Setup:**
- Model A (current): Rule-based + logistic regression
- Model B (new): LSTM neural network
- Split: 50/50 for 2 weeks

**Metrics:**
- **Primary:** Spam caught (recall)
- **Secondary:** Legitimate emails blocked (precision)
- **Guardrail:** User complaints

**Results:**
```
           Spam Caught  False Positives  User Complaints
Model A    92%          0.5%             10
Model B    95%          0.4%             8
```

**Analysis:**
- B catches 3% more spam (significant, p=0.02)
- B has fewer false positives (significant, p=0.04)
- Fewer complaints (directionally better)

**Decision:** Deploy Model B ✓

**When NOT to A/B Test:**

1. **Insufficient traffic:** Need thousands of users for statistical power
2. **Critical changes:** Can't risk exposing users to potentially worse model
3. **Offline metrics sufficient:** Sometimes offline evaluation is enough
4. **High cost:** A/B testing expensive models may not be feasible

**Key Insight:** "Offline accuracy improved 2%" doesn't mean production performance will improve. Real users in real environments are the ultimate test. A/B testing prevents deploying models that look good in development but fail in production. Always validate with real traffic before full rollout.

---

### Experiment Tracking

**Definition:** Systematic recording of machine learning experiments including code, data, hyperparameters, metrics, and artifacts to ensure reproducibility and facilitate comparison.

**Explanation:** ML development is iterative—you try dozens or hundreds of experiments. Without tracking, you lose critical information: Which hyperparameters gave 92% accuracy? What data was used for the best model? Can I reproduce last week's results? Experiment tracking solves these problems by automatically logging everything.

**What to Track:**

**1. Code:**
- Git commit hash
- Branch name
- Changed files

**2. Data:**
- Dataset version
- Train/validation/test splits
- Data preprocessing steps

**3. Hyperparameters:**
- Learning rate, batch size, epochs
- Model architecture
- Regularization parameters

**4. Metrics:**
- Training/validation loss per epoch
- Final accuracy, precision, recall
- Convergence time

**5. Artifacts:**
- Trained model files
- Plots (loss curves, confusion matrices)
- Logs

**6. Environment:**
- Python version, library versions
- Hardware (GPU type, RAM)
- Random seeds

**7. Metadata:**
- Experiment name, description
- Timestamp, duration
- Author, tags

**Example Experiment Log:**

```
Experiment: fraud-detection-v12
Date: 2024-01-15 14:30
Duration: 2h 15m
Git: commit a3f5d2c on branch feature/lstm

Data:
  - Dataset: transactions_2024_q1.csv (50K samples)
  - Split: 70/15/15 train/val/test
  - Features: 23 numerical, 5 categorical

Hyperparameters:
  - Model: LSTM(128, 64)
  - Batch size: 32
  - Learning rate: 0.001
  - Epochs: 50
  - Dropout: 0.3
  - Optimizer: Adam

Results:
  - Train accuracy: 94.2%
  - Val accuracy: 91.8%
  - Test accuracy: 91.5%
  - Precision: 0.89
  - Recall: 0.93
  - F1: 0.91

Artifacts:
  - Model: models/fraud_v12.h5
  - Plots: plots/training_curve.png
```

**Tools:**

**MLflow:**
- Open source, language-agnostic
- Tracks experiments, packages models
- **Example:** `mlflow.log_param("lr", 0.001)`

**Weights & Biases (W&B):**
- Cloud-based, great visualizations
- Real-time monitoring
- Collaboration features

**TensorBoard:**
- From TensorFlow, works with PyTorch too
- Excellent for tracking training curves
- Embedding visualization

**Neptune.ai:**
- Cloud-based, metadata-rich
- Good for teams
- Model registry integration

**Example: Using MLflow**

```python
import mlflow

# Start experiment
mlflow.start_run()

# Log parameters
mlflow.log_param("learning_rate", 0.001)
mlflow.log_param("batch_size", 32)

# Train model
model = train_model(X_train, y_train)

# Log metrics
mlflow.log_metric("accuracy", 0.915)
mlflow.log_metric("f1_score", 0.91)

# Log model
mlflow.sklearn.log_model(model, "model")

# End experiment
mlflow.end_run()
```

Now all experiments are searchable:
```bash
mlflow ui  # Opens web dashboard
```

**Benefits:**

**1. Reproducibility:**
- Recreate exact experiment months later
- Share experiments with team
- Debug issues by comparing past runs

**2. Comparison:**
- Compare 100+ experiments side-by-side
- Find best hyperparameters
- Visualize trends (learning rate vs accuracy)

**3. Collaboration:**
- Team sees what others tried
- Avoid duplicate work
- Build on each other's experiments

**4. Debugging:**
- When did accuracy drop?
- What changed between experiments?
- Trace issues to code/data changes

**5. Reporting:**
- Show stakeholders progress
- Justify model choices with data
- Create paper/blog post with full details

**Best Practices:**

**1. Track Everything Automatically:**
- Don't manually log—automate
- Use experiment tracking library
- Track more than you think you need

**2. Name Experiments Descriptively:**
- Bad: "experiment_42"
- Good: "lstm-fraud-detection-dropout-0.3"

**3. Tag and Organize:**
- Use tags: "production-candidate", "baseline", "ablation-study"
- Group related experiments
- Archive old experiments

**4. Version Data:**
- Don't just track "data.csv"
- Track "data_v3_2024-01-15.csv"
- Include data hash/checksum

**5. Document:**
- Add notes/descriptions
- Explain unusual results
- Link to relevant issues/tickets

**Example: Comparing Experiments**

```
| Experiment | LR    | Batch | Epochs | Val Acc | Notes               |
|------------|-------|-------|--------|---------|---------------------|
| baseline   | 0.01  | 64    | 20     | 85.2%   | Simple linear model |
| deep-v1    | 0.001 | 32    | 50     | 88.5%   | 3-layer MLP         |
| deep-v2    | 0.001 | 32    | 50     | 89.1%   | + dropout 0.3       |
| lstm-v1    | 0.001 | 32    | 50     | 91.5%   | LSTM architecture   |
| lstm-v2    | 0.0005| 16    | 100    | 92.3%   | Lower LR, more iter |
```

**Insight:** LSTM works best, lower LR + longer training helps

**Experiment Lifecycle:**

1. **Exploration:** Try many things, log everything
2. **Refinement:** Focus on promising approaches
3. **Optimization:** Hyperparameter tuning
4. **Validation:** Rigorous testing of best model
5. **Production:** Deploy tracked experiment

**Key Insight:** "I got 95% accuracy" is useless without knowing how. Experiment tracking turns ML from alchemy (random experimentation) into engineering (systematic iteration with full documentation). Future you will thank past you for tracking everything.

---
