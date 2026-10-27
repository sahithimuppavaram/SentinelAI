
# SentinelAI — V2 ML Experiment

## Objective
Compare Logistic Regression and XGBoost for detecting
risk-worthy activity in synthetic insider-risk events.

## Dataset
- Total events: 10,000
- Routine: 9,404
- Legitimate unusual: 394
- Risk-worthy: 202
- Risk prevalence: 2.02%

## Evaluation Design
- Chronological split: 60% train, 20% validation, 20% test
- Validation used to select F1-maximizing thresholds
- Test set contains 2,000 events, including 42 risky events

## Logistic Regression
- Threshold: 0.95
- Precision: 0.3250
- Recall: 0.3095
- F1: 0.3171
- Average Precision: 0.3153
- ROC-AUC: 0.8814
- False positives: 27
- False negatives: 29

## XGBoost
- Threshold: 0.28
- Precision: 0.3784
- Recall: 0.3333
- F1: 0.3544
- Average Precision: 0.2688
- ROC-AUC: 0.8736
- False positives: 23
- False negatives: 28

## Error Analysis
XGBoost generated 23 false positives:
- 12 routine events
- 11 legitimate unusual events

XGBoost missed 28 of 42 risk-worthy test events.

Both models missed 26 of the same risky events.

## Findings
1. XGBoost achieved higher F1 at its selected threshold.
2. Logistic Regression achieved higher Average Precision.
3. Legitimate unusual events are a major source of false positives.
4. Low-volume risky events are often missed.
5. More contextual information is needed for improved detection.

## Limitations
- All data is synthetic.
- Scenarios are generated mostly independently.
- Some labels may not be predictable from available features.
- User history is simulated, not real operational behavior.
- Same-timestamp unit test remains to be verified.
- Label availability at training cutoffs is not yet enforced.
- Performance has not been evaluated across independent
  scenario generators or real-world datasets.
- No deployment, latency, throughput, or drift benchmarks yet.

## Next Experiment
V3 will introduce realistic actor profiles, authorized workflows,
resource histories, AI-agent permissions, and multi-event scenarios.

We will evaluate model quality under distribution shift and
compare predictions at a fixed analyst alert budget.
