
# SentinelAI — Authorized Misuse Detection Design

## Objective

Improve the detection of potentially risky behavior
performed through legitimate, authorized access.

## Current Problem

V3 models detect permission violations more reliably
than authorized misuse.

Logistic Regression:
- Permission violation recall: 100%
- Authorized misuse recall: 16.7%

XGBoost:
- Permission violation recall: 41.7%
- Authorized misuse recall: 16.7%

These results come from a small synthetic test set.

## New Behavioral Signals

1. Recent file-access frequency
2. Number of sensitive resources accessed
3. Transfer volume within the last hour
4. Deviation from normal actor activity
5. New or uncommon destinations
6. Repeated tool calls by AI agents
7. Changes in session behavior

## New Risk Scenarios

### A. Authorized Bulk Exfiltration
A user with legitimate access performs an unusual
sequence of sensitive file downloads and uploads.

### B. Suspicious AI-Agent Activity
An authorized agent makes an unusually large number
of sensitive resource access requests.

### C. Unusual Access Sequence
A user accesses many previously unseen documents
within a short period.

## Hard-Negative Scenarios

1. Authorized bulk data export
2. Scheduled backup
3. Approved AI-agent batch processing
4. Legitimate high-volume analytics work

## Evaluation Requirements

- Compare precision and recall by scenario.
- Measure false-positive rates for hard negatives.
- Evaluate incident-level recall.
- Respect chronological ordering.
- Prevent incident leakage between partitions.
- Use labels available at each training cutoff.
- Compare models under the same alert budget.

## Limitations

All events and outcomes are synthetic.

Success on this simulator does not establish
real-world insider-risk detection accuracy.
