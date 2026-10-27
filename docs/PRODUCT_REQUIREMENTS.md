
# SentinelAI — Product Requirements

## 1. Project Overview
SentinelAI is an ML-powered insider-risk detection and AI investigation platform.

It analyzes simulated employee and AI-agent activity to identify potentially risky behavior and assist security analysts with investigations.

## 2. Problem Statement
Organizations generate large volumes of activity events.

Traditional rule-based systems can produce excessive alerts and miss behavior that is unusual for a specific user.

My goal is to evaluate activity in context, prioritize potentially risky events, and support evidence-based human investigation.

## 3. Target Users
- Security analysts
- Insider-risk investigation teams
- Security operations managers

## 4. Core Functional Requirements

FR-01: Ingest employee and AI-agent activity events.

FR-02: Clean, validate, and transform raw data.

FR-03: Generate behavioral features from historical activity.

FR-04: Train ML models for classification and anomaly detection.

FR-05: Produce calibrated risk probabilities and anomaly scores.

FR-06: Generate alerts based on configurable thresholds.

FR-07: Explain model predictions using feature attribution.

FR-08: Provide an AI assistant that retrieves authorized evidence and summarizes incidents.

FR-09: Allow analysts to review alerts and record feedback.

FR-10: Monitor model quality, data drift, latency, and throughput.

## 5. Non-Functional Requirements
- Scalable processing of 2M+ synthetic events
- Low-latency inference
- Reliable event processing and duplicate prevention
- Reproducible training experiments
- Secure storage and access controls
- Auditable model and investigation decisions

## 6. Model Evaluation
We will measure:
- Precision
- Recall
- F1-score
- PR-AUC
- False alerts per 1,000 events
- Probability calibration
- Inference latency and throughput

Performance targets will be chosen after establishing a baseline.

## 7. Project Boundaries
- Use synthetic or properly licensed public datasets.
- Do not process real employee personal information.
- Predictions indicate activity requiring review, not confirmed malicious intent.
- AI agents cannot autonomously punish users or take destructive actions.
- Keep ground-truth labels separate from inference features.
