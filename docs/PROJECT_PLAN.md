# SentinelAI — Engineering blueprint (v1.0)

**Purpose:** An independently developed, privacy-safe insider-risk ML engineering portfolio project. Inspired by problems described in Mimecast's Machine Learning Engineer II / Incydr job description, but **not** affiliated with Mimecast and **not** a reproduction of its internal architecture.

**North star:** From synthetic activity telemetry to analyst-reviewed, evidence-based risk alerts in a reproducible, testable, observable ML system.

## 1. Product charter

### Problem
Security analysts need to triage a small number of potentially risky data-movement events among a large stream of legitimate employee and AI-agent activity. Low recall misses real threats; poor precision causes alert fatigue. We want to detect events warranting review, not infer a person's intent.

### Primary user
Security analyst. Secondary user: ML engineer operating and improving detection models. Synthetic tenant employees and synthetic automated agents are the modeled event sources.

### End-to-end product journey
1. Consume an event (file upload/download, external share, access, or an agent tool call).
2. Validate the event and enrich it with historical user/role/destination features using event time.
3. Run a calibrated supervised detector and/or anomaly detector.
4. Apply an alert policy using thresholds and operational capacity.
5. Store immutable evidence references, prediction version, and explanation metadata.
6. Present alerts in the analyst dashboard; allow review and disposition.
7. Run a **read-only, evidence-grounded** LLM investigation when requested.
8. Track delayed analyst labels and monitor model + operational performance.

### Scope guardrails
- All locally generated identities, activities, and documents are fake. Do not collect actual employee data.
- An ML risk score is not proof of malicious intent; no automatic punitive action.
- LLM investigations never trigger actions on accounts, users, or files. All tool calls are authorized and audited.
- Do not mistake results from synthetic data for real-world effectiveness.
- Add components only when they solve a documented requirement.

## 2. Technology choices (purposeful, not maximalist)

| Layer | Primary choice | Why / what it teaches |
|---|---|---|
| Implementation language | Python | Core JD language; type hints, tests, concurrency |
| Dataframes / numerics | Pandas, NumPy; PyArrow/Parquet as data grows | Cleaning, transforms, reproducible partitioned data |
| ML baseline / classical | scikit-learn, XGBoost | Feature pipelines, classification, calibration, imbalance |
| Anomaly / temporal | Isolation Forest; forecasting baseline | Unsupervised detection and time-series reasoning |
| Deep learning | PyTorch (only if comparative experiment supports it) | Mini-batching, loss, Adam, training/evaluation |
| Text feature extraction | spaCy, optionally Hugging Face for a justified document task | NLP on synthetic text; avoid bolting on arbitrary OCR |
| Local event stream | Kafka | Producer/consumer, ordering, offsets, replay |
| Operational store | PostgreSQL | Alerts, feedback, users, audit trail; SQL skills |
| Object storage | Local Parquet, then S3 | Training history and artifacts |
| Inference | FastAPI | Model serving, validation, concurrency, performance |
| ML lifecycle | MLflow | Tracking, artifacts, versioning |
| Investigation AI | LangGraph + Bedrock (local stub first) | Tool calling, state, retrieval, evaluations and guardrails |
| Observability | Structured logs, Prometheus, Grafana | p95 latency, throughput, error rate, feature health |
| Packaging / CI | Docker, pytest, GitHub Actions | Reproducibility, tests, automation |
| Cloud | ECS + S3 + select Kinesis/Lambda/SageMaker services + Terraform | A production-shaped deployment, not every AWS SKU |
| UI | Streamlit | Human-readable analyst flow |

**Explicit non-goals:** Java/C++/Kotlin just to check boxes, duplicate RAG frameworks, Redshift without an analytics need, automatic response/remediation, proprietary Mimecast telemetry.

## 3. Data contract, simulation and labeling

### Minimum canonical event schema
- `event_id` (unique string)
- `tenant_id` (synthetic organization)
- `actor_id` (synthetic user or agent ID)
- `actor_type` (`human` or `ai_agent`)
- `event_time` (timezone-aware UTC timestamp)
- `action` (download, upload, share, access, agent_tool_call)
- `bytes_transferred` (nonnegative)
- `file_sensitivity` (public, internal, confidential, restricted)
- `destination_type` (approved, personal_cloud, external_email, removable_media, ai_tool)
- `resource_id` (synthetic file or resource ID)
- `device_id` (synthetic device)
- `request_id` (for replay/idempotency)

**Separate review/label table** (not supplied to inference) includes `event_id`, `review_label` (review-worthy / not review-worthy / unknown), `source` (simulated scenario vs analyst), `label_time`, `scenario_id`, `review_notes`. Never put `scenario_id`, future data, or label proxies into the feature vector.

### Simulation requirements
- Role-dependent baselines, working hours, approved destinations, repeated users, seasonality.
- Rare risky scenarios with overlap and ambiguous legitimate behavior, not trivial if/else labels.
- Unseen attack scenarios, user/time/tenant holdouts, injected drift, noisy/incomplete labels.
- Dataset starts small; progress to **>2 million examples actually ingested, processed and benchmarked**, not merely generated.
- Try the publicly documented Carnegie Mellon SEI CERT synthetic insider-threat test dataset as an *optional external benchmark*, after checking its schema and access/usage terms.

## 4. Modeling, labels and evaluation protocol

**Prediction target:** Probability that an activity/event warrants security review using features observable by event time. The threshold drives an alert; the score is not an allegation against the actor.

**Model comparisons:** Rule-based baseline → logistic regression → XGBoost → Isolation Forest → optional PyTorch temporal model. Separate temporal forecasting experiment predicts baseline volumes; it is not mixed into a classification score without validation.

**Splits:** Time-ordered training/validation/test plus grouped holdouts for selected users/tenants; never fit imputers, encoders or scalers on the test set. Tune threshold on validation only and freeze test before final evaluation.

**Key metrics:** PR-AUC (average precision), precision/recall at an explicit daily alert budget, false-positive alerts per 1,000 events, recall by scenario, calibration/Brier score, confidence intervals. Track p50/p95 inference latency and events/second separately from model quality. Report class prevalence and limitations.

**Scientific practice:** Hypothesis → experimental protocol → baseline → results → confidence interval/uncertainty → decision. Ensure synthetic generator doesn't simply hard-code the same feature-to-label relationship as the model learns.

## 5. Architecture

**Local pipeline**

```
Synthetic events / optional CERT benchmark
       ├── offline partitions (Parquet) ──> training + validation ──> MLflow
       └── Kafka stream ──> consumer / event-time features ──> FastAPI model scoring
                                             └──> PostgreSQL alerts / feedback / audit
                                                              ├──> Streamlit analyst UI
                                                              └──> LangGraph read-only investigation
Monitoring: logs, metrics, feature/drift reports, delayed analyst labels
```

**Cloud mapping when justified**: Local Parquet → S3; Kafka → Kinesis; local training → SageMaker training job if needed; FastAPI container → ECS; PostgreSQL → RDS only if managed storage needed; investigation LLM → Bedrock; Terraform for infrastructure. Glue/Athena useful if a data catalog or ad hoc querying is needed. Redshift is not a default component.

**Critical separation:** Event scoring does not wait for the LLM. Agent investigations are initiated after alert creation and are bounded by evidence permissions. Training and inference share tested feature definitions.

## 6. Milestone-by-milestone implementation

### M0 — Product and system contract
**Build:** project charter, actor flows, label definition, measurable success criteria, event schema, architecture decision log (ADR-001).
**Done when:** a reviewer can explain what an alert means, what data goes in, and how success is assessed.
**Learn:** problem framing, precision vs recall cost, business requirements.

### M1 — Data foundation and defensible baseline
**Build:** schema-validated event simulator; partitioned Parquet; EDA notebook; separate label table; chronological split; rule-based and logistic regression baselines; leakage checks.
**Done when:** deterministic generation works, data integrity tests pass, and honest baseline metrics reproduce from a fresh clone.
**Learn:** Pandas, NumPy, SQL, feature engineering, statistical distributions, leakage.

### M2 — Strong ML experiments
**Build:** role/actor temporal features, XGBoost, Isolation Forest, imbalance handling, calibrated thresholds, SHAP; statistical comparisons and bootstrap intervals. Optional forecasting and PyTorch sequence experiment after baselines.
**Done when:** model comparisons use the same frozen holdout and report explicit operational cost of false alarms.
**Learn:** probability, gradients, optimization, hypothesis testing, classification, time series and deep learning.

### M3 — Live detection service
**Build:** Kafka producer + consumer, event-time/late-event handling, idempotent event writes, FastAPI scoring API, PostgreSQL tables, alert policy, integration tests.
**Done when:** a deterministic replay produces the same alert set with no duplicate records; latency and throughput are measured with declared hardware/load.
**Learn:** streaming, REST, SQL, concurrency, architecture choices.

### M4 — MLOps and system reliability
**Build:** Docker Compose, MLflow experiments/models, pytest, GitHub Actions, structured logs, Prometheus/Grafana, drift and data quality checks, safe model rollback.
**Done when:** build+tests pass on CI and the operator can identify a drift or latency regression.
**Learn:** reproducibility, CI/CD, operations, model monitoring.

### M5 — Agentic investigation and privacy
**Build:** LangGraph read-only investigator; authorized SQL/evidence tools; synthetic document metadata lookup; evidence citations; audit logs; prompt injection and unsafe-tool-call tests. Bedrock optional behind a provider adapter.
**Done when:** agent answers grounded incident questions without fabricating evidence or invoking unauthorized operations.
**Learn:** RAG, LangGraph, tool calling, governance, LLM evaluation and prompt security.

### M6 — Cloud deployment and scale
**Build:** cost-controlled AWS ECS inference deployment, S3 artifacts, optional Kinesis stream and SageMaker job, Terraform, IAM least privilege, smoke tests, cleanup commands. Expand partitioned processing to 2M+ events with profiling.
**Done when:** deploy/redeploy/teardown all work, costs and quotas have been inspected, and performance results are reproducible.
**Learn:** core AWS services, IaC, data scale, cloud tradeoffs.

### M7 — UX, demo and interview defense
**Build:** Streamlit alert triage UI, human/agent alert cases, investigation demo, polished README, architecture diagram, model card, evaluation report, 5-minute and 15-minute presentations.
**Done when:** fresh-clone demo runs following README; no unaudited performance claims; can explain design and alternatives aloud.
**Learn:** stakeholder communication, ownership, tradeoffs, technical storytelling.

## 7. Minimum viable demo vs extensions

**Non-negotiable demo:** trusted synthetic data → honest baseline/XGBoost comparison → streaming or replay → FastAPI → stored alert → reviewed disposition → evidence-grounded agent summary → observable metrics.

**Extensions only after MVP works:** PyTorch temporal sequence model, large-scale SageMaker training, optional CERT benchmark, synthetic-document NLP/OCR, broader AWS services, dashboard polish.

## 8. Engineering conventions

- One Git repository with `docs/`, `src/`, `tests/`, `configs/`, `infra/`, and `notebooks/`.
- Keep generated data, secrets and model binaries out of Git unless small and intentionally published.
- Use configuration/env vars; never commit cloud credentials.
- Preserve experiment metadata, random seeds, dataset splits, versions and hardware benchmarks.
- Add one ADR for each major decision (algorithm, database, transport, deployment).
- Every milestone gets a written: "problem / approach / design tradeoff / evidence / limitation / next step" summary and 5 mock interview questions.

## 9. First work session — only M0.1

1. Create a clean project folder/repository named `sentinel-ai` (keep the old ZIP as a baseline experiment, not the final product).
2. Add this blueprint as `docs/PROJECT_PLAN.md`.
3. Create `docs/PRODUCT_SPEC.md` with the 4 headings `Problem`, `Who uses it`, `What is an alert?`, `What is a successful demo?`.
4. Draft the canonical event schema and label definition **before** writing a new generator.
5. Review the schema and metric contract; then begin M1 data generation.

**Exit check:** You can answer in one minute: "What event are we predicting, what information is available at prediction time, and how do we know whether the model helped the analyst?"

## 10. Reference materials

- Mimecast ML Engineer II / Incydr job description as supplied by the user (primary scope source).
- CMU Software Engineering Institute, *Insider Threat Test Dataset*: https://www.sei.cmu.edu/library/insider-threat-test-dataset/ (optional synthetic benchmark; not yet loaded into this project).
- AWS Free Tier and pricing: https://docs.aws.amazon.com/us_en/awsaccountbilling/latest/aboutv2/free-tier.html (check eligibility/region/services before cloud usage).
