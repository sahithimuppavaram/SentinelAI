
# SentinelAI — Data Schema

## 1. Purpose

Define the structure of human and AI-agent activity events used for insider-risk detection.

All initial data will be synthetic.

## 2. Raw Activity Event Schema

| Field | Type | Description |
|---|---|---|
| event_id | string | Unique event identifier |
| event_time | datetime | Timestamp in UTC |
| actor_id | string | Employee or AI-agent identifier |
| actor_type | string | human or ai_agent |
| owner_employee_id | string/null | Employee responsible for an AI agent |
| session_id | string | Activity session identifier |
| action | string | Type of activity performed |
| resource_id | string | Resource being accessed |
| sensitivity | string | Data sensitivity classification |
| destination_type | string | Destination category |
| size_mb | float | Data transferred in MB |
| source_type | string | endpoint, cloud, or agent_tool |

## 3. Allowed Values

actor_type:
- human
- ai_agent

action:
- file_read
- file_download
- file_upload
- tool_call

sensitivity:
- public
- internal
- confidential
- restricted

destination_type:
- internal
- approved_cloud
- unapproved_external

source_type:
- endpoint
- cloud
- agent_tool

## 4. Separate Ground-Truth Labels

Labels will be stored separately from raw activity.

| Field | Type | Description |
|---|---|---|
| event_id | string | References the activity event |
| risk_label | integer | 0 = routine, 1 = risk-worthy |
| scenario_type | string | Synthetic behavior scenario |
| label_source | string | How the label was created |
| label_available_at | datetime | When the label becomes available |

Labels are for training and evaluation, not inference inputs.

## 5. Validation Rules

- Every event_id must be unique.
- event_time must be a valid UTC timestamp.
- size_mb must be nonnegative.
- Categorical values must match allowed values.
- Every AI-agent event must have an owner_employee_id.
- Required fields must not be missing.
- Labels must not be included among model features.

## 6. Future Behavioral Features

Examples:
- Files downloaded in the previous hour
- Average daily transfer volume
- Deviation from an actor's historical baseline
- Activity outside usual working hours
- Percentage of transfers to unapproved destinations
- Number of sensitive resources accessed recently

Historical features must use only information available at the time of prediction.

## 7. Ground-Truth Limitations

Synthetic risk labels are not proof of malicious intent.

We will simulate routine and risk-worthy scenarios, introduce behavioral variation and noise, and evaluate on held-out scenarios.

Model performance on synthetic data must not be described as validated real-world detection performance.
