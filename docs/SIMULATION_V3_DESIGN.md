
# SentinelAI — V3 Simulation Design

## Objective
Develop a context-aware synthetic activity simulator for
machine learning-based insider-risk detection.

## 1. Actor Profiles

Simulate 100 employees across three roles:

- Software Engineer
- Data Analyst
- Finance Analyst

Each employee has:
- Unique employee ID
- Department and role
- Primary timezone
- Normal working hours
- Typical daily activity volume
- Typical file-transfer distribution
- Approved destination categories
- Resource access permissions

Simulate 20 AI agents linked to employee owners.

Each agent has:
- Unique agent ID
- Owner employee ID
- Approved tool set
- Permitted resource categories
- Typical activity frequency

## 2. Routine Activity

Generate chronological events based on actor profiles.

Events include:
- File reads
- File downloads
- File uploads
- Agent tool calls

Routine events should vary by:
- Employee role
- Time of day
- Day of week
- Session
- Resource sensitivity
- Destination
- Historical behavior

## 3. Legitimate Unusual Activity

Include:
- Authorized bulk exports
- Late-night maintenance
- Large approved-cloud transfers
- Temporary changes in workloads
- Approved AI-agent batch processing

These activities are not inherently risk-worthy.

## 4. Risk-Worthy Scenarios

Scenario A: Sensitive Data Exfiltration
- Unusual restricted-file access
- Multiple downloads
- External transfer

Scenario B: Suspicious Account Activity
- Off-hours activity
- Access outside normal resource patterns
- Unusual destination behavior

Scenario C: AI-Agent Permission Violation
- Tool invocation beyond approved scope
- Sensitive resource access
- Suspicious data movement

Risk-worthy scenarios can include small transfers.

## 5. Event Schema

Retain the V2 event fields.

Add:
- incident_id
- workflow_id
- resource_category
- tool_name
- authorization_status
- actor_timezone

Keep incident outcomes and scenario labels in separate
ground-truth files.

## 6. Output Files

data/v3/actors.csv
data/v3/events.csv
data/v3/event_labels.csv
data/v3/incidents.csv
data/v3/simulation_metadata.json

## 7. Validation

- Unique event IDs
- Correct actor ownership
- Valid event timestamps
- Consistent session and incident relationships
- Valid authorization states
- No labels in model feature inputs
- No future events in historical feature calculations

## 8. Evaluation Requirements

- Chronological train/validation/test split
- No leakage across related incidents
- Labels available only after their simulated review time
- Independent held-out incident scenarios
- Precision, recall and Average Precision
- False alerts per 1,000 events
- Incident-level recall
- Recall at a fixed analyst alert budget
- Confidence intervals where appropriate

## 9. Limitations

Synthetic events cannot establish real-world detection performance.

Risk-worthy events represent simulated activities requiring
analyst review, not confirmed malicious intent.

Document assumptions and avoid using hidden simulator
metadata as prediction inputs.
