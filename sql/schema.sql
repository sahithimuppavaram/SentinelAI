
CREATE TABLE IF NOT EXISTS events (
    event_id VARCHAR(50) PRIMARY KEY,
    actor_id VARCHAR(50) NOT NULL,
    actor_type VARCHAR(20) NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    action VARCHAR(50) NOT NULL,
    resource_id VARCHAR(100),
    destination_type VARCHAR(50),
    size_mb DOUBLE PRECISION NOT NULL
        CHECK (size_mb >= 0),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    event_id VARCHAR(50) NOT NULL
        REFERENCES events(event_id),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    risk_score DOUBLE PRECISION NOT NULL
        CHECK (risk_score BETWEEN 0 AND 1),
    threshold DOUBLE PRECISION NOT NULL
        CHECK (threshold BETWEEN 0 AND 1),
    is_alert BOOLEAN NOT NULL,
    predicted_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id BIGSERIAL PRIMARY KEY,
    prediction_id BIGINT NOT NULL UNIQUE
        REFERENCES predictions(prediction_id),
    status VARCHAR(30) NOT NULL DEFAULT 'open'
        CHECK (
            status IN (
                'open',
                'investigating',
                'resolved',
                'dismissed'
            )
        ),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_events_actor_time
ON events(actor_id, event_time);

CREATE INDEX IF NOT EXISTS idx_predictions_event
ON predictions(event_id);

CREATE INDEX IF NOT EXISTS idx_alerts_status
ON alerts(status);
