
import os

import pandas as pd
import psycopg
import streamlit as st

from psycopg.rows import dict_row

from src.investigation_agent import investigate
from src.claude_investigation import generate_claude_summary


st.set_page_config(
    page_title="SentinelAI",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ SentinelAI")
st.caption(
    "Experimental insider-risk monitoring dashboard "
    "| Synthetic activity data"
)


def get_connection():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not configured."
        )

    return psycopg.connect(
        database_url,
        row_factory=dict_row,
    )


def load_alerts():
    query = """
        SELECT
            a.alert_id,
            a.status,
            a.created_at,
            e.event_id,
            e.actor_id,
            e.actor_type,
            e.event_time,
            e.action,
            e.resource_id,
            e.size_mb,
            p.risk_score,
            p.threshold,
            p.model_name
        FROM alerts a
        JOIN predictions p
            ON a.prediction_id = p.prediction_id
        JOIN events e
            ON p.event_id = e.event_id
        ORDER BY a.created_at DESC, a.alert_id DESC
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            return pd.DataFrame(cursor.fetchall())


def load_metrics():
    query = """
        SELECT
            (SELECT COUNT(*) FROM events)
                AS total_events,
            (SELECT COUNT(*) FROM predictions)
                AS total_predictions,
            (SELECT COUNT(*) FROM alerts)
                AS total_alerts,
            (SELECT COUNT(*) FROM alerts
             WHERE status = 'open')
                AS open_alerts
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchone()


def load_recent_events():
    query = """
        SELECT
            e.event_id,
            e.actor_id,
            e.actor_type,
            e.event_time,
            e.action,
            e.size_mb,
            p.risk_score,
            p.is_alert
        FROM events e
        LEFT JOIN LATERAL (
            SELECT risk_score, is_alert
            FROM predictions
            WHERE event_id = e.event_id
            ORDER BY predicted_at DESC,
                     prediction_id DESC
            LIMIT 1
        ) p ON TRUE
        ORDER BY e.created_at DESC, e.event_id DESC
        LIMIT 100
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            return pd.DataFrame(cursor.fetchall())


if "investigation_results" not in st.session_state:
    st.session_state.investigation_results = {}

if st.button("Refresh data"):
    st.rerun()


try:
    metrics = load_metrics()
    alerts = load_alerts()
    events = load_recent_events()

except (psycopg.Error, RuntimeError) as error:
    st.error(f"Database connection failed: {error}")
    st.stop()


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Events",
    metrics["total_events"],
)

col2.metric(
    "ML Predictions",
    metrics["total_predictions"],
)

col3.metric(
    "Total Alerts",
    metrics["total_alerts"],
)

col4.metric(
    "Open Alerts",
    metrics["open_alerts"],
)


st.divider()

st.subheader("Security Alerts")

if alerts.empty:
    st.info("No alerts have been generated yet.")

else:
    display_alerts = alerts.copy()

    display_alerts["risk_score"] = (
        display_alerts["risk_score"].round(4)
    )

    st.dataframe(
        display_alerts[
            [
                "alert_id",
                "actor_id",
                "event_id",
                "event_time",
                "action",
                "risk_score",
                "status",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Investigate an Alert")

    alert_ids = [
        int(value)
        for value in alerts["alert_id"].tolist()
    ]

    selected_id = st.selectbox(
        "Select alert ID",
        alert_ids,
    )

    selected = alerts.loc[
        alerts["alert_id"] == selected_id
    ].iloc[0]

    left, right = st.columns(2)

    with left:
        st.write("**Actor:**", selected["actor_id"])
        st.write("**Event:**", selected["event_id"])
        st.write("**Action:**", selected["action"])

    with right:
        st.metric(
            "ML Risk Score",
            f"{selected['risk_score']:.4f}",
        )
        st.write(
            "**Alert Threshold:**",
            round(float(selected["threshold"]), 4),
        )
        st.write("**Status:**", selected["status"])

    st.info(
        "A high ML score is not proof of malicious "
        "intent. All findings require human review."
    )

    st.markdown("#### AI Investigation Assistant")

    if not os.getenv("ANTHROPIC_API_KEY"):
        st.warning(
            "Claude API key not configured. "
            "Start Streamlit with ANTHROPIC_API_KEY "
            "set in the terminal."
        )

    if st.button(
        "Generate AI Investigation",
        type="primary",
        disabled=not bool(
            os.getenv("ANTHROPIC_API_KEY")
        ),
    ):
        try:
            with st.spinner(
                "Collecting evidence and asking Claude..."
            ):
                report = investigate(selected_id)

                summary, usage = (
                    generate_claude_summary(report)
                )

            st.session_state.investigation_results[
                selected_id
            ] = {
                "report": report,
                "summary": summary,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            }

        except Exception as error:
            st.error(
                "Investigation request failed. "
                "Check that PostgreSQL and your Claude "
                "API credentials are available."
            )
            st.caption(
                f"Error type: {type(error).__name__}"
            )

    saved_result = (
        st.session_state.investigation_results.get(
            selected_id
        )
    )

    if saved_result:
        st.success(
            f"Investigation completed for Alert "
            f"{selected_id}"
        )

        st.markdown(
            saved_result["summary"]
        )

        with st.expander(
            "View underlying database evidence"
        ):
            st.json(saved_result["report"])

        st.caption(
            "Claude API usage: "
            f"{saved_result['input_tokens']} input "
            f"tokens, "
            f"{saved_result['output_tokens']} "
            "output tokens."
        )

        st.warning(
            "This is an AI-generated summary of "
            "synthetic, incomplete records. "
            "Repeated test events may be present. "
            "Verify all claims before acting."
        )


st.divider()

st.subheader("Recent Activity")

if events.empty:
    st.info("No events recorded.")

else:
    st.dataframe(
        events,
        use_container_width=True,
        hide_index=True,
    )


st.subheader("Risk Score Distribution")

if events["risk_score"].notna().any():
    scores = events["risk_score"].dropna()

    labels = [
        f"{i / 10:.1f}–{(i + 1) / 10:.1f}"
        for i in range(10)
    ]

    score_ranges = pd.cut(
        scores,
        bins=[i / 10 for i in range(11)],
        labels=labels,
        include_lowest=True,
    )

    distribution = (
        score_ranges.value_counts()
        .sort_index()
        .rename_axis("Risk score range")
        .reset_index(name="Number of events")
    )

    st.bar_chart(
        distribution,
        x="Risk score range",
        y="Number of events",
    )



st.divider()

st.caption(
    "SentinelAI | Synthetic data | "
    "Human-in-the-loop security research prototype"
)
