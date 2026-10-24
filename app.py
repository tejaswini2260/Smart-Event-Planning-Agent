"""
app.py - Streamlit interface for the Smart Event Planning Agent.
Run with:  streamlit run app.py
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

import database as db
from agents import (EMBED_MODEL, EVENT_TYPES, FOOD_PREFERENCES, LLM_MODEL, ORCHESTRATION_RULES,
                    EventPlanningOrchestrator)
from utils import budget_utils as bu

st.set_page_config(page_title="Smart Event Planning Agent", page_icon="🌸", layout="wide")

st.markdown(
    """
    <style>
    h1, h2, h3 { color: #5b4b8a; }
    [data-testid="stSidebar"] { border-right: 2px solid #e6dcff; }
    div[data-testid="stMetric"] { background: #ffffff; border: 1px solid #e6dcff;
        border-radius: 12px; padding: 10px 14px; }
    .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
        background: #b9a7f5; color: #2d2550; border: none; border-radius: 10px; font-weight: 600; }
    .stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
        background: #9b7fd4; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = ["Plan an Event", "Saved Events", "Budget Overview", "Preparation Checklist",
         "Agent Execution Monitor", "About Project"]
STATUS_ICON = {"SUCCESS": "✅", "WARNING": "⚠️", "FAILED": "❌", "SKIPPED": "⏭️"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def init_state() -> None:
    for key, default in {"orchestrator": None, "result": None, "event_id": None,
                         "ollama_status": None, "pending_delete": None}.items():
        st.session_state.setdefault(key, default)


def get_orchestrator() -> EventPlanningOrchestrator:
    if st.session_state.orchestrator is None:
        st.session_state.orchestrator = EventPlanningOrchestrator()
    return st.session_state.orchestrator


def safe_db(func, *args, default=None, **kwargs):
    """Run a database function and show a friendly error instead of a traceback."""
    try:
        return func(*args, **kwargs)
    except db.DatabaseError as exc:
        st.error(f"Database problem: {exc}")
        return default


def event_label(event: Dict[str, Any]) -> str:
    return f"#{event['id']} - {event['event_type']} on {event['event_date']} ({event['location']})"


def budget_dataframe(analysis: Dict[str, Any]) -> pd.DataFrame:
    rows = [{"Category": c, "Share (%)": analysis["percentages"][c], "Amount (INR)": a,
             "Amount": bu.format_inr(a)} for c, a in analysis["allocations"].items()]
    return pd.DataFrame(rows)


def build_markdown_export(result: Dict[str, Any]) -> str:
    req = result.get("requirements") or {}
    lines = [f"# Event Plan: {req.get('event_type', 'Event')} ({req.get('event_date', '')})", "",
             f"**Validation status:** {result.get('validation_status')}", "",
             "## Event details"]
    for key, label in [("event_type", "Event type"), ("event_date", "Date"), ("location", "Location"),
                       ("guest_count", "Guests"), ("venue_capacity", "Venue capacity"),
                       ("theme", "Theme"), ("food_preference", "Food"),
                       ("special_requirements", "Special requirements")]:
        lines.append(f"- {label}: {req.get(key, '')}")
    if req.get("total_budget"):
        lines.append(f"- Total budget: {bu.format_inr(req['total_budget'])}")
    lines += ["", "## Event plan", result.get("plan_text", "")]
    budget = result.get("budget")
    if budget:
        lines += ["", "## Budget breakdown (illustrative estimates)", "| Category | Amount |", "|---|---|"]
        lines += [f"| {c} | {bu.format_inr(a)} |" for c, a in budget["allocations"].items()]
    schedule = result.get("schedule") or {}
    if schedule.get("timeline"):
        lines += ["", "## Event schedule", "| Time | Activity | Notes |", "|---|---|---|"]
        lines += [f"| {r['Time']} | {r['Activity']} | {r['Notes']} |" for r in schedule["timeline"]]
    if schedule.get("preparation_tasks"):
        lines += ["", "## Preparation checklist"]
        lines += [f"- [ ] {t['due_date']}: {t['task']}" for t in schedule["preparation_tasks"]]
    qa = result.get("qa")
    if qa:
        lines += ["", f"## Quality assurance: {qa.get('result')}"]
        lines += [f"- {'PASS' if c['passed'] else 'FAIL'}: {c['check']} - {c['detail']}" for c in qa.get("checks", [])]
    return "\n".join(lines)


def render_banner(result: Dict[str, Any]) -> None:
    status = result.get("validation_status")
    if status == "VALIDATED":
        st.success("✅ VALIDATED - all critical checks passed (capacity, budget maths and plan checks).")
    elif status == "NOT_VALIDATED":
        st.warning("⚠️ NOT FULLY VALIDATED - a critical constraint failed. Read the issues below before using this plan.")
    else:
        st.error("❌ INVALID INPUT - no plan was generated. Please fix the problems below.")


def render_trace_table(result: Dict[str, Any]) -> None:
    trace = result.get("trace") or []
    if not trace:
        st.info("No execution trace is available for this plan.")
        return
    df = pd.DataFrame([{
        "Step": t["step"], "Agent": t["agent"], "Action": t["action"],
        "Status": f"{STATUS_ICON.get(t['status'], '')} {t['status']}",
        "Time (ms)": t["duration_ms"], "Result": t["summary"]} for t in trace])
    st.dataframe(df, hide_index=True)


def render_result(result: Dict[str, Any], allow_save: bool = True) -> None:
    render_banner(result)
    for e in result.get("errors", []):
        st.error(e)
    for w in result.get("warnings", []):
        st.warning(w)
    if result.get("validation_status") == "INVALID_INPUT":
        with st.expander("Agent execution trace", expanded=True):
            render_trace_table(result)
        return

    tabs = st.tabs(["📝 Event Plan", "💰 Budget", "🗓️ Schedule", "📚 Knowledge Sources",
                    "🤖 Agent Trace", "✅ QA Report"])
    req = result.get("requirements") or {}

    with tabs[0]:
        if result.get("plan_source") == "fallback":
            st.info("The language model was unavailable, so a template-based fallback plan was used.")
        st.markdown(result.get("plan_text", ""))
        c1, c2 = st.columns(2)
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", f"event_plan_{req.get('event_type', 'event')}_{req.get('event_date', '')}")
        c1.download_button("⬇️ Download plan (Markdown)", data=build_markdown_export(result),
                           file_name=f"{safe_name}.md", mime="text/markdown")
        if allow_save:
            if st.session_state.event_id:
                c2.success(f"Saved as event #{st.session_state.event_id}")
            elif c2.button("💾 Save plan to database"):
                qa_result = (result.get("qa") or {}).get("result", "N/A")
                new_id = safe_db(db.save_event, req, result["plan_text"], result["validation_status"], qa_result, result)
                if new_id:
                    safe_db(db.save_checklist, new_id, (result.get("schedule") or {}).get("preparation_tasks", []))
                    st.session_state.event_id = new_id
                    st.rerun()

    with tabs[1]:
        budget = result.get("budget")
        if budget:
            m1, m2, m3 = st.columns(3)
            m1.metric("Total budget", bu.format_inr(budget["total_budget"]))
            m2.metric("Catering estimate needed", bu.format_inr(budget["catering_required_estimate"]))
            m3.metric("Catering shortfall", bu.format_inr(budget["catering_shortfall"]))
            df = budget_dataframe(budget)
            st.dataframe(df[["Category", "Share (%)", "Amount"]], hide_index=True)
            st.bar_chart(df.set_index("Category")["Amount (INR)"])
            for s in budget.get("suggestions", []):
                st.warning(s)
            st.caption(budget.get("note", ""))

    with tabs[2]:
        schedule = result.get("schedule") or {}
        if schedule.get("timeline"):
            st.dataframe(pd.DataFrame(schedule["timeline"])[["Phase", "Time", "Activity", "Notes"]], hide_index=True)
        venue = result.get("venue") or {}
        if venue:
            st.subheader("Venue and logistics check")
            (st.success if venue.get("capacity_ok") else st.error)(
                f"{venue.get('guest_count')} guests vs capacity {venue.get('venue_capacity')}: "
                + ("capacity is sufficient." if venue.get("capacity_ok") else "CAPACITY EXCEEDED."))
            st.json(venue.get("logistics", {}))
            for s in venue.get("suggestions", []) + venue.get("notes", []):
                st.info(s)

    with tabs[3]:
        knowledge = result.get("knowledge") or {}
        st.caption(f"Retrieval method: {knowledge.get('method', 'n/a')}")
        if not knowledge.get("chunks"):
            st.info("No knowledge sections were retrieved.")
        for c in knowledge.get("chunks", []):
            with st.expander(f"{c['title']}  (score {c['score']})"):
                st.caption(f"Source: {c['source']}")
                st.write(c["text"])

    with tabs[4]:
        render_trace_table(result)

    with tabs[5]:
        render_qa(result.get("qa"))


def render_qa(qa: Optional[Dict[str, Any]]) -> None:
    if not qa:
        st.info("No QA report is available.")
        return
    (st.success if qa["result"] == "PASS" else st.error)(f"QA result: {qa['result']}")
    df = pd.DataFrame([{"Result": "✅ PASS" if c["passed"] else "❌ FAIL",
                        "Check": c["check"], "Critical": "Yes" if c["critical"] else "No",
                        "Explanation": c["detail"]} for c in qa.get("checks", [])])
    st.dataframe(df, hide_index=True)
    for r in qa.get("reasons", []):
        st.write("•", r)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> str:
    st.sidebar.title("🌸 Smart Event Planner")
    page = st.sidebar.radio("Navigate", PAGES)
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Check Ollama status") or st.session_state.ollama_status is None:
        st.session_state.ollama_status = get_orchestrator().client.check_status()
    status = st.session_state.ollama_status
    if status["running"] and not status["missing_models"]:
        st.sidebar.success("Ollama ready")
    elif status["running"]:
        st.sidebar.warning(status["message"])
    else:
        st.sidebar.error(status["message"])
    st.sidebar.caption(f"LLM: {LLM_MODEL}\n\nEmbeddings: {EMBED_MODEL}")
    return page


# --------------------------------------------------------------------------- #
# Page 1: Plan an Event
# --------------------------------------------------------------------------- #
def page_plan() -> None:
    st.title("🌸 Plan an Event")
    st.caption("Seven specialised agents validate, research, budget, schedule, write and check your event plan.")
    with st.form("event_form"):
        c1, c2 = st.columns(2)
        event_type = c1.selectbox("Event type", EVENT_TYPES)
        event_date = c2.date_input("Event date", value=date.today() + timedelta(days=45), min_value=date.today())
        location = c1.text_input("Location (city / area)", placeholder="e.g. Pune")
        guests = c2.number_input("Number of guests", min_value=1, max_value=5000, value=100, step=1)
        budget = c1.number_input("Total budget (INR)", min_value=0, value=500000, step=10000)
        capacity = c2.number_input("Venue capacity (maximum guests)", min_value=1, value=150, step=10)
        theme = c1.text_input("Event theme", placeholder="e.g. Royal pink and gold")
        food = c2.selectbox("Food preference", FOOD_PREFERENCES)
        special = st.text_area("Special requirements", placeholder="e.g. Outdoor seating, wheelchair access, evening start")
        submitted = st.form_submit_button("✨ Generate Event Plan")

    if submitted:
        raw = {"event_type": event_type, "event_date": event_date, "location": location,
               "guest_count": guests, "total_budget": budget, "venue_capacity": capacity,
               "theme": theme, "food_preference": food, "special_requirements": special}
        with st.spinner("The agents are working... the language model can take a minute or two on a normal PC."):
            st.session_state.result = get_orchestrator().run(raw)
        st.session_state.event_id = None

    if st.session_state.result:
        if st.session_state.event_id and not submitted:
            st.info(f"Showing saved event #{st.session_state.event_id}.")
        render_result(st.session_state.result)


# --------------------------------------------------------------------------- #
# Page 2: Saved Events
# --------------------------------------------------------------------------- #
def page_saved() -> None:
    st.title("📁 Saved Events")
    events = safe_db(db.get_all_events, default=[])
    if not events:
        st.info("No saved events yet. Generate a plan and click 'Save plan to database'.")
        return
    df = pd.DataFrame([{
        "ID": e["id"], "Type": e["event_type"], "Date": e["event_date"], "Location": e["location"],
        "Guests": e["guest_count"], "Budget": bu.format_inr(e["total_budget"]),
        "Status": e["validation_status"], "QA": e["qa_result"], "Saved at": e["created_at"]} for e in events])
    st.dataframe(df, hide_index=True)

    ids = [e["id"] for e in events]
    by_id = {e["id"]: e for e in events}
    selected = st.selectbox("Select an event", ids, format_func=lambda i: event_label(by_id[i]))
    event = safe_db(db.get_event, selected)
    if not event:
        st.warning("That event no longer exists.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Event date", event["event_date"])
    c2.metric("Guests", event["guest_count"])
    c3.metric("Budget", bu.format_inr(event["total_budget"]))
    c4.metric("Status", event["validation_status"])
    st.markdown(event["plan_text"])

    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("📂 Load into app"):
        loaded = event.get("result") or {
            "validation_status": event["validation_status"], "requirements": {
                k: event[k] for k in ("event_type", "event_date", "location", "guest_count", "total_budget",
                                      "venue_capacity", "theme", "food_preference", "special_requirements")},
            "plan_text": event["plan_text"], "trace": [], "errors": [], "warnings": []}
        st.session_state.result = loaded
        st.session_state.event_id = event["id"]
        st.success("Loaded. Open 'Plan an Event' to view it or 'Agent Execution Monitor' to inspect the agents.")
    if b2.button("🗑️ Delete event"):
        st.session_state.pending_delete = selected
    if st.session_state.pending_delete == selected:
        st.warning(f"Delete event #{selected} and its checklist permanently? This cannot be undone.")
        y, n, _ = st.columns([1, 1, 3])
        if y.button("Yes, delete"):
            if safe_db(db.delete_event, selected, default=False):
                if st.session_state.event_id == selected:
                    st.session_state.event_id = None
            st.session_state.pending_delete = None
            st.rerun()
        if n.button("Cancel"):
            st.session_state.pending_delete = None
            st.rerun()


# --------------------------------------------------------------------------- #
# Page 3: Budget Overview
# --------------------------------------------------------------------------- #
def page_budget() -> None:
    st.title("💰 Budget Overview")
    st.caption("Uses the same deterministic calculation as the Budget Planning Agent. All amounts are estimates.")
    current = (st.session_state.result or {}).get("requirements") or {}
    c1, c2 = st.columns(2)
    event_type = c1.selectbox("Event type", EVENT_TYPES,
                              index=EVENT_TYPES.index(current["event_type"]) if current.get("event_type") in EVENT_TYPES else 0)
    food = c2.selectbox("Food preference", FOOD_PREFERENCES,
                        index=FOOD_PREFERENCES.index(current["food_preference"]) if current.get("food_preference") in FOOD_PREFERENCES else 0)
    total = c1.number_input("Total budget (INR)", min_value=0, value=int(current.get("total_budget", 500000)), step=10000)
    guests = c2.number_input("Number of guests", min_value=1, value=int(current.get("guest_count", 100)), step=1)

    ok, amount, message = bu.validate_budget(total)
    if not ok:
        st.error(message)
        return
    analysis = bu.analyze_budget(amount, int(guests), event_type, food)
    m1, m2, m3 = st.columns(3)
    m1.metric("Total budget", bu.format_inr(amount))
    m2.metric("Budget per guest", bu.format_inr(analysis["per_guest_budget"]))
    m3.metric("Catering shortfall", bu.format_inr(analysis["catering_shortfall"]))
    if analysis["status"] == "OVERRUN":
        st.error("Budget overrun detected (estimate): catering needs "
                 f"{bu.format_inr(analysis['catering_required_estimate'])} but only "
                 f"{bu.format_inr(analysis['allocations']['Catering'])} is allocated.")
        for s in analysis["suggestions"]:
            st.warning(s)
    else:
        st.success("The catering allocation covers the estimated menu cost.")

    df = budget_dataframe(analysis)
    st.dataframe(df[["Category", "Share (%)", "Amount"]], hide_index=True)
    st.bar_chart(df.set_index("Category")["Amount (INR)"])

    st.subheader("Track your planned spending")
    spending = {}
    cols = st.columns(3)
    for i, cat in enumerate(bu.CATEGORIES):
        spending[cat] = cols[i % 3].number_input(
            f"{cat} spend (INR)", min_value=0, value=int(analysis["allocations"][cat]), step=1000,
            key=f"spend_{cat}_{amount}_{event_type}")
    comp = bu.compare_spending(analysis["allocations"], spending, amount)
    k1, k2, k3 = st.columns(3)
    k1.metric("Planned spend", bu.format_inr(comp["total_spent"]))
    k2.metric("Remaining budget", bu.format_inr(comp["remaining"]))
    k3.metric("Overspent categories", len(comp["over_categories"]))
    if comp["overspent"]:
        st.error(f"Total planned spending exceeds the budget by {bu.format_inr(-comp['remaining'])}.")
    elif comp["over_categories"]:
        st.warning("Over allocation in: " + ", ".join(comp["over_categories"]))
    else:
        st.success("Planned spending is within every allocation.")
    cdf = pd.DataFrame(comp["rows"])
    cdf["Status"] = cdf["Status"].map({"Over budget": "🔴 Over budget", "Within budget": "🟢 Within budget"})
    st.dataframe(cdf, hide_index=True)
    st.bar_chart(cdf.set_index("Category")[["Allocated", "Planned Spend"]])


# --------------------------------------------------------------------------- #
# Page 4: Preparation Checklist
# --------------------------------------------------------------------------- #
def page_checklist() -> None:
    st.title("🗒️ Preparation Checklist")
    events = safe_db(db.get_all_events, default=[])
    if not events:
        st.info("Save an event plan first; its preparation tasks become your checklist.")
        return
    ids = [e["id"] for e in events]
    by_id = {e["id"]: e for e in events}
    index = ids.index(st.session_state.event_id) if st.session_state.event_id in ids else 0
    selected = st.selectbox("Event", ids, index=index, format_func=lambda i: event_label(by_id[i]))
    items = safe_db(db.get_checklist, selected, default=[])

    if not items:
        st.info("This event has no checklist yet.")
        event = safe_db(db.get_event, selected)
        tasks = ((event or {}).get("result") or {}).get("schedule", {}).get("preparation_tasks", [])
        if tasks and st.button("Generate checklist from the saved plan"):
            safe_db(db.save_checklist, selected, tasks)
            st.rerun()
    else:
        summary = safe_db(db.get_checklist_summary, selected, default={"total": 0, "completed": 0, "remaining": 0})
        m1, m2, m3 = st.columns(3)
        m1.metric("Total tasks", summary["total"])
        m2.metric("Completed", summary["completed"])
        m3.metric("Remaining", summary["remaining"])
        st.progress(summary["completed"] / summary["total"] if summary["total"] else 0.0)
        for item in items:
            label = f"{item['task']}  (due {item['due_date']})" if item["due_date"] else item["task"]
            checked = st.checkbox(label, value=item["completed"], key=f"chk_{item['id']}")
            if checked != item["completed"]:
                safe_db(db.update_checklist_item, item["id"], checked)
                st.rerun()

    with st.form("add_task_form", clear_on_submit=True):
        new_task = st.text_input("Add your own task")
        new_due = st.date_input("Due date", value=date.today() + timedelta(days=7))
        if st.form_submit_button("➕ Add task") and new_task.strip():
            safe_db(db.add_checklist_item, selected, new_task, new_due)
            st.rerun()


# --------------------------------------------------------------------------- #
# Page 5: Agent Execution Monitor
# --------------------------------------------------------------------------- #
def page_monitor() -> None:
    st.title("🤖 Agent Execution Monitor")
    st.subheader("Agents in execution order")
    st.dataframe(pd.DataFrame(get_orchestrator().describe_agents()), hide_index=True)
    with st.expander("Deterministic orchestration rules (not LLM decisions)"):
        for code, text in ORCHESTRATION_RULES:
            st.write(f"**{code}** - {text}")

    result = st.session_state.result
    if not result:
        st.info("Generate or load a plan to see its execution trace.")
        return
    render_banner(result)
    st.subheader("Execution trace")
    render_trace_table(result)
    for t in result.get("trace", []):
        with st.expander(f"Step {t['step']}: {t['agent']} - {STATUS_ICON.get(t['status'], '')} {t['status']}"):
            st.write(f"**Action:** {t['action']}")
            st.write(f"**Result:** {t['summary']}")
            st.caption(f"Finished at {t['time']} in {t['duration_ms']} ms")
            for e in t.get("errors", []):
                st.error(e)
            for w in t.get("warnings", []):
                st.warning(w)

    st.subheader("Why did each validation rule pass or fail?")
    render_qa(result.get("qa"))

    st.subheader("Retrieved knowledge sections")
    knowledge = result.get("knowledge") or {}
    st.caption(f"Method: {knowledge.get('method', 'n/a')}")
    for q in knowledge.get("queries", []):
        st.write("🔎", q)
    for c in knowledge.get("chunks", []):
        with st.expander(f"{c['title']} - similarity {c['score']}"):
            st.write(c["text"])


# --------------------------------------------------------------------------- #
# Page 6: About
# --------------------------------------------------------------------------- #
def page_about() -> None:
    st.title("ℹ️ About This Project")
    st.markdown(f"""
**Smart Event Planning Agent** is a local multi-agent AI application that plans weddings, birthdays,
engagements, anniversaries, naming ceremonies and corporate events.

### Objective
Show how several specialised agents, retrieval (RAG), a local LLM and deterministic validation can
cooperate to produce an event plan that is checked before it is presented as valid.

### Multi-agent architecture
1. **Requirement Analysis Agent** - validates inputs.
2. **Knowledge Research Agent** - RAG retrieval from `knowledge_base/event_packages.txt`.
3. **Budget Planning Agent** - deterministic budget split and overrun detection.
4. **Venue and Logistics Agent** - capacity decision and logistics estimates.
5. **Schedule Planning Agent** - timeline and dated preparation tasks.
6. **Event Plan Writer Agent** - writes the plan with `{LLM_MODEL}` through Ollama.
7. **Quality Assurance Agent** - independently re-checks everything and returns PASS/FAIL.

The **EventPlanningOrchestrator** calls them in order and applies fixed routing rules (R1-R6).

### Key concepts
- **RAG:** the knowledge file is split into sections, each section is converted to an embedding vector with
  `{EMBED_MODEL}`, the query is embedded too, and sections are ranked by cosine similarity (NumPy).
- **Embeddings:** lists of numbers that capture meaning, so similar text has similar vectors.
- **Deterministic validation:** budget sums, capacity and overruns are calculated by Python, never by the LLM.
- **Orchestration:** the order of agents and the decisions after failures are Python rules, which makes
  results reproducible. The LLM is used for writing, not for critical constraints.

### Technologies
Python, Streamlit, Ollama, NumPy, SQLite (`sqlite3`), pandas, pytest.

### Known limitations
- The workflow is a rule-driven multi-agent pipeline, not a fully autonomous agent that re-plans itself.
- Prices and splits are illustrative estimates, not real vendor quotations.
- The app does **not** book venues, contact vendors or process payments.
- A 3B-parameter model can write generic or imperfect text; the QA Agent catches only some problems.
- Generation can be slow on computers without a GPU.
""")


# --------------------------------------------------------------------------- #
def main() -> None:
    init_state()
    try:
        db.init_db()
    except db.DatabaseError as exc:
        st.error(f"The database could not be opened: {exc}")
        st.stop()
    page = render_sidebar()
    {"Plan an Event": page_plan, "Saved Events": page_saved, "Budget Overview": page_budget,
     "Preparation Checklist": page_checklist, "Agent Execution Monitor": page_monitor,
     "About Project": page_about}[page]()


main()