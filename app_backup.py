
import streamlit as st
from datetime import date

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Event Planning Agent",
    page_icon="🎉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #fff8fb;
    }

    .main-title {
        font-size: 36px;
        font-weight: 800;
        color: #8e2455;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #66515c;
        font-size: 16px;
        margin-bottom: 24px;
    }

    .section-title {
        color: #8e2455;
        font-size: 23px;
        font-weight: 700;
    }

    div.stButton > button[kind="primary"] {
        background-color: #a83268;
        color: white;
        border-radius: 9px;
        border: none;
        min-height: 45px;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #7e204d;
        color: white;
    }

    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #f0d9e5;
        padding: 15px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "event_plan" not in st.session_state:
    st.session_state.event_plan = None

if "completed_tasks" not in st.session_state:
    st.session_state.completed_tasks = []

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.title("🎊 Event Planner")
    st.caption("Your personal event planning assistant")

    page = st.radio(
        "Navigation",
        [
            "Plan an Event",
            "Budget Overview",
            "Preparation Checklist",
            "About Project"
        ]
    )

    st.divider()

    st.markdown("### Project Features")
    st.write("✓ Event requirement collection")
    st.write("✓ Budget estimation")
    st.write("✓ Guest management")
    st.write("✓ Event preparation checklist")
    st.write("✓ Planning validation")

    st.caption("Agentic AI project — development version")

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    '<div class="main-title">Smart Event Planning Agent 🎉</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Plan weddings, birthdays, engagements, and other events '
    'with a personalized budget and preparation plan.'
    '</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# EVENT PLANNING PAGE
# --------------------------------------------------
if page == "Plan an Event":

    st.markdown(
        '<div class="section-title">Tell Us About Your Event</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter your requirements below to generate a proposed "
        "event plan."
    )

    with st.form("event_form"):

        col1, col2 = st.columns(2)

        with col1:
            event_type = st.selectbox(
                "Event Type",
                [
                    "Wedding",
                    "Birthday Party",
                    "Engagement / Sakharpuda",
                    "Haldi Ceremony",
                    "Anniversary",
                    "Naming Ceremony",
                    "Corporate Event"
                ]
            )

            budget = st.number_input(
                "Total Budget (₹)",
                min_value=5000,
                max_value=10000000,
                value=100000,
                step=5000
            )

            guests = st.number_input(
                "Number of Guests",
                min_value=1,
                max_value=10000,
                value=100,
                step=10
            )

            location = st.text_input(
                "Event Location",
                placeholder="e.g. Pune, Maharashtra"
            )

        with col2:
            event_date = st.date_input(
                "Event Date",
                min_value=date.today()
            )

            theme = st.selectbox(
                "Decoration Theme",
                [
                    "Traditional Maharashtrian",
                    "Floral",
                    "Royal",
                    "Modern Minimalist",
                    "Colorful Birthday",
                    "Corporate Professional"
                ]
            )

            venue_type = st.selectbox(
                "Preferred Venue",
                [
                    "Banquet Hall",
                    "Wedding Lawn",
                    "Hotel",
                    "Outdoor Venue",
                    "Home",
                    "Not Decided"
                ]
            )

            catering = st.checkbox(
                "Include Catering",
                value=True
            )

            photography = st.checkbox(
                "Include Photography",
                value=True
            )

            entertainment = st.checkbox(
                "Include Entertainment",
                value=True
            )

        submitted = st.form_submit_button(
            "✨ Generate Event Plan",
            type="primary",
            use_container_width=True
        )

    if submitted:

        if not location.strip():
            st.error("Please enter your event location.")

        else:
            # Initial budget allocation rules
            shares = {
                "Venue": 0.25,
                "Decoration": 0.15,
                "Catering": 0.35 if catering else 0,
                "Photography": 0.10 if photography else 0,
                "Entertainment": 0.08 if entertainment else 0,
                "Miscellaneous": 0.07
            }

            # Normalize active categories so total = user budget
            active_total = sum(shares.values())

            if active_total <= 0:
                st.error("Unable to calculate the budget.")
                st.stop()

            normalized = {
                category: share / active_total
                for category, share in shares.items()
            }

            breakdown = {
                category: round(budget * share, 2)
                for category, share in normalized.items()
            }

            plan = {
                "event_type": event_type,
                "budget": float(budget),
                "guests": int(guests),
                "location": location.strip(),
                "event_date": event_date,
                "theme": theme,
                "venue_type": venue_type,
                "catering": catering,
                "photography": photography,
                "entertainment": entertainment,
                "breakdown": breakdown
            }

            st.session_state.event_plan = plan
            st.session_state.completed_tasks = []

            st.success("Your proposed event plan has been generated!")

    # Display the generated plan
    plan = st.session_state.event_plan

    if plan is not None:

        st.divider()
        st.subheader("📋 Your Event Plan")

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric("Event Budget", f"₹{plan['budget']:,.0f}")

        with metric2:
            st.metric("Guests", f"{plan['guests']:,}")

        with metric3:
            st.metric(
                "Budget per Guest",
                f"₹{plan['budget'] / plan['guests']:,.0f}"
            )

        details1, details2 = st.columns(2)

        with details1:
            st.markdown("#### Event Details")
            st.write(f"**Event:** {plan['event_type']}")
            st.write(f"**Location:** {plan['location']}")
            st.write(
                f"**Date:** {plan['event_date'].strftime('%d %B %Y')}"
            )

        with details2:
            st.markdown("#### Preferences")
            st.write(f"**Theme:** {plan['theme']}")
            st.write(f"**Venue:** {plan['venue_type']}")
            st.write(
                f"**Catering:** {'Yes' if plan['catering'] else 'No'}"
            )
            st.write(
                f"**Photography:** "
                f"{'Yes' if plan['photography'] else 'No'}"
            )
            st.write(
                f"**Entertainment:** "
                f"{'Yes' if plan['entertainment'] else 'No'}"
            )

        st.divider()
        st.subheader("💰 Estimated Budget Breakdown")

        for category, amount in plan["breakdown"].items():
            if amount > 0:
                percentage = amount / plan["budget"]
                st.write(
                    f"**{category}:** ₹{amount:,.0f} "
                    f"({percentage:.0%})"
                )
                st.progress(min(percentage, 1.0))

        st.caption(
            "These are illustrative budget allocations, not actual "
            "vendor quotes. Real prices depend on local rates, "
            "availability, guest requirements, and service quality."
        )

        st.divider()
        st.subheader("🗓️ Suggested Preparation Timeline")

        timeline = [
            (
                "4–8 weeks before",
                "Shortlist venues, set requirements, and compare packages."
            ),
            (
                "2–4 weeks before",
                "Finalize decoration, catering, and photography."
            ),
            (
                "1 week before",
                "Confirm guest count, schedule, and vendor arrangements."
            ),
            (
                "1 day before",
                "Review setup, supplies, and the final event checklist."
            ),
            (
                "Event day",
                "Check venue readiness and follow the event schedule."
            )
        ]

        for when, task in timeline:
            st.markdown(f"**{when}:** {task}")

        st.info(
            "Timeline entries are general suggestions. Adjust them "
            "according to the event date and vendor requirements."
        )

        # Download the plan as a text file
        report_lines = [
            "SMART EVENT PLANNING AGENT",
            "=" * 35,
            f"Event: {plan['event_type']}",
            f"Location: {plan['location']}",
            f"Date: {plan['event_date']}",
            f"Guests: {plan['guests']}",
            f"Budget: INR {plan['budget']:,.0f}",
            f"Theme: {plan['theme']}",
            f"Venue: {plan['venue_type']}",
            "",
            "BUDGET BREAKDOWN",
            "-" * 25
        ]

        for category, amount in plan["breakdown"].items():
            report_lines.append(
                f"{category}: INR {amount:,.0f}"
            )

        report_text = "\n".join(report_lines)

        st.download_button(
            "📥 Download Event Plan",
            data=report_text,
            file_name="event_plan.txt",
            mime="text/plain"
        )

# --------------------------------------------------
# BUDGET OVERVIEW PAGE
# --------------------------------------------------
elif page == "Budget Overview":

    st.subheader("💰 Budget Overview")

    plan = st.session_state.event_plan

    if plan is None:
        st.info(
            "Generate an event plan first to view its budget overview."
        )

    else:
        st.write(f"**Event:** {plan['event_type']}")
        st.write(f"**Budget:** ₹{plan['budget']:,.0f}")

        for category, amount in plan["breakdown"].items():
            st.write(f"{category}: ₹{amount:,.0f}")

        st.metric(
            "Total Allocated",
            f"₹{sum(plan['breakdown'].values()):,.0f}"
        )

        st.caption(
            "This page shows the planned allocation, not verified "
            "market prices or actual spending."
        )

# --------------------------------------------------
# CHECKLIST PAGE
# --------------------------------------------------
elif page == "Preparation Checklist":

    st.subheader("✅ Event Preparation Checklist")

    plan = st.session_state.event_plan

    if plan is None:
        st.info("Generate an event plan before using the checklist.")

    else:
        st.write(f"**Event:** {plan['event_type']}")

        tasks = [
            "Confirm venue requirements and capacity",
            "Prepare the guest list",
            "Compare decoration packages",
            "Finalize catering requirements",
            "Confirm photography arrangements",
            "Prepare the event-day schedule",
            "Confirm arrangements with vendors",
            "Review the final budget",
            "Check all arrangements before the event"
        ]

        for task in tasks:
            done = task in st.session_state.completed_tasks

            if st.checkbox(task, value=done, key=f"check_{task}"):
                if task not in st.session_state.completed_tasks:
                    st.session_state.completed_tasks.append(task)
            else:
                if task in st.session_state.completed_tasks:
                    st.session_state.completed_tasks.remove(task)

        completed = len(st.session_state.completed_tasks)
        total = len(tasks)

        st.progress(completed / total)
        st.write(f"Completed {completed} of {total} tasks.")

# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------
elif page == "About Project":

    st.subheader("ℹ️ About This Project")

    st.write(
        """
        The Smart Event Planning Agent is an academic project
        designed to assist users in organizing events according
        to their budgets, guest counts, dates, locations, and
        personal preferences.
        """
    )

    st.markdown("### Current Version")
    st.write("- Event requirement collection")
    st.write("- Budget allocation")
    st.write("- Event preparation timeline")
    st.write("- Checklist management")
    st.write("- Event plan download")

    st.warning(
        "This version is the initial planning interface. It does "
        "not yet use an LLM-based autonomous agent, RAG retrieval, "
        "live venue availability, or real vendor APIs. Those features "
        "must be implemented and tested separately."
    )

    st.markdown("### Planned Agentic AI Features")
    st.write("- LLM-based task planning and tool selection")
    st.write("- Knowledge retrieval from event packages and guides")
    st.write("- Budget constraint checking and plan revision")
    st.write("- Evidence-based recommendations and safety checks")