
import json
from datetime import date
from pathlib import Path

import numpy as np
import ollama
import streamlit as st

import database as db


# ==================================================
# CONFIGURATION
# ==================================================

APP_TITLE = " Smart Event Planning Agent"
CHAT_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"

BASE_DIR = Path(__file__).parent
KNOWLEDGE_FILE = (
    BASE_DIR / "knowledge_base" / "event_packages.txt"
)

db.init_db()

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide"
)


# ==================================================
# STYLING
# ==================================================

st.markdown("""
<style>
/* Soft pink + lavender visual theme */
:root {
    --rose: #bd4f83;
    --rose-dark: #963b68;
    --lavender: #8a78c7;
    --ink: #34263b;
    --muted: #766779;
    --panel: rgba(255, 255, 255, 0.92);
}
.stApp {
    background: linear-gradient(145deg, #fff8fc 0%, #fbf7ff 48%, #fffafd 100%);
    color: var(--ink);
}
[data-testid="stHeader"] { background: rgba(255,255,255,0.72); }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fff0f7 0%, #f4efff 100%);
    border-right: 1px solid #f0dce9;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #87375f; }
.stApp h1, .stApp h2, .stApp h3 { color: #87375f; letter-spacing: -0.02em; }
.stApp h1 { font-weight: 800; }
.stApp p, .stApp label { color: #4f4054; }
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.9);
    border: 1px solid #f0d9e8;
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 5px 18px rgba(128, 65, 105, 0.05);
}
[data-testid="stMetricLabel"] { color: #766779; }
[data-testid="stMetricValue"] { color: #963b68; font-weight: 750; }
[data-testid="stForm"], [data-testid="stExpander"],
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border-radius: 16px;
}
[data-testid="stForm"] {
    background: rgba(255,255,255,0.78);
    border: 1px solid #f0dce9;
    padding: 22px;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #efd9e8 !important;
    border-radius: 16px !important;
}
div.stButton > button, div.stDownloadButton > button,
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(100deg, #bd4f83, #8a78c7);
    color: white;
    border: 0;
    border-radius: 12px;
    padding: 0.58rem 1rem;
    font-weight: 650;
    transition: transform .15s ease, box-shadow .15s ease;
    box-shadow: 0 4px 12px rgba(158, 72, 132, 0.14);
}
div.stButton > button:hover, div.stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    color: white;
    border: 0;
    transform: translateY(-1px);
    box-shadow: 0 7px 18px rgba(158, 72, 132, 0.2);
}
input, textarea, [data-baseweb="select"] > div {
    border-radius: 10px !important;
}
[data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg, #bd4f83, #9b8ade); }
hr { border-color: #efdce9; }
.hero-card {
    background: linear-gradient(115deg, #fff0f7 0%, #f1edff 100%);
    border: 1px solid #efd9e9;
    border-radius: 22px;
    padding: 26px 30px;
    margin: 4px 0 20px 0;
    box-shadow: 0 8px 26px rgba(128, 65, 105, 0.07);
}
.hero-eyebrow {
    color: #963b68;
    text-transform: uppercase;
    letter-spacing: .12em;
    font-size: .76rem;
    font-weight: 800;
    margin-bottom: 8px;
}
.hero-title { color: #542d50; font-size: 2rem; font-weight: 850; margin: 0; }
.hero-copy { color: #6f5c76; font-size: 1rem; margin-top: 8px; }
.status-pill {
    display: inline-block;
    background: rgba(255,255,255,.8);
    border: 1px solid #e8d5ed;
    border-radius: 999px;
    padding: 6px 11px;
    margin: 10px 6px 0 0;
    color: #704a78;
    font-size: .82rem;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# SESSION STATE
# ==================================================

defaults = {
    "event_plan": "",
    "event_details": {},
    "checklist": [],
    "current_event_id": None,
    "loaded_event_id": None,
    "checklist_saved_notice": False
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==================================================
# KNOWLEDGE BASE / RAG
# ==================================================

def load_knowledge():
    if not KNOWLEDGE_FILE.exists():
        return []

    content = KNOWLEDGE_FILE.read_text(encoding="utf-8")

    return [
        part.strip()
        for part in content.split("\n\n")
        if part.strip()
    ]


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def retrieve_knowledge(query, top_k=3):
    documents = load_knowledge()

    if not documents:
        return []

    try:
        response = ollama.embed(
            model=EMBED_MODEL,
            input=[query] + documents
        )

        vectors = response["embeddings"]
        query_vector = vectors[0]

        ranked = []

        for document, vector in zip(
            documents, vectors[1:]
        ):
            score = cosine_similarity(query_vector, vector)
            ranked.append((score, document))

        ranked.sort(key=lambda item: item[0], reverse=True)

        return [
            document for score, document in ranked[:top_k]
        ]

    except Exception as error:
        st.warning(
            "RAG search was unavailable. The planner will "
            f"continue without retrieved information. Details: {error}"
        )
        return documents[:top_k]


# ==================================================
# PLANNING TOOLS
# ==================================================

def calculate_budget(budget, guest_count):
    allocations = {
        "Venue and decoration": 0.30,
        "Food and catering": 0.35,
        "Photography and entertainment": 0.15,
        "Invitations and miscellaneous": 0.10,
        "Emergency reserve": 0.10
    }

    breakdown = {
        name: round(budget * percentage, 2)
        for name, percentage in allocations.items()
    }

    return {
        "total_budget": budget,
        "guest_count": guest_count,
        "budget_per_guest": round(
            budget / guest_count, 2
        ) if guest_count else 0,
        "breakdown": breakdown
    }


def check_venue_capacity(guest_count, venue_capacity):
    suitable = guest_count <= venue_capacity

    if suitable:
        message = (
            f"The listed capacity of {venue_capacity} "
            f"can accommodate {guest_count} guests."
        )
    else:
        message = (
            f"The venue capacity is {venue_capacity}, "
            f"but there are {guest_count} guests. "
            "Consider a larger venue."
        )

    return {
        "suitable": suitable,
        "message": message
    }


def create_schedule(event_type, event_date):
    event = event_type.lower()

    if "birthday" in event:
        activities = [
            ("4:00 PM", "Guest arrival and welcome"),
            ("4:30 PM", "Games and entertainment"),
            ("5:30 PM", "Cake cutting"),
            ("6:00 PM", "Dinner and photographs"),
            ("7:00 PM", "Event closing")
        ]

    elif "wedding" in event or "marriage" in event:
        activities = [
            ("9:00 AM", "Venue and decoration check"),
            ("10:00 AM", "Guest welcome"),
            ("11:00 AM", "Main ceremony"),
            ("1:00 PM", "Lunch"),
            ("3:00 PM", "Photography and guest farewell")
        ]

    else:
        activities = [
            ("10:00 AM", "Venue and equipment check"),
            ("11:00 AM", "Guest arrival"),
            ("12:00 PM", "Main event"),
            ("1:00 PM", "Food and refreshments"),
            ("3:00 PM", "Photography and closing")
        ]

    return {
        "event_date": str(event_date),
        "activities": activities
    }


# ==================================================
# LOCAL AI GENERATION
# ==================================================

def generate_event_plan(
    details, budget_data, venue_data, schedule, knowledge
):
    knowledge_text = (
        "\n\n".join(knowledge)
        if knowledge
        else "No relevant knowledge-base sections found."
    )

    prompt = f"""
Create a practical event plan based on the following data.

EVENT DETAILS:
{json.dumps(details, indent=2, ensure_ascii=False)}

BUDGET TOOL RESULTS:
{json.dumps(budget_data, indent=2)}

VENUE CAPACITY TOOL RESULTS:
{json.dumps(venue_data, indent=2)}

SCHEDULE TOOL RESULTS:
{json.dumps(schedule, indent=2)}

RETRIEVED KNOWLEDGE BASE:
{knowledge_text}

Include:
1. Event overview
2. Theme and decoration suggestions
3. Budget breakdown and cost-saving tips
4. The supplied event schedule
5. Venue suitability
6. Catering and logistics
7. A preparation checklist
8. Potential risks and precautions

Treat prices as estimates. Do not invent confirmed bookings,
real vendors, or supplier quotations. Be clear and organized.
"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a practical AI event-planning assistant. "
                    "Use the supplied calculation results accurately."
                )
            },
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0.4}
    )

    return response["message"]["content"]


# ==================================================
# EVENT LOADING
# ==================================================

def load_saved_event(event_id):
    event = db.get_event(event_id)

    if event is None:
        st.error("This saved event could not be found.")
        return

    st.session_state.current_event_id = event["id"]
    st.session_state.loaded_event_id = event["id"]
    st.session_state.event_plan = event["generated_plan"] or ""

    st.session_state.event_details = {
        "event_type": event["event_type"],
        "event_date": event["event_date"],
        "location": event["location"] or "",
        "guest_count": event["guest_count"],
        "budget_in_rupees": event["budget"],
        "venue_capacity": event["venue_capacity"],
        "theme": event["theme"] or "",
        "food_preference": event["food_preference"] or "",
        "special_requirements": (
            event["special_requirements"] or ""
        )
    }

    st.session_state.checklist = event["checklist"]

    # Reset checklist widget values to the saved event's values.
    for index, item in enumerate(st.session_state.checklist):
        st.session_state[f"task_{index}"] = item["done"]


# ==================================================
# HEADER AND NAVIGATION
# ==================================================

st.markdown("""
<div class="hero-card">
  <div class="hero-eyebrow">Your personal event assistant</div>
  <div class="hero-title">🎉 Smart Event Planning Agent</div>
  <div class="hero-copy">Plan memorable events with local AI, practical planning tools, and your own saved event history.</div>
  <span class="status-pill">✦ Local AI · Ollama</span>
  <span class="status-pill">⌕ Knowledge retrieval · RAG</span>
  <span class="status-pill">▣ Saved locally · SQLite</span>
</div>
""", unsafe_allow_html=True)

st.caption("Choose a workspace from the sidebar to plan an event, review budgets, track tasks, or manage saved plans.")

page = st.sidebar.radio(
    "Navigation",
    [
        "Plan an Event",
        "Saved Events",
        "Budget Overview",
        "Preparation Checklist",
        "About Project"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### Local AI")
st.sidebar.caption(f"Chat model: {CHAT_MODEL}")
st.sidebar.caption(f"Embedding model: {EMBED_MODEL}")


# ==================================================
# PAGE 1: PLAN AN EVENT
# ==================================================

if page == "Plan an Event":
    st.header("Plan Your Event")
    st.caption("Tell us the essentials. Your local AI agent will combine retrieved guidance and planning tools into a structured plan.")

    with st.form("event_form"):
        col1, col2 = st.columns(2)

        with col1:
            event_type = st.selectbox(
                "Event Type",
                [
                    "Birthday Party",
                    "Wedding / Marriage",
                    "Engagement",
                    "Corporate Event",
                    "Anniversary",
                    "Baby Shower",
                    "Other"
                ]
            )

            event_date = st.date_input(
                "Event Date",
                value=date.today()
            )

            location = st.text_input(
                "Event Location",
                placeholder="Enter city or venue area"
            )

            guest_count = st.number_input(
                "Number of Guests",
                min_value=1,
                max_value=100000,
                value=100,
                step=10
            )

        with col2:
            budget = st.number_input(
                "Total Budget (₹)",
                min_value=1000,
                max_value=100000000,
                value=50000,
                step=5000
            )

            venue_capacity = st.number_input(
                "Venue Capacity",
                min_value=1,
                max_value=100000,
                value=150,
                step=10
            )

            theme = st.selectbox(
                "Preferred Theme",
                [
                    "Traditional",
                    "Modern",
                    "Floral",
                    "Elegant",
                    "Minimal",
                    "Colorful",
                    "AI Recommendation"
                ]
            )

            food_preference = st.selectbox(
                "Food Preference",
                [
                    "Vegetarian",
                    "Non-vegetarian",
                    "Both",
                    "Not decided"
                ]
            )

        special_requirements = st.text_area(
            "Special Requirements",
            placeholder=(
                "Decoration, photography, games, accessibility, etc."
            )
        )

        submitted = st.form_submit_button(
            "✨ Generate and Save Event Plan",
            use_container_width=True
        )

    if submitted:
        details = {
            "event_type": event_type,
            "event_date": str(event_date),
            "location": location or "Not specified",
            "guest_count": int(guest_count),
            "budget_in_rupees": float(budget),
            "venue_capacity": int(venue_capacity),
            "theme": theme,
            "food_preference": food_preference,
            "special_requirements": special_requirements
        }

        budget_data = calculate_budget(
            float(budget), int(guest_count)
        )

        venue_data = check_venue_capacity(
            int(guest_count), int(venue_capacity)
        )

        schedule = create_schedule(event_type, event_date)

        query = (
            f"{event_type} theme {theme}, {guest_count} guests, "
            f"food {food_preference}, budget {budget} rupees"
        )

        try:
            with st.spinner("Retrieving event knowledge..."):
                knowledge = retrieve_knowledge(query)

            with st.spinner("Generating your event plan with Ollama..."):
                plan = generate_event_plan(
                    details,
                    budget_data,
                    venue_data,
                    schedule,
                    knowledge
                )

            checklist = [
                {"task": task, "done": False}
                for task in [
                    "Confirm the venue booking",
                    "Finalize the event budget",
                    "Confirm decoration requirements",
                    "Arrange catering and refreshments",
                    "Confirm photography and entertainment",
                    "Send invitations to guests",
                    "Check sound, lighting and electricity",
                    "Confirm vendors and delivery times",
                    "Prepare emergency contacts",
                    "Inspect the venue before the event"
                ]
            ]

            # Save to SQLite immediately after successful generation.
            saved_id = db.save_event(
                details=details,
                generated_plan=plan,
                checklist=checklist
            )

            st.session_state.event_plan = plan
            st.session_state.event_details = details
            st.session_state.checklist = checklist
            st.session_state.current_event_id = saved_id
            st.session_state.loaded_event_id = saved_id

            for index in range(len(checklist)):
                st.session_state[f"task_{index}"] = False

            st.success(
                f"Event plan generated and saved! Database record ID: {saved_id}"
            )

        except Exception as error:
            st.error(
                "The plan could not be generated or saved. "
                "Check Ollama and try again."
            )
            st.code(str(error))

    if st.session_state.event_plan:
        st.divider()
        st.subheader("Your Event Plan")

        if st.session_state.current_event_id is not None:
            st.caption(
                f"Saved event ID: {st.session_state.current_event_id}"
            )

        st.markdown(st.session_state.event_plan)

        st.download_button(
            "⬇️ Download Event Plan",
            data=st.session_state.event_plan,
            file_name="smart_event_plan.txt",
            mime="text/plain"
        )


# ==================================================
# PAGE 2: SAVED EVENTS
# ==================================================

elif page == "Saved Events":
    st.header(" Saved Events")
    st.caption("Reopen a previous plan, review its details, or remove an event you no longer need.")

    events = db.get_all_events()

    if not events:
        st.info("No events have been saved yet.")
    else:
        st.write(f"Total saved events: {len(events)}")

        options = {
            (
                f"ID {event['id']} | {event['event_type']} | "
                f"{event['event_date']} | "
                f"{event['guest_count']} guests"
            ): event["id"]
            for event in events
        }

        selected_label = st.selectbox(
            "Choose an event",
            list(options.keys())
        )

        selected_id = options[selected_label]

        selected_event = db.get_event(selected_id)

        if selected_event:
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Guests", selected_event["guest_count"]
            )

            col2.metric(
                "Budget",
                f"₹{selected_event['budget']:,.0f}"
            )

            col3.metric(
                "Venue Capacity",
                selected_event["venue_capacity"]
            )

            st.write(
                f"**Location:** {selected_event['location'] or 'Not specified'}"
            )

            st.write(
                f"**Created:** {selected_event['created_at']}"
            )

            left, right = st.columns(2)

            with left:
                if st.button("Load Event", use_container_width=True):
                    load_saved_event(selected_id)
                    st.success(
                        "Event loaded. Open Plan an Event or "
                        "Preparation Checklist to view it."
                    )

            with right:
                if st.button(
                    "Delete Event",
                    use_container_width=True
                ):
                    st.session_state["confirm_delete_id"] = selected_id

            if st.session_state.get("confirm_delete_id") == selected_id:
                st.warning(
                    "Deleting an event permanently removes its "
                    "saved plan and checklist from this database."
                )

                confirm_col, cancel_col = st.columns(2)

                with confirm_col:
                    if st.button("Confirm Permanent Delete"):
                        db.delete_event(selected_id)

                        if (
                            st.session_state.current_event_id
                            == selected_id
                        ):
                            st.session_state.current_event_id = None
                            st.session_state.loaded_event_id = None
                            st.session_state.event_plan = ""
                            st.session_state.event_details = {}
                            st.session_state.checklist = []

                        del st.session_state["confirm_delete_id"]

                        st.rerun()

                with cancel_col:
                    if st.button("Cancel Delete"):
                        del st.session_state["confirm_delete_id"]
                        st.rerun()

            with st.expander("View saved event plan"):
                st.markdown(
                    selected_event["generated_plan"] or
                    "No plan has been saved."
                )

            st.download_button(
                "⬇️ Download Selected Plan",
                data=selected_event["generated_plan"] or "",
                file_name=f"event_{selected_id}_plan.txt",
                mime="text/plain"
            )


# ==================================================
# PAGE 3: BUDGET OVERVIEW
# ==================================================

elif page == "Budget Overview":
    st.header("💰 Budget Overview")
    st.caption("Review an estimated allocation for the currently loaded event. Values are planning estimates, not supplier quotations.")

    details = st.session_state.event_details

    if details:
        budget_data = calculate_budget(
            details["budget_in_rupees"],
            details["guest_count"]
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Budget",
            f"₹{budget_data['total_budget']:,.0f}"
        )

        col2.metric(
            "Guests",
            f"{budget_data['guest_count']:,}"
        )

        col3.metric(
            "Budget per Guest",
            f"₹{budget_data['budget_per_guest']:,.2f}"
        )

        st.subheader("Estimated Allocation")

        for category, amount in budget_data["breakdown"].items():
            percentage = (
                amount / budget_data["total_budget"]
                if budget_data["total_budget"]
                else 0
            )

            st.write(f"**{category}: ₹{amount:,.2f}**")

            st.progress(
                min(max(percentage, 0.0), 1.0),
                text=f"{percentage * 100:.0f}%"
            )

        st.caption(
            "These are estimated allocations, not actual vendor quotations."
        )

    else:
        st.info(
            "Generate a new plan or load an event from Saved Events "
            "to view its budget."
        )


# ==================================================
# PAGE 4: PREPARATION CHECKLIST
# ==================================================

elif page == "Preparation Checklist":
    st.header("✅ Preparation Checklist")
    st.caption("Track important tasks and save your progress to the local database.")

    event_id = st.session_state.current_event_id
    checklist = st.session_state.checklist

    if event_id is None or not checklist:
        st.info(
            "Generate a plan or load a saved event first."
        )
    else:
        st.caption(f"Saved event ID: {event_id}")

        updated_checklist = []

        for index, item in enumerate(checklist):
            key = f"task_{index}"

            if key not in st.session_state:
                st.session_state[key] = bool(item["done"])

            checked = st.checkbox(
                item["task"],
                key=key
            )

            updated_checklist.append({
                "task": item["task"],
                "done": checked
            })

        completed = sum(
            1 for item in updated_checklist if item["done"]
        )

        total = len(updated_checklist)
        progress = completed / total if total else 0

        st.progress(
            progress,
            text=f"{completed} of {total} tasks completed"
        )

        if st.button("💾 Save Checklist Progress"):
            db.update_checklist(
                event_id,
                updated_checklist
            )

            st.session_state.checklist = updated_checklist

            st.success(
                "Checklist progress saved permanently to SQLite."
            )

        checklist_text = "\n".join(
            [
                ("[x] " if item["done"] else "[ ] ")
                + item["task"]
                for item in updated_checklist
            ]
        )

        st.download_button(
            "⬇️ Download Checklist",
            data=checklist_text,
            file_name="event_checklist.txt",
            mime="text/plain"
        )


# ==================================================
# PAGE 5: ABOUT PROJECT
# ==================================================

elif page == "About Project":
    st.header("About the Project")

    st.markdown("""
**Smart Event Planning Agent** helps users organize events using
AI-generated plans and practical planning tools.

### Main Features

- Local AI generation using Ollama and Llama 3.2
- Retrieval-Augmented Generation (RAG) from local event information
- Budget estimation and allocation
- Venue-capacity checking
- Sample event scheduling
- Preparation checklist with persistent progress
- SQLite storage for event records
- Saved-event history and event-plan downloads

### Technologies

- Python
- Streamlit
- Ollama
- Llama 3.2
- Nomic Embed Text
- NumPy
- SQLite

### Database

The local database is stored in `event_planner.db`.
It contains event details, generated plans, and checklist progress.

The application does not automatically make real vendor bookings
or process payments. Generated budgets are estimates.
""")

    st.subheader("System Workflow")

    st.markdown("""
1. The user enters event details.
2. Python tools calculate the budget, venue suitability and schedule.
3. RAG retrieves relevant information from the knowledge base.
4. Ollama generates the event plan.
5. The event plan and checklist are saved to SQLite.
6. The user can reopen saved events, track tasks and download plans.
""")


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Smart Event Planning Agent | Ollama + RAG + SQLite"
)
