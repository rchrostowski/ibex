import os
import json
import uuid
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ----------------------------
# CONFIG
# ----------------------------
APP_TITLE = "IBEX"
PRODUCTS_CSV = "products.csv"
EXCLUSIONS_CSV = "exclusions.csv"
LOGO_PATH = "assets/ibex_logo.png"

# ----------------------------
# PAGE SETUP
# ----------------------------
logo_img = Image.open(LOGO_PATH)

st.set_page_config(
    page_title="IBEX Performance Audit",
    page_icon=logo_img,
    layout="wide"
)

# Hide Streamlit chrome for cleaner SaaS look
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# DATA LOADERS
# ----------------------------
@st.cache_data(show_spinner=False)
def load_products():
    df = pd.read_csv(PRODUCTS_CSV)
    df.columns = [c.strip() for c in df.columns]

    required = [
        "Product_ID","Category","Ingredient","Brand","Store","Link",
        "Serving_Form","Typical_Use","Timing","Avoid_If",
        "Third_Party_Tested","NSF_Certified","Price","Est_Monthly_Cost","Notes"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in products.csv: {missing}")

    return df

@st.cache_data(show_spinner=False)
def load_exclusions():
    df = pd.read_csv(EXCLUSIONS_CSV)
    df.columns = [c.strip() for c in df.columns]

    if "Excluded_Category_or_Ingredient" not in df.columns or "Reason" not in df.columns:
        raise ValueError("exclusions.csv must contain Excluded_Category_or_Ingredient and Reason columns")

    return df

# ----------------------------
# AI HELPERS
# ----------------------------
def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        st.error("OPENAI_API_KEY not set in Streamlit Secrets.")
        st.stop()
    return OpenAI(api_key=api_key)

def shortlist_products(products, goals, gi_sensitive, caffeine_sensitive):
    p = products.copy()

    if goals:
        mask = False
        for g in goals:
            mask = mask | p["Typical_Use"].astype(str).str.contains(g, case=False, na=False)
        p = p[mask] if mask is not False else p

    if gi_sensitive:
        p = p[~p["Avoid_If"].astype(str).str.contains("GI", case=False, na=False)]
    if caffeine_sensitive:
        p = p[~p["Avoid_If"].astype(str).str.contains("caffeine", case=False, na=False)]

    if len(p) < 25:
        p = products.copy()

    return p.head(70)

def run_ai(intake, products_shortlist, exclusions):
    client = get_openai_client()

    approved_products = products_shortlist[[
        "Product_ID","Category","Ingredient","Brand","Store",
        "Serving_Form","Typical_Use","Timing","Avoid_If",
        "Third_Party_Tested","NSF_Certified","Notes"
    ]].to_dict(orient="records")

    system_prompt = (
        "You are IBEX, an AI that builds personalized supplement systems for athletes. "
        "You are NOT a medical provider. Do not diagnose or treat. "
        "Only select products from the approved list. "
        "Never select anything matching the exclusions list. "
        "Return ONLY valid JSON in the required format."
    )

    payload = {
        "intake": intake,
        "approved_products": approved_products,
        "exclusions": exclusions.to_dict(orient="records"),
        "output_format": {
            "flags": [],
            "consult_professional": False,
            "included_product_ids": [],
            "schedule": {"AM": [], "PM": [], "Training": []},
            "reasons": {},
            "notes_for_athlete": []
        }
    }

    response = client.chat.completions.create(
        model=st.secrets.get("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload)}
        ],
        temperature=0.2
    )

    content = response.choices[0].message.content.strip()
    start, end = content.find("{"), content.rfind("}")
    return json.loads(content[start:end+1])

# ----------------------------
# UI HEADER
# ----------------------------
col1, col2 = st.columns([1, 5])
with col1:
    st.image(LOGO_PATH, width=110)
with col2:
    st.markdown("## IBEX")
    st.caption("Personalized performance systems for athletes")

st.divider()

# ----------------------------
# LOAD DATA
# ----------------------------
products = load_products()
exclusions = load_exclusions()

# ----------------------------
# QUESTIONNAIRE
# ----------------------------
st.header("Performance Audit")

with st.form("audit"):
    c1, c2, c3 = st.columns(3)
    with c1:
        name = st.text_input("Full name")
        email = st.text_input("Email")
        school = st.text_input("School", value="Lehigh")
    with c2:
        sport = st.text_input("Sport")
        position = st.text_input("Position / Event")
        season = st.selectbox("Season status", ["In-season", "Pre-season", "Off-season"])
    with c3:
        training_days = st.slider("Training days/week", 0, 7, 5)
        intensity = st.slider("Training intensity (1–10)", 1, 10, 7)
        travel = st.selectbox("Travel frequency", ["Never", "Sometimes", "Often"])

    goals = st.multiselect(
        "Primary goals",
        ["strength","endurance","recovery","sleep","gut","joints","focus","general health"]
    )

    c4, c5 = st.columns(2)
    with c4:
        sleep_hours = st.number_input("Sleep hours/night", 0.0, 12.0, 7.0, 0.5)
        sleep_quality = st.slider("Sleep quality (1–10)", 1, 10, 6)
    with c5:
        stress = st.slider("Stress level (1–10)", 1, 10, 6)
        soreness = st.slider("Soreness/Fatigue (1–10)", 1, 10, 6)

    gi_sensitive = st.checkbox("GI sensitive")
    caffeine_sensitive = st.checkbox("Caffeine sensitive")

    current_stack = st.text_area("Current supplements (if any)")
    avoid_ingredients = st.text_input("Ingredients to avoid")
    open_notes = st.text_area("Anything else we should know?")

    consent = st.checkbox("I understand this is not medical advice.")
    submitted = st.form_submit_button("Build my IBEX system")

# ----------------------------
# RESULTS
# ----------------------------
if submitted:
    if not consent:
        st.error("Consent is required.")
        st.stop()

    rid = str(uuid.uuid4())

    intake = {
        "id": rid,
        "name": name,
        "email": email,
        "school": school,
        "sport": sport,
        "position": position,
        "season": season,
        "training_days": training_days,
        "intensity": intensity,
        "travel": travel,
        "goals": goals,
        "sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality,
        "stress": stress,
        "soreness": soreness,
        "gi_sensitive": gi_sensitive,
        "caffeine_sensitive": caffeine_sensitive,
        "current_stack": current_stack,
        "avoid_ingredients": avoid_ingredients,
        "notes": open_notes
    }

    shortlist = shortlist_products(products, goals, gi_sensitive, caffeine_sensitive)

    with st.spinner("Analyzing your audit and building your system…"):
        ai = run_ai(intake, shortlist, exclusions)

    st.success("Your system is ready.")

    if ai.get("consult_professional"):
        st.warning("We recommend consulting a qualified professional before making changes.")

    prod_map = products.set_index("Product_ID").to_dict(orient="index")

    st.subheader("Recommended System")
    cols = st.columns(3)
    for i, pid in enumerate(ai["included_product_ids"]):
        p = prod_map.get(pid)
        if not p:
            continue
        with cols[i % 3]:
            st.markdown(f"### {p['Brand']} — {p['Ingredient']}")
            st.write(p["Category"])
            st.write(f"Timing: {p['Timing']}")
            st.write(ai["reasons"].get(pid, "Personalized to your audit"))
            if isinstance(p["Link"], str) and p["Link"].startswith("http"):
                st.link_button("View product", p["Link"])
            st.divider()

    st.subheader("Daily Schedule")
    for block in ["AM", "PM", "Training"]:
        st.markdown(f"**{block}**")
        for pid in ai["schedule"].get(block, []):
            p = prod_map.get(pid, {})
            st.write(f"- {p.get('Ingredient', pid)} ({p.get('Brand','')})")

    st.subheader("Checkout")
    st.write("Start your monthly IBEX subscription:")

    cA, cB = st.columns(2)
    with cA:
        st.link_button("Subscribe — Basic", st.secrets.get("STRIPE_BASIC_LINK", "#"))
    with cB:
        st.link_button("Subscribe — Performance", st.secrets.get("STRIPE_PERF_LINK", "#"))

    st.caption(f"Reference ID: {rid}")
