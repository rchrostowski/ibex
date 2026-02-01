import os
import json
import uuid
import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

APP_TITLE = "IBEX Performance Audit"
DATA_PRODUCTS = os.path.join("data", "products.csv")
DATA_EXCLUSIONS = os.path.join("data", "exclusions.csv")

st.set_page_config(page_title=APP_TITLE, page_icon="🦌", layout="wide")

@st.cache_data(show_spinner=False)
def load_products() -> pd.DataFrame:
    df = pd.read_csv(DATA_PRODUCTS)
    df.columns = [c.strip() for c in df.columns]
    required = [
        "Product_ID","Category","Ingredient","Brand","Store","Link","Serving_Form",
        "Typical_Use","Timing","Avoid_If","Third_Party_Tested","NSF_Certified",
        "Price","Est_Monthly_Cost","Notes"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"products.csv missing columns: {missing}")
    return df

@st.cache_data(show_spinner=False)
def load_exclusions() -> pd.DataFrame:
    df = pd.read_csv(DATA_EXCLUSIONS)
    df.columns = [c.strip() for c in df.columns]
    if "Excluded_Category_or_Ingredient" not in df.columns:
        if "excluded_item" in df.columns:
            df = df.rename(columns={"excluded_item":"Excluded_Category_or_Ingredient"})
        else:
            raise ValueError("exclusions.csv must include 'Excluded_Category_or_Ingredient'")
    if "Reason" not in df.columns:
        raise ValueError("exclusions.csv must include 'Reason'")
    return df[["Excluded_Category_or_Ingredient","Reason"]]

def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    if OpenAI is None:
        st.error("openai package not installed. Install requirements.txt.")
        return None
    return OpenAI(api_key=api_key)

def shortlist_products(products: pd.DataFrame, goals: list[str], gi_sensitive: bool, caffeine_sensitive: bool) -> pd.DataFrame:
    p = products.copy()
    if goals:
        mask = False
        for g in goals:
            mask = mask | p["Typical_Use"].astype(str).str.contains(g, case=False, na=False)
        if mask is not False:
            p = p[mask]
    if gi_sensitive:
        p = p[~p["Avoid_If"].astype(str).str.contains("GI", case=False, na=False)]
    if caffeine_sensitive:
        p = p[~p["Avoid_If"].astype(str).str.contains("caffeine", case=False, na=False)]
    if len(p) < 25:
        p = products.copy()
    return p.head(70)

def build_payload(intake: dict, approved_products: list[dict], exclusions: list[dict]) -> str:
    schema = {
        "flags": ["string"],
        "consult_professional": "boolean",
        "included_product_ids": ["IBX-0001"],
        "excluded_product_ids": ["IBX-0002"],
        "schedule": {"AM": ["IBX-0001"], "PM": ["IBX-0003"], "Training": ["IBX-0004"]},
        "reasons": {"IBX-0001": "short non-medical reason"},
        "notes_for_athlete": ["bullet", "bullet"]
    }
    payload = {
        "intake": intake,
        "approved_products": approved_products,
        "exclusions": exclusions,
        "output_format": schema
    }
    return json.dumps(payload, ensure_ascii=False)

def run_ai(intake: dict, products_shortlist: pd.DataFrame, exclusions_df: pd.DataFrame) -> dict:
    client = get_openai_client()
    if client is None:
        raise RuntimeError("Missing OPENAI_API_KEY (Streamlit Secrets or env var).")
    approved = products_shortlist[[
        "Product_ID","Category","Ingredient","Brand","Store","Serving_Form",
        "Typical_Use","Timing","Avoid_If","Third_Party_Tested","NSF_Certified","Notes"
    ]].to_dict(orient="records")
    exclusions = exclusions_df.to_dict(orient="records")

    system = (
        "You are IBEX, an assistant that organizes a personalized supplement system for student-athletes. "
        "You are NOT a medical provider. Do NOT diagnose, treat, or make medical claims. "
        "Only select products from approved_products. "
        "Never select anything that matches the EXCLUSIONS list. "
        "If intake suggests a medical issue or medication interaction, set consult_professional=true and keep recommendations conservative. "
        "Return ONLY valid JSON matching output_format."
    )

    model = st.secrets.get("OPENAI_MODEL", "gpt-4.1-mini") if hasattr(st, "secrets") else os.getenv("OPENAI_MODEL","gpt-4.1-mini")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":build_payload(intake, approved, exclusions)}
        ],
        temperature=0.2
    )
    content = resp.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start:end+1])
        raise

def render_products(product_ids: list[str], products_df: pd.DataFrame, reasons: dict):
    prod_map = products_df.set_index("Product_ID").to_dict(orient="index")
    cols = st.columns(3)
    for idx, pid in enumerate(product_ids):
        p = prod_map.get(pid)
        if not p:
            continue
        with cols[idx % 3]:
            st.markdown(f"### {p['Brand']} — {p['Ingredient']}")
            st.write(f"**Category:** {p['Category']}")
            st.write(f"**Store:** {p['Store']}")
            st.write(f"**Form:** {p['Serving_Form']}")
            st.write(f"**Timing (default):** {p['Timing']}")
            st.write(f"**Reason:** {reasons.get(pid, 'Personalized to your audit')}")
            link = str(p.get("Link","")).strip()
            if link and link != "in-store" and link.startswith("http"):
                st.link_button("View / Reference", link)
            st.divider()

def render_schedule(schedule: dict, products_df: pd.DataFrame):
    prod_map = products_df.set_index("Product_ID").to_dict(orient="index")
    scols = st.columns(3)
    for j, key in enumerate(["AM","PM","Training"]):
        with scols[j]:
            st.markdown(f"**{key}**")
            items = schedule.get(key, []) if isinstance(schedule, dict) else []
            if not items:
                st.write("—")
                continue
            for pid in items:
                p = prod_map.get(pid, {})
                st.write(f"- {p.get('Ingredient', pid)} ({p.get('Brand','')})")

st.title("🦌 IBEX")
st.caption("Personalized performance systems. Not medical advice.")

products = load_products()
exclusions_df = load_exclusions()

with st.expander("Admin: data sanity check", expanded=False):
    st.write(f"Products loaded: {len(products)}")
    st.write(f"Exclusions loaded: {len(exclusions_df)}")

STRIPE_BASIC_LINK = st.secrets.get("STRIPE_BASIC_LINK", "") if hasattr(st, "secrets") else os.getenv("STRIPE_BASIC_LINK","")
STRIPE_PERF_LINK = st.secrets.get("STRIPE_PERF_LINK", "") if hasattr(st, "secrets") else os.getenv("STRIPE_PERF_LINK","")

st.header("Performance Audit")
with st.form("audit"):
    c1, c2, c3 = st.columns(3)
    with c1:
        name = st.text_input("Full name")
        email = st.text_input("Email")
        school = st.text_input("School", value="Lehigh")
    with c2:
        sport = st.text_input("Sport")
        position = st.text_input("Position / event")
        season_status = st.selectbox("Season status", ["In-season","Pre-season","Off-season"])
    with c3:
        training_days = st.slider("Training days/week", 0, 7, 5)
        intensity = st.slider("Training intensity (1–10)", 1, 10, 7)
        travel = st.selectbox("Travel frequency", ["Never","Sometimes","Often"])

    st.subheader("Goals")
    goals = st.multiselect("Select all that apply", ["strength","endurance","recovery","sleep","gut","joints","focus","general health"])

    st.subheader("Recovery & lifestyle")
    c4, c5, c6 = st.columns(3)
    with c4:
        sleep_hours = st.number_input("Sleep hours/night", min_value=0.0, max_value=12.0, value=7.0, step=0.5)
        sleep_quality = st.slider("Sleep quality (1–10)", 1, 10, 6)
    with c5:
        stress = st.slider("Stress (1–10)", 1, 10, 6)
        soreness = st.slider("Soreness/Fatigue (1–10)", 1, 10, 6)
    with c6:
        gi_sensitive = st.checkbox("GI sensitive / stomach issues", value=False)
        caffeine_sensitive = st.checkbox("Caffeine sensitive", value=False)

    st.subheader("Current stack / constraints")
    current_supps = st.text_area("What supplements are you already taking (if any)?")
    avoid_ingredients = st.text_input("Any ingredients you want to avoid?")
    open_notes = st.text_area("Anything else about your training, recovery, schedule, or concerns you want us to know?")

    consent = st.checkbox("I understand this is not medical advice and I should consult a professional for medical concerns.")
    submitted = st.form_submit_button("Build my IBEX system")

if submitted:
    if not consent:
        st.error("Please check the consent box to proceed.")
        st.stop()

    rid = str(uuid.uuid4())
    intake = {
        "rid": rid,
        "name": name,
        "email": email,
        "school": school,
        "sport": sport,
        "position": position,
        "season_status": season_status,
        "training_days_per_week": training_days,
        "intensity_1_to_10": intensity,
        "travel_frequency": travel,
        "goals": goals,
        "sleep_hours": sleep_hours,
        "sleep_quality_1_to_10": sleep_quality,
        "stress_1_to_10": stress,
        "soreness_1_to_10": soreness,
        "gi_sensitive": gi_sensitive,
        "caffeine_sensitive": caffeine_sensitive,
        "current_supplements": current_supps,
        "avoid_ingredients": avoid_ingredients,
        "open_notes": open_notes
    }

    shortlist = shortlist_products(products, goals, gi_sensitive, caffeine_sensitive)

    with st.spinner("Analyzing your audit and building your system…"):
        ai_out = run_ai(intake, shortlist, exclusions_df)

    st.success("System ready.")
    flags = ai_out.get("flags", [])
    consult = ai_out.get("consult_professional", False)
    included = ai_out.get("included_product_ids", [])
    schedule = ai_out.get("schedule", {})
    reasons = ai_out.get("reasons", {})
    notes = ai_out.get("notes_for_athlete", [])

    st.subheader("Your IBEX System")
    if consult:
        st.warning("Based on what you shared, we recommend consulting a qualified professional. We kept your plan conservative.")
    if flags:
        st.caption("Signals detected: " + ", ".join(flags))

    render_products(included, products, reasons)

    st.subheader("Schedule")
    render_schedule(schedule, products)

    st.subheader("Notes")
    for n in notes:
        st.write(f"• {n}")

    st.subheader("Checkout")
    cA, cB = st.columns(2)
    with cA:
        if STRIPE_BASIC_LINK:
            st.link_button("Subscribe — Basic", STRIPE_BASIC_LINK)
        else:
            st.info("Set STRIPE_BASIC_LINK in Streamlit secrets to enable checkout.")
    with cB:
        if STRIPE_PERF_LINK:
            st.link_button("Subscribe — Performance", STRIPE_PERF_LINK)
        else:
            st.info("Set STRIPE_PERF_LINK in Streamlit secrets to enable checkout.")

    st.caption(f"Internal reference id: {rid}")
