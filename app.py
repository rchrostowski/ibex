import os
import json
import uuid
import base64
import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ----------------------------
# PATHS (match your repo)
# ----------------------------
PRODUCTS_CSV = "data/products.csv"
EXCLUSIONS_CSV = "data/exclusions.csv"
LOGO_PNG = "assets/ibex_logo.png"
LOGO_SVG = "assets/ibex_logo.svg"  # optional if you upload later

APP_TITLE = "IBEX Performance Audit"

# ----------------------------
# PAGE SETUP
# ----------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="⚡", layout="wide")

# Hide Streamlit chrome for cleaner look
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# UTIL: file checks + SVG rendering
# ----------------------------
def require_file(path: str, friendly: str):
    if not os.path.exists(path):
        st.error(f"Missing {friendly}: `{path}`")
        st.info("Fix: upload the file to your GitHub repo in the correct folder, then reboot the app.")
        st.stop()

def svg_as_html(svg_path: str, height_px: int = 90) -> str:
    """Embed an SVG from disk into HTML for crisp scaling."""
    with open(svg_path, "r", encoding="utf-8") as f:
        svg = f.read()
    # Strip XML header if present (safer in HTML)
    svg = svg.replace('<?xml version="1.0" encoding="UTF-8"?>', "").strip()
    return f"<div style='height:{height_px}px; width:auto;'>{svg}</div>"

def png_base64_img(png_path: str) -> str:
    """Inline PNG as base64 in HTML to control crisp sizing."""
    with open(png_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"

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
        raise ValueError(f"products.csv missing columns: {missing}")
    return df

@st.cache_data(show_spinner=False)
def load_exclusions():
    df = pd.read_csv(EXCLUSIONS_CSV)
    df.columns = [c.strip() for c in df.columns]
    if "Excluded_Category_or_Ingredient" not in df.columns or "Reason" not in df.columns:
        raise ValueError("exclusions.csv must have columns: Excluded_Category_or_Ingredient, Reason")
    return df

# ----------------------------
# OPENAI
# ----------------------------
def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        st.error("Missing OPENAI_API_KEY in Streamlit Secrets.")
        st.info("Manage app → Settings → Secrets, then add OPENAI_API_KEY.")
        st.stop()
    if OpenAI is None:
        st.error("openai package not installed. Ensure requirements.txt includes `openai`.")
        st.stop()
    return OpenAI(api_key=api_key)

def shortlist_products(products: pd.DataFrame, goals: list[str], gi_sensitive: bool, caffeine_sensitive: bool) -> pd.DataFrame:
    """
    Keeps 'unlimited personalization' while keeping the AI payload manageable.
    Returns up to ~70 relevant products.
    """
    p = products.copy()

    # Goal match
    if goals:
        mask = False
        for g in goals:
            mask = mask | p["Typical_Use"].astype(str).str.contains(g, case=False, na=False)
        if mask is not False:
            p = p[mask]

    # Soft filters for sensitivities
    if gi_sensitive:
        p = p[~p["Avoid_If"].astype(str).str.contains("GI", case=False, na=False)]
    if caffeine_sensitive:
        p = p[~p["Avoid_If"].astype(str).str.contains("caffeine", case=False, na=False)]

    if len(p) < 25:
        p = products.copy()

    return p.head(70)

def run_ai(intake: dict, products_shortlist: pd.DataFrame, exclusions: pd.DataFrame) -> dict:
    client = get_openai_client()

    approved_products = products_shortlist[[
        "Product_ID","Category","Ingredient","Brand","Store","Serving_Form",
        "Typical_Use","Timing","Avoid_If","Third_Party_Tested","NSF_Certified","Notes"
    ]].to_dict(orient="records")

    schema = {
        "flags": ["string"],
        "consult_professional": "boolean",
        "included_product_ids": ["IBX-0001"],
        "excluded_product_ids": ["IBX-0002"],
        "schedule": {"AM": ["IBX-0001"], "PM": ["IBX-0003"], "Training": ["IBX-0004"]},
        "reasons": {"IBX-0001": "short non-medical reason"},
        "notes_for_athlete": ["bullet", "bullet"]
    }

    system_prompt = (
        "You are IBEX, an assistant that organizes a personalized supplement system for athletes. "
        "You are NOT a medical provider. Do NOT diagnose, treat, or make medical claims. "
        "Only select products from approved_products. "
        "Never select anything that matches the exclusions list. "
        "If the intake mentions serious symptoms, medication, or a medical condition, set consult_professional=true and keep recommendations conservative. "
        "Return ONLY valid JSON matching the output_format schema."
    )

    payload = {
        "intake": intake,
        "approved_products": approved_products,
        "exclusions": exclusions.to_dict(orient="records"),
        "output_format": schema
    }

    model = st.secrets.get("OPENAI_MODEL", "gpt-4.1-mini")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload)}
        ],
        temperature=0.2
    )

    content = resp.choices[0].message.content.strip()

    # Robust parse (in case it wraps JSON)
    try:
        return json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start:end+1])
        raise

# ----------------------------
# RENDER HELPERS
# ----------------------------
def render_header():
    # Prefer SVG if present (crisp)
    if os.path.exists(LOGO_SVG):
        logo_html = svg_as_html(LOGO_SVG, height_px=92)
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:18px; margin-top:6px; margin-bottom:6px;">
              <div style="width:110px;">{logo_html}</div>
              <div>
                <h1 style="margin:0; font-size:42px; letter-spacing:0.5px;">IBEX</h1>
                <div style="margin-top:4px; color:#6b7280; font-size:15px;">
                  Personalized performance systems for athletes
                </div>
              </div>
            </div>
            <hr style="margin-top:12px; margin-bottom:18px;"/>
            """,
            unsafe_allow_html=True
        )
    else:
        # PNG fallback (bigger, sharper)
        require_file(LOGO_PNG, "logo image")
        png_data = png_base64_img(LOGO_PNG)
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:18px; margin-top:6px; margin-bottom:6px;">
              <img src="{png_data}" style="height:96px; width:auto; image-rendering:auto;" />
              <div>
                <h1 style="margin:0; font-size:42px; letter-spacing:0.5px;">IBEX</h1>
                <div style="margin-top:4px; color:#6b7280; font-size:15px;">
                  Personalized performance systems for athletes
                </div>
              </div>
            </div>
            <hr style="margin-top:12px; margin-bottom:18px;"/>
            """,
            unsafe_allow_html=True
        )

def render_products(product_ids: list[str], products_df: pd.DataFrame, reasons: dict):
    prod_map = products_df.set_index("Product_ID").to_dict(orient="index")
    cols = st.columns(3)
    for i, pid in enumerate(product_ids):
        p = prod_map.get(pid)
        if not p:
            continue
        with cols[i % 3]:
            st.markdown(f"### {p['Brand']} — {p['Ingredient']}")
            st.write(f"**Category:** {p['Category']}")
            st.write(f"**Store:** {p['Store']}")
            st.write(f"**Form:** {p['Serving_Form']}")
            st.write(f"**Timing (default):** {p['Timing']}")
            st.write(f"**Why:** {reasons.get(pid, 'Personalized to your audit')}")
            link = str(p.get("Link", "")).strip()
            if link.startswith("http"):
                st.link_button("View product", link)
            st.divider()

def render_schedule(schedule: dict, products_df: pd.DataFrame):
    prod_map = products_df.set_index("Product_ID").to_dict(orient="index")
    scols = st.columns(3)
    for j, key in enumerate(["AM", "PM", "Training"]):
        with scols[j]:
            st.markdown(f"**{key}**")
            items = schedule.get(key, []) if isinstance(schedule, dict) else []
            if not items:
                st.write("—")
                continue
            for pid in items:
                p = prod_map.get(pid, {})
                st.write(f"- {p.get('Ingredient', pid)} ({p.get('Brand', '')})")

# ----------------------------
# START APP
# ----------------------------
# Make sure required files exist
require_file(PRODUCTS_CSV, "products.csv (put it in /data)")
require_file(EXCLUSIONS_CSV, "exclusions.csv (put it in /data)")

render_header()

# Load data
try:
    products = load_products()
    exclusions = load_exclusions()
except Exception as e:
    st.error(f"Data load error: {e}")
    st.stop()

with st.expander("Admin: sanity check", expanded=False):
    st.write(f"Products loaded: {len(products)}")
    st.write(f"Exclusions loaded: {len(exclusions)}")

# Checkout links from secrets
STRIPE_BASIC_LINK = st.secrets.get("STRIPE_BASIC_LINK", "")
STRIPE_PERF_LINK = st.secrets.get("STRIPE_PERF_LINK", "")

st.header("Performance Audit")
st.caption("Not medical advice. For educational and organizational purposes only.")

with st.form("audit"):
    c1, c2, c3 = st.columns(3)
    with c1:
        name = st.text_input("Full name")
        email = st.text_input("Email")
        school = st.text_input("School", value="Lehigh")
    with c2:
        sport = st.text_input("Sport")
        position = st.text_input("Position / Event")
        season_status = st.selectbox("Season status", ["In-season", "Pre-season", "Off-season"])
    with c3:
        training_days = st.slider("Training days/week", 0, 7, 5)
        intensity = st.slider("Training intensity (1–10)", 1, 10, 7)
        travel = st.selectbox("Travel frequency", ["Never", "Sometimes", "Often"])

    st.subheader("Goals")
    goals = st.multiselect(
        "Select all that apply",
        ["strength","endurance","recovery","sleep","gut","joints","focus","general health"]
    )

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
    current_supps = st.text_area("What supplements are you already taking (if any)?", placeholder="Creatine, fish oil, whey…")
    avoid_ingredients = st.text_input("Any ingredients you want to avoid?", placeholder="e.g., caffeine, dairy, artificial sweeteners")
    open_notes = st.text_area(
        "Anything else about your training, recovery, schedule, or concerns you want us to know?",
        placeholder="Midterms crush my sleep… preworkouts mess my stomach up…"
    )

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
        try:
            ai_out = run_ai(intake, shortlist, exclusions)
        except Exception as e:
            st.error(f"AI error: {e}")
            st.info("Fix: ensure OPENAI_API_KEY is set in Streamlit Secrets and billing is enabled.")
            st.stop()

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
    st.write("Choose a plan to start your monthly subscription.")
    cA, cB = st.columns(2)
    with cA:
        if STRIPE_BASIC_LINK:
            st.link_button("Subscribe — Basic", STRIPE_BASIC_LINK)
        else:
            st.info("Set STRIPE_BASIC_LINK in Streamlit Secrets to enable checkout.")
    with cB:
        if STRIPE_PERF_LINK:
            st.link_button("Subscribe — Performance", STRIPE_PERF_LINK)
        else:
            st.info("Set STRIPE_PERF_LINK in Streamlit Secrets to enable checkout.")

    st.caption(f"Internal reference id: {rid}")

