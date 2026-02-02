import os
import json
import uuid
import base64
from datetime import date

import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ----------------------------
# FILE PATHS (match your repo)
# ----------------------------
PRODUCTS_CSV = "data/products.csv"
EXCLUSIONS_CSV = "data/exclusions.csv"
LOGO_PNG = "assets/ibex_logo.png"
LOGO_SVG = "assets/ibex_logo.svg"  # optional

APP_TITLE = "IBEX"
APP_TAGLINE = "Personalized performance systems for athletes"

# ----------------------------
# PAGE SETUP
# ----------------------------
st.set_page_config(page_title=f"{APP_TITLE} • Performance Audit", page_icon="⚡", layout="wide")

# ----------------------------
# PREMIUM CSS
# ----------------------------
st.markdown(
    """
    <style>
      /* Hide Streamlit chrome */
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}

      /* App background + typography */
      .stApp {
        background: linear-gradient(180deg, rgba(250,250,252,1) 0%, rgba(246,247,251,1) 30%, rgba(245,246,249,1) 100%);
      }
      html, body, [class*="css"] {
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      }

      /* Cards */
      .ibx-card {
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 18px;
        padding: 18px 18px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
      }
      .ibx-card h3, .ibx-card h2, .ibx-card h1 {
        margin: 0 0 6px 0;
      }
      .ibx-muted { color: rgba(15, 23, 42, 0.62); }
      .ibx-small { font-size: 0.92rem; }
      .ibx-pill {
        display:inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(15,23,42,0.10);
        background: rgba(248,250,252,0.9);
        font-size: 12px;
        margin-right: 8px;
        margin-top: 8px;
      }
      .ibx-divider {
        height: 1px;
        background: rgba(15,23,42,0.08);
        margin: 14px 0;
      }

      /* Sidebar styling */
      section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.80);
        border-right: 1px solid rgba(15, 23, 42, 0.08);
      }
      section[data-testid="stSidebar"] .stMarkdown, 
      section[data-testid="stSidebar"] label, 
      section[data-testid="stSidebar"] span {
        color: rgba(15,23,42,0.88);
      }

      /* Buttons */
      .stButton button, .stLinkButton a {
        border-radius: 12px !important;
        padding: 0.65rem 1rem !important;
      }

      /* Make tabs look cleaner */
      button[data-baseweb="tab"] {
        border-radius: 12px 12px 0 0 !important;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# UTIL
# ----------------------------
def require_file(path: str, friendly: str):
    if not os.path.exists(path):
        st.error(f"Missing {friendly}: `{path}`")
        st.info("Fix: upload the file to your GitHub repo in the correct folder, then reboot the app.")
        st.stop()

def png_base64_img(png_path: str) -> str:
    with open(png_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def render_brand_header():
    # Crisp SVG if you have it; otherwise render PNG larger and clean
    if os.path.exists(LOGO_SVG):
        # Streamlit can't natively render SVG crisply everywhere; safest is HTML embed
        with open(LOGO_SVG, "r", encoding="utf-8") as f:
            svg = f.read().replace('<?xml version="1.0" encoding="UTF-8"?>', "").strip()
        st.markdown(
            f"""
            <div class="ibx-card" style="display:flex; align-items:center; gap:18px;">
              <div style="width:120px; height:78px; display:flex; align-items:center;">{svg}</div>
              <div>
                <div style="font-size:40px; font-weight:750; letter-spacing:0.4px;">{APP_TITLE}</div>
                <div class="ibx-muted" style="margin-top:-2px;">{APP_TAGLINE}</div>
                <div>
                  <span class="ibx-pill">AI-tailored</span>
                  <span class="ibx-pill">Athlete-first</span>
                  <span class="ibx-pill">No medical claims</span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        require_file(LOGO_PNG, "logo image (assets/ibex_logo.png)")
        png_data = png_base64_img(LOGO_PNG)
        st.markdown(
            f"""
            <div class="ibx-card" style="display:flex; align-items:center; gap:18px;">
              <img src="{png_data}" style="height:78px; width:auto;" />
              <div>
                <div style="font-size:40px; font-weight:750; letter-spacing:0.4px;">{APP_TITLE}</div>
                <div class="ibx-muted" style="margin-top:-2px;">{APP_TAGLINE}</div>
                <div>
                  <span class="ibx-pill">AI-tailored</span>
                  <span class="ibx-pill">Athlete-first</span>
                  <span class="ibx-pill">Privacy-first</span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

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

def is_yes(val) -> bool:
    return str(val).strip().lower() in {"y", "yes", "true", "1"}

# ----------------------------
# PLAN LOGIC (Option A)
# ----------------------------
BASIC_CORE_CATEGORIES = {
    "Creatine",
    "Omega-3",
    "Magnesium",
    "Vitamin D",
    "Electrolytes",
    "Protein",
    "Multivitamin",
    "Zinc",
    "Vitamin C",
    "Probiotic",
    "Fiber",
    "Collagen",
    "Tart Cherry",
}

def filter_products_by_plan(products: pd.DataFrame, plan: str) -> pd.DataFrame:
    p = products.copy()
    p["Category_norm"] = p["Category"].astype(str).str.strip()
    if plan == "Basic":
        return p[p["Category_norm"].isin(BASIC_CORE_CATEGORIES)]
    return p

def shortlist_products(products: pd.DataFrame, goals: list[str], gi_sensitive: bool, caffeine_sensitive: bool, plan: str) -> pd.DataFrame:
    p = filter_products_by_plan(products, plan)

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

    if plan == "Basic":
        p = p.assign(
            nsf=p["NSF_Certified"].apply(is_yes),
            tpt=p["Third_Party_Tested"].apply(lambda x: str(x).strip().lower() in {"y", "yes", "true", "1", "unknown"})
        ).sort_values(["nsf", "tpt"], ascending=[False, False]).drop(columns=["nsf", "tpt"])

    if len(p) < 25:
        p = filter_products_by_plan(products, plan).copy()

    cap = 55 if plan == "Basic" else 85
    return p.head(cap)

def run_ai(intake: dict, products_shortlist: pd.DataFrame, exclusions: pd.DataFrame, plan: str) -> dict:
    client = get_openai_client()

    approved_products = products_shortlist[[
        "Product_ID","Category","Ingredient","Brand","Store","Serving_Form",
        "Typical_Use","Timing","Avoid_If","Third_Party_Tested","NSF_Certified","Notes"
    ]].to_dict(orient="records")

    output_schema = {
        "flags": ["string"],
        "consult_professional": "boolean",
        "included_product_ids": ["IBX-0001"],
        "excluded_product_ids": ["IBX-0002"],
        "schedule": {"AM": ["IBX-0001"], "PM": ["IBX-0003"], "Training": ["IBX-0004"]},
        "reasons": {"IBX-0001": "short non-medical reason"},
        "notes_for_athlete": ["bullet", "bullet"]
    }

    plan_rules = (
        "Plan: BASIC. Keep it conservative and foundational. Prefer NSF Certified for Sport or third-party tested when possible. "
        "Avoid niche/experimental items. Keep the stack simple."
        if plan == "Basic"
        else
        "Plan: PERFORMANCE. Expanded optimization allowed. You may include conditional advanced items when clearly supported by intake. "
        "Still avoid excluded items and keep it non-medical."
    )

    system_prompt = (
        "You are IBEX, an assistant that organizes a personalized supplement system for athletes. "
        "You are NOT a medical provider. Do NOT diagnose, treat, or make medical claims. "
        "Only select products from approved_products. "
        "Never select anything that matches the exclusions list. "
        "If intake mentions serious symptoms, medications, or a medical condition, set consult_professional=true and keep recommendations conservative. "
        f"{plan_rules} "
        "Return ONLY valid JSON matching output_format schema."
    )

    payload = {
        "plan": plan,
        "intake": intake,
        "approved_products": approved_products,
        "exclusions": exclusions.to_dict(orient="records"),
        "output_format": output_schema
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
def render_products(product_ids: list[str], products_df: pd.DataFrame, reasons: dict):
    prod_map = products_df.set_index("Product_ID").to_dict(orient="index")

    # Card grid
    cols = st.columns(3, gap="large")
    for i, pid in enumerate(product_ids):
        p = prod_map.get(pid)
        if not p:
            continue
        with cols[i % 3]:
            link = str(p.get("Link", "")).strip()
            badge_left = f"{p['Category']}"
            badge_right = f"{p['Timing']}"

            st.markdown(
                f"""
                <div class="ibx-card">
                  <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div class="ibx-pill">{badge_left}</div>
                    <div class="ibx-pill">{badge_right}</div>
                  </div>
                  <h3 style="margin-top:10px;">{p['Ingredient']}</h3>
                  <div class="ibx-muted ibx-small">{p['Brand']} • {p['Serving_Form']} • {p['Store']}</div>
                  <div class="ibx-divider"></div>
                  <div style="font-weight:650;">Why</div>
                  <div class="ibx-muted">{reasons.get(pid, "Personalized to your audit")}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if link.startswith("http"):
                st.link_button("View product", link)

def render_schedule(schedule: dict, products_df: pd.DataFrame):
    prod_map = products_df.set_index("Product_ID").to_dict(orient="index")

    blocks = [("AM", "Morning"), ("PM", "Evening"), ("Training", "Training")]
    cols = st.columns(3, gap="large")
    for i, (key, title) in enumerate(blocks):
        items = schedule.get(key, []) if isinstance(schedule, dict) else []
        with cols[i]:
            st.markdown(
                f"""
                <div class="ibx-card">
                  <h3 style="margin:0;">{title}</h3>
                  <div class="ibx-muted ibx-small">Recommended timing</div>
                  <div class="ibx-divider"></div>
                """,
                unsafe_allow_html=True
            )
            if not items:
                st.write("—")
            else:
                for pid in items:
                    p = prod_map.get(pid, {})
                    st.write(f"- **{p.get('Ingredient', pid)}** — {p.get('Brand','')}")
            st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# PRIVACY POLICY CONTENT
# ----------------------------
def render_privacy_policy():
    today = date.today().strftime("%B %d, %Y")
    st.markdown(
        f"""
        <div class="ibx-card">
          <h2 style="margin:0;">Privacy Policy</h2>
          <div class="ibx-muted ibx-small">Effective date: {today}</div>
          <div class="ibx-divider"></div>

          <p><strong>IBEX</strong> (“IBEX,” “we,” “us”) provides an athlete-focused performance audit and personalized supplement organization experience.
          This Privacy Policy explains what we collect, how we use it, and the choices you have.</p>

          <h3>1) What we collect</h3>
          <ul>
            <li><strong>Information you provide</strong>: questionnaire responses (e.g., training, sleep, stress), and contact info (email, name if provided).</li>
            <li><strong>Checkout information</strong>: if you subscribe, payments and shipping details are collected and processed by <strong>Stripe</strong>. We do not store your full card number.</li>
            <li><strong>Basic technical data</strong>: standard logs (e.g., approximate device/browser info) may be processed by our hosting provider for reliability and security.</li>
          </ul>

          <h3>2) How we use information</h3>
          <ul>
            <li>To generate your personalized recommendations and schedule.</li>
            <li>To operate subscriptions and fulfill shipments (if you purchase).</li>
            <li>To provide support and respond to messages.</li>
            <li>To improve the product (aggregated, non-identifying insights where possible).</li>
          </ul>

          <h3>3) AI processing</h3>
          <p>Your questionnaire inputs are sent to an AI model to generate recommendations. We do not intend for you to submit medical records or highly sensitive personal data.
          <strong>IBEX is not a medical service</strong>, and recommendations are informational/organizational only.</p>

          <h3>4) What we do NOT do</h3>
          <ul>
            <li>We do <strong>not</strong> sell your personal information.</li>
            <li>We do <strong>not</strong> share your information with third parties for their marketing.</li>
          </ul>

          <h3>5) Who we share data with</h3>
          <ul>
            <li><strong>Stripe</strong> (payments and subscription management).</li>
            <li><strong>Service providers</strong> needed to run the site and deliver the service (e.g., hosting).</li>
          </ul>

          <h3>6) Data retention</h3>
          <p>We keep data only as long as needed to provide the service, meet legal obligations, and resolve disputes. You can request deletion (see below).</p>

          <h3>7) Your choices</h3>
          <ul>
            <li>You can request access, correction, or deletion of your data by contacting us.</li>
            <li>You can cancel your subscription at any time through Stripe’s customer portal (if enabled) or by contacting support.</li>
          </ul>

          <h3>8) Security</h3>
          <p>We use reasonable safeguards to protect information. No method of transmission or storage is 100% secure.</p>

          <h3>9) Contact</h3>
          <p>Email: <strong>support@ibexperformance.com</strong> (replace with your real support email)<br/>
          If you don’t have a domain email yet, use your Gmail for now and update later.</p>

          <div class="ibx-divider"></div>
          <div class="ibx-muted ibx-small">
            This template is provided for startup MVP use and is not legal advice. If you scale, have counsel review it.
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_faq():
    st.markdown(
        """
        <div class="ibx-card">
          <h2 style="margin:0;">FAQ</h2>
          <div class="ibx-divider"></div>
          <h3>Is this medical advice?</h3>
          <p class="ibx-muted">No. IBEX is informational and organizational. Consult a qualified professional for medical concerns.</p>
          <h3>What’s the difference between Basic and Performance?</h3>
          <p class="ibx-muted"><strong>Basic</strong> uses a conservative, core-only supplement catalog. <strong>Performance</strong> unlocks a larger catalog and conditional optimization.</p>
          <h3>Do you sell my data?</h3>
          <p class="ibx-muted">No.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------
# APP START
# ----------------------------
require_file(PRODUCTS_CSV, "products.csv (put it in /data)")
require_file(EXCLUSIONS_CSV, "exclusions.csv (put it in /data)")
render_brand_header()

# Load data
try:
    products = load_products()
    exclusions = load_exclusions()
except Exception as e:
    st.error(f"Data load error: {e}")
    st.stop()

# Secrets
STRIPE_BASIC_LINK = st.secrets.get("STRIPE_BASIC_LINK", "")
STRIPE_PERF_LINK = st.secrets.get("STRIPE_PERF_LINK", "")

# Session state for no-scroll UX
if "ai_out" not in st.session_state:
    st.session_state.ai_out = None
if "last_plan" not in st.session_state:
    st.session_state.last_plan = "Basic"
if "last_rid" not in st.session_state:
    st.session_state.last_rid = None

tabs = st.tabs(["Audit", "Privacy", "FAQ"])

# ----------------------------
# TAB 1: AUDIT
# ----------------------------
with tabs[0]:
    # Results area always at the top (no scroll needed)
    if st.session_state.ai_out:
        ai_out = st.session_state.ai_out
        plan = st.session_state.last_plan

        st.markdown(
            f"""
            <div class="ibx-card">
              <h2 style="margin:0;">Your {plan} System</h2>
              <div class="ibx-muted ibx-small">Reference ID: {st.session_state.last_rid}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("")

        if ai_out.get("consult_professional", False):
            st.warning("Based on what you shared, consult a qualified professional. We kept this conservative.")
        flags = ai_out.get("flags", [])
        if flags:
            st.caption("Signals detected: " + ", ".join(flags))

        st.subheader("Recommended Stack")
        render_products(ai_out.get("included_product_ids", []), products, ai_out.get("reasons", {}))

        st.subheader("Schedule")
        render_schedule(ai_out.get("schedule", {}), products)

        st.subheader("Notes")
        for n in ai_out.get("notes_for_athlete", []):
            st.write(f"• {n}")

        st.subheader("Checkout")
        if plan == "Basic":
            c1, c2 = st.columns([1, 1], gap="large")
            with c1:
                st.markdown("<div class='ibx-card'><h3 style='margin:0;'>IBEX Basic</h3><div class='ibx-muted'>Core, conservative system</div></div>", unsafe_allow_html=True)
                if STRIPE_BASIC_LINK:
                    st.link_button("Subscribe — IBEX Basic", STRIPE_BASIC_LINK)
                else:
                    st.info("Set STRIPE_BASIC_LINK in Streamlit Secrets.")
            with c2:
                st.markdown("<div class='ibx-card'><h3 style='margin:0;'>IBEX Performance</h3><div class='ibx-muted'>Expanded optimization</div></div>", unsafe_allow_html=True)
                if STRIPE_PERF_LINK:
                    st.link_button("Upgrade — IBEX Performance", STRIPE_PERF_LINK)
                else:
                    st.info("Set STRIPE_PERF_LINK in Streamlit Secrets.")
        else:
            st.markdown("<div class='ibx-card'><h3 style='margin:0;'>IBEX Performance</h3><div class='ibx-muted'>Expanded optimization</div></div>", unsafe_allow_html=True)
            if STRIPE_PERF_LINK:
                st.link_button("Subscribe — IBEX Performance", STRIPE_PERF_LINK)
            else:
                st.info("Set STRIPE_PERF_LINK in Streamlit Secrets.")

        st.write("")
        if st.button("Start a new audit"):
            st.session_state.ai_out = None
            st.session_state.last_rid = None
            st.rerun()

    else:
        st.markdown(
            """
            <div class="ibx-card">
              <h2 style="margin:0;">Performance Audit</h2>
              <div class="ibx-muted">Complete the audit in the sidebar. Your results appear here instantly.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("")
        st.info("Tip: Choose a plan in the sidebar first. The plan changes what the AI is allowed to recommend.")

    # Sidebar = “app-like” flow; results stay in main
    with st.sidebar:
        st.markdown(f"### {APP_TITLE} Audit")
        st.caption("Plan → Audit → Instant system. No scrolling for results.")

        plan = st.radio(
            "Choose your plan",
            ["Basic", "Performance"],
            index=0 if st.session_state.last_plan == "Basic" else 1,
            horizontal=True
        )

        st.markdown("**Plan meaning**")
        if plan == "Basic":
            st.write("Core, conservative stack. Prefer tested/certified when possible.")
        else:
            st.write("Expanded catalog + conditional optimization.")

        st.markdown("---")

        with st.form("audit_form"):
            st.markdown("#### About you")
            name = st.text_input("Full name", value="")
            email = st.text_input("Email", value="")
            school = st.text_input("School", value="Lehigh")

            st.markdown("#### Sport & training")
            sport = st.text_input("Sport", value="")
            position = st.text_input("Position / Event", value="")
            season_status = st.selectbox("Season status", ["In-season", "Pre-season", "Off-season"])
            training_days = st.slider("Training days/week", 0, 7, 5)
            intensity = st.slider("Training intensity (1–10)", 1, 10, 7)
            travel = st.selectbox("Travel frequency", ["Never", "Sometimes", "Often"])

            st.markdown("#### Goals")
            goals = st.multiselect(
                "Select all that apply",
                ["strength","endurance","recovery","sleep","gut","joints","focus","general health"]
            )

            st.markdown("#### Recovery & lifestyle")
            sleep_hours = st.number_input("Sleep hours/night", min_value=0.0, max_value=12.0, value=7.0, step=0.5)
            sleep_quality = st.slider("Sleep quality (1–10)", 1, 10, 6)
            stress = st.slider("Stress (1–10)", 1, 10, 6)
            soreness = st.slider("Soreness/Fatigue (1–10)", 1, 10, 6)
            gi_sensitive = st.checkbox("GI sensitive / stomach issues", value=False)
            caffeine_sensitive = st.checkbox("Caffeine sensitive", value=False)

            st.markdown("#### Current stack / notes")
            current_supps = st.text_area("Supplements you already take (optional)", placeholder="Creatine, fish oil, whey…")
            avoid_ingredients = st.text_input("Ingredients to avoid (optional)", placeholder="e.g., caffeine")
            open_notes = st.text_area(
                "Other concerns or context (optional)",
                placeholder="Anything that would help tailor the plan…"
            )

            st.markdown("---")
            st.caption("By continuing, you agree to our Privacy Policy (see Privacy tab).")
            consent = st.checkbox("I understand this is not medical advice.", value=False)

            submitted = st.form_submit_button("Build my system")

        if submitted:
            if not consent:
                st.error("Please check the consent box to proceed.")
                st.stop()

            rid = str(uuid.uuid4())
            intake = {
                "rid": rid,
                "plan": plan,
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

            shortlist = shortlist_products(products, goals, gi_sensitive, caffeine_sensitive, plan)

            with st.spinner("Generating your system…"):
                try:
                    client = get_openai_client()
                    ai_out = run_ai(intake, shortlist, exclusions, plan)
                except Exception as e:
                    st.error(f"AI error: {e}")
                    st.info("Check OPENAI_API_KEY in Streamlit Secrets and confirm OpenAI billing is enabled.")
                    st.stop()

            # Save results and rerun so user is at top (no scrolling)
            st.session_state.ai_out = ai_out
            st.session_state.last_plan = plan
            st.session_state.last_rid = rid
            st.rerun()

# ----------------------------
# TAB 2: PRIVACY
# ----------------------------
with tabs[1]:
    render_privacy_policy()

# ----------------------------
# TAB 3: FAQ
# ----------------------------
with tabs[2]:
    render_faq()


