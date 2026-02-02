import os
import json
import uuid
import base64
from datetime import date, datetime

import pandas as pd
import streamlit as st

# Optional: logo support
try:
    from PIL import Image
except Exception:
    Image = None

# OpenAI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Stripe (optional)
try:
    import stripe
except Exception:
    stripe = None

# Supabase (optional)
try:
    from supabase import create_client
except Exception:
    create_client = None


# =========================================================
# CONFIG / PATHS
# =========================================================
APP_TITLE = "IBEX"
APP_TAGLINE = "Personalized performance systems for athletes"

PRODUCTS_CSV = "data/products.csv"
EXCLUSIONS_CSV = "data/exclusions.csv"

# Put a HIGH-RES square logo here (512x512 or 1024x1024)
LOGO_PATH = "assets/ibex_logo.png"

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title=f"{APP_TITLE} • Performance Audit",
    page_icon=LOGO_PATH,  # favicon/browser tab (works if file exists)
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PREMIUM STYLING (includes FIXES for sidebar dropdown text)
# =========================================================
st.markdown(
    """
<style>
:root{
  --bg:#f6f7fb;
  --card:#ffffff;
  --text:#0f172a;
  --sub:#334155;
  --muted:#64748b;
  --border:#e5e7eb;
  --accent:#ef4444;
  --accent2:#111827;

  --side:#0b1220;
  --sideBorder:#132033;
  --sideText:#e5e7eb;
}

/* hide streamlit chrome */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.stApp{ background: var(--bg); }
html, body, [class*="css"]{
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
}

/* Main text */
h1,h2,h3,h4,h5{ color:var(--text) !important; letter-spacing:-0.2px; }
p,li,span,div,label{ color:var(--sub); }

/* Sidebar */
section[data-testid="stSidebar"]{
  background: var(--side);
  border-right:1px solid var(--sideBorder);
}
section[data-testid="stSidebar"] *{
  color: var(--sideText) !important;
}
section[data-testid="stSidebar"] a{ color:#93c5fd !important; }

/* Inputs (main area) */
input, textarea, select {
  background:#fff !important;
  color:var(--text) !important;
  border:1px solid var(--border) !important;
  border-radius:12px !important;
}

/* Tabs */
button[data-baseweb="tab"]{
  color: var(--sub) !important;
  font-weight: 700;
  border-radius: 12px 12px 0 0 !important;
}
button[data-baseweb="tab"][aria-selected="true"]{
  color: var(--accent) !important;
  border-bottom: 3px solid var(--accent) !important;
}

/* Cards */
.ibx-card{
  background: var(--card);
  border:1px solid rgba(15, 23, 42, 0.08);
  border-radius: 22px;
  padding: 28px;
  box-shadow: 0 18px 45px rgba(2, 6, 23, 0.08);
  margin-bottom: 18px;
}
.ibx-muted{ color: var(--muted) !important; }
.ibx-badge{
  display:inline-block;
  padding: 6px 10px;
  border-radius:999px;
  border:1px solid rgba(15,23,42,0.10);
  background: rgba(248,250,252,1);
  font-size: 12px;
  margin-right:8px;
  color: var(--sub) !important;
}
.ibx-divider{
  height:1px;
  background: rgba(15,23,42,0.08);
  margin: 14px 0;
}

/* Buttons */
.stButton button, .stLinkButton a{
  border-radius: 14px !important;
  padding: 0.78rem 1.05rem !important;
  font-weight: 850 !important;
  color:#ffffff !important;
}
.stButton button{
  background: var(--accent) !important;
  border: none !important;
}
.stButton button:hover{ opacity: 0.92; }
.stLinkButton a{
  background: var(--accent2) !important;
  border: 1px solid rgba(17,24,39,0.15) !important;
  color: #fff !important;
}
.stLinkButton a:hover{ opacity:0.92; }

/* Reduce extra whitespace above */
.block-container{ padding-top: 1.0rem; }

/* ---------------------------------------------------------
   SIDEBAR INPUT + DROPDOWN TEXT COLOR FIXES
--------------------------------------------------------- */
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stTextArea textarea,
section[data-testid="stSidebar"] .stNumberInput input{
  background:#ffffff !important;
  color: var(--text) !important;
  border:1px solid rgba(229,231,235,0.35) !important;
  border-radius: 14px !important;
}

section[data-testid="stSidebar"] .stTextInput input::placeholder,
section[data-testid="stSidebar"] .stTextArea textarea::placeholder,
section[data-testid="stSidebar"] .stNumberInput input::placeholder{
  color: var(--muted) !important;
  opacity: 1 !important;
}

/* BaseWeb select boxes */
section[data-testid="stSidebar"] [data-baseweb="select"] > div{
  background:#ffffff !important;
  border:1px solid rgba(229,231,235,0.35) !important;
  border-radius: 14px !important;
}
/* Selected value */
section[data-testid="stSidebar"] [data-baseweb="select"] *{
  color: var(--text) !important;
}
/* caret */
section[data-testid="stSidebar"] [data-baseweb="select"] svg{
  color: var(--text) !important;
}

/* Dropdown menu (portal) */
div[data-baseweb="menu"]{
  background:#ffffff !important;
  border:1px solid rgba(15,23,42,0.10) !important;
  border-radius: 14px !important;
  overflow:hidden !important;
}
div[data-baseweb="menu"] *{
  color: var(--text) !important;
}
div[data-baseweb="menu"] [role="option"]:hover{
  background: rgba(15,23,42,0.06) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# HELPERS
# =========================================================
def require_file(path: str, friendly: str):
    if not os.path.exists(path):
        st.error(f"Missing {friendly}: `{path}`")
        st.info("Fix: upload the file to your GitHub repo in the correct folder, then reboot the app.")
        st.stop()

def load_logo():
    if not os.path.exists(LOGO_PATH) or Image is None:
        return None
    try:
        return Image.open(LOGO_PATH)
    except Exception:
        return None

def new_rid() -> str:
    return str(uuid.uuid4())

def get_query_params() -> dict:
    # Works across Streamlit versions
    try:
        return dict(st.query_params)
    except Exception:
        return st.experimental_get_query_params()

def set_query_params(**kwargs):
    try:
        st.query_params.clear()
        for k, v in kwargs.items():
            st.query_params[k] = v
    except Exception:
        st.experimental_set_query_params(**kwargs)

def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY in Streamlit Secrets.")
        st.stop()
    if OpenAI is None:
        st.error("openai package not installed. Add `openai` to requirements.txt.")
        st.stop()
    return OpenAI(api_key=api_key)

def is_yes(val) -> bool:
    return str(val).strip().lower() in {"y", "yes", "true", "1"}

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

def supabase_client():
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        return None
    if create_client is None:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None

def stripe_enabled():
    return stripe is not None and bool(st.secrets.get("STRIPE_SECRET_KEY", ""))

def stripe_init():
    stripe.api_key = st.secrets.get("STRIPE_SECRET_KEY")

def stripe_retrieve_checkout_session(session_id: str):
    """
    Stripe expansion gotcha:
    - shipping_details is NOT expandable (it is returned inline)
    - line_items requires list_line_items OR retrieve with expand=['line_items']
    """
    stripe_init()
    # Retrieve session (shipping_details will already be present if collected)
    sess = stripe.checkout.Session.retrieve(
        session_id,
        expand=["customer", "subscription"]  # keep simple; line_items via separate call for reliability
    )
    line_items = stripe.checkout.Session.list_line_items(session_id, limit=20)
    return sess, line_items

def extract_audit_id_from_session(sess: dict) -> str | None:
    """
    Payment Links can collect a custom field for 'IBEX Audit ID'.
    Stripe returns that in sess['custom_fields'] (array).
    We'll scan for anything that looks like a UUID.
    """
    try:
        cfs = sess.get("custom_fields") or []
        for f in cfs:
            val = None
            if "text" in f and isinstance(f["text"], dict):
                val = f["text"].get("value")
            if isinstance(val, str) and len(val) >= 32 and "-" in val:
                return val.strip()
    except Exception:
        pass
    return None

def safe_json(obj):
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return {}

# =========================================================
# PLAN COPY
# =========================================================
PLAN_COPY = {
    "Basic": {
        "headline": "Foundations, done right.",
        "sub": "A clean, conservative essentials stack built for consistency and safety.",
        "bullets": [
            "Core performance stack (the boring stuff that actually works)",
            "Strong preference for third-party tested / certified options",
            "Simple, repeatable schedule built around training",
        ],
        "note": "Best for: most athletes who want a safe baseline.",
    },
    "Performance": {
        "headline": "Optimization mode.",
        "sub": "A deeper system with conditional additions based on your training load and audit signals.",
        "bullets": [
            "Expanded catalog (sleep, recovery, gut, joints when clearly needed)",
            "More precise timing (training vs off days)",
            "Built for athletes chasing marginal gains (without sketchy stuff)",
        ],
        "note": "Best for: high-volume training, high stress, or athletes who want every edge.",
    },
}

BASIC_CORE_CATEGORIES = {
    "Creatine","Omega-3","Magnesium","Vitamin D","Electrolytes","Protein",
    "Multivitamin","Zinc","Vitamin C","Probiotic","Fiber","Collagen","Tart Cherry"
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
            tpt=p["Third_Party_Tested"].apply(lambda x: str(x).strip().lower() in {"y","yes","true","1","unknown"})
        ).sort_values(["nsf","tpt"], ascending=[False, False]).drop(columns=["nsf","tpt"])

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
        "Plan: BASIC. Conservative and foundational. Keep stack simple. Prefer NSF/third-party tested. Avoid niche/experimental items."
        if plan == "Basic"
        else
        "Plan: PERFORMANCE. Expanded optimization. You may add conditional advanced items if clearly supported by intake. Still conservative on risk."
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

    content = (resp.choices[0].message.content or "").strip()
    try:
        return json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start:end+1])
        raise

def render_header():
    logo = load_logo()
    if logo is not None:
        c1, c2 = st.columns([1, 7], gap="large")
        with c1:
            st.image(logo, width=140)
        with c2:
            st.markdown(
                f"<div style='font-size:46px; font-weight:900; color:#0f172a; margin-top:2px;'>{APP_TITLE}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='ibx-muted' style='font-size:16px; margin-top:-6px;'>{APP_TAGLINE}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="margin-top:10px;">
                  <span class="ibx-badge">Plan-aware AI</span>
                  <span class="ibx-badge">Privacy-first</span>
                  <span class="ibx-badge">Athlete-safe guardrails</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f"""
            <div class="ibx-card">
              <div style="font-size:46px; font-weight:900; color:#0f172a;">{APP_TITLE}</div>
              <div class="ibx-muted" style="font-size:16px;">{APP_TAGLINE}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def render_products(product_ids: list[str], products_df: pd.DataFrame, reasons: dict):
    prod_map = products_df.set_index("Product_ID").to_dict(orient="index")
    cols = st.columns(3, gap="large")
    for i, pid in enumerate(product_ids):
        p = prod_map.get(pid)
        if not p:
            continue
        with cols[i % 3]:
            st.markdown("<div class='ibx-card'>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap;">
                  <span class="ibx-badge">{p['Category']}</span>
                  <span class="ibx-badge">{p['Timing']}</span>
                </div>
                <div style="margin-top:12px; font-size:18px; font-weight:900; color:#0f172a;">
                  {p['Ingredient']}
                </div>
                <div class="ibx-muted" style="margin-top:2px;">
                  {p['Brand']} • {p['Serving_Form']} • {p['Store']}
                </div>
                <div class="ibx-divider"></div>
                <div style="font-weight:900; color:#0f172a;">Why this</div>
                <div class="ibx-muted" style="margin-top:4px;">
                  {reasons.get(pid, "Personalized to your audit")}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

def render_schedule(schedule: dict, products_df: pd.DataFrame):
    prod_map = products_df.set_index("Product_ID").to_dict(orient="index")
    blocks = [("AM","Morning"), ("Training","Training"), ("PM","Evening")]
    cols = st.columns(3, gap="large")
    for i, (key, title) in enumerate(blocks):
        with cols[i]:
            st.markdown("<div class='ibx-card'>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:18px; font-weight:950; color:#0f172a;'>{title}</div>", unsafe_allow_html=True)
            st.markdown("<div class='ibx-muted' style='margin-top:-2px;'>Recommended timing</div>", unsafe_allow_html=True)
            st.markdown("<div class='ibx-divider'></div>", unsafe_allow_html=True)
            items = schedule.get(key, []) if isinstance(schedule, dict) else []
            if not items:
                st.markdown("<div class='ibx-muted'>—</div>", unsafe_allow_html=True)
            else:
                for pid in items:
                    p = prod_map.get(pid, {})
                    st.markdown(f"- **{p.get('Ingredient', pid)}** — {p.get('Brand','')}")
            st.markdown("</div>", unsafe_allow_html=True)

def render_privacy_policy():
    eff = date.today().strftime("%B %d, %Y")
    st.markdown(
        f"""
<div class="ibx-card">
  <div style="font-size:26px; font-weight:950; color:#0f172a;">Privacy Policy</div>
  <div class="ibx-muted" style="margin-top:4px;">Effective: {eff}</div>
  <div class="ibx-divider"></div>

  <p><strong>IBEX</strong> (“IBEX,” “we,” “us”) provides an athlete-focused performance audit and personalized supplement organization experience.
  This Privacy Policy explains what we collect, how we use it, and your choices.</p>

  <h3>What we collect</h3>
  <ul>
    <li><strong>Audit inputs</strong> you choose to provide (training, sleep, stress, preferences).</li>
    <li><strong>Contact info</strong> (email, name if provided).</li>
    <li><strong>Checkout info</strong> (shipping + payment) processed by <strong>Stripe</strong>. We do not store your full card number.</li>
  </ul>

  <h3>How we use it</h3>
  <ul>
    <li>Generate your personalized system and schedule.</li>
    <li>Operate subscriptions and fulfill shipments (if you purchase).</li>
    <li>Provide support and improve the product (prefer aggregated insights when possible).</li>
  </ul>

  <h3>What we do NOT do</h3>
  <ul>
    <li>We do <strong>not</strong> sell your personal data.</li>
    <li>We do <strong>not</strong> share your data with third parties for their marketing.</li>
  </ul>

  <h3>AI processing</h3>
  <p>Your audit inputs are sent to an AI model to generate recommendations. IBEX is not a medical service and does not provide medical advice.</p>

  <h3>Retention & deletion</h3>
  <p>We retain data only as long as needed to provide the service and meet legal obligations. You can request deletion at any time.</p>

  <h3>Contact</h3>
  <p>Email: <strong>support@ibexperformance.com</strong> (replace with your real support email)</p>

  <div class="ibx-divider"></div>
  <div class="ibx-muted" style="font-size:12px;">Template for MVP use, not legal advice.</div>
</div>
""",
        unsafe_allow_html=True
    )

def render_faq():
    st.markdown(
        """
<div class="ibx-card">
  <div style="font-size:26px; font-weight:950; color:#0f172a;">FAQ</div>
  <div class="ibx-divider"></div>

  <h3>Is this medical advice?</h3>
  <p class="ibx-muted">No. IBEX is informational/organizational. Consult a qualified professional for medical concerns.</p>

  <h3>What’s the difference between Basic and Performance?</h3>
  <p class="ibx-muted"><strong>Basic</strong> is a conservative, essentials-only system. <strong>Performance</strong> unlocks a broader catalog and conditional optimization.</p>

  <h3>Do you sell my data?</h3>
  <p class="ibx-muted">No.</p>
</div>
""",
        unsafe_allow_html=True
    )

# =========================================================
# SUPABASE SAVE (AUDIT + ORDER)
# =========================================================
def supabase_save_audit(intake: dict, ai_out: dict):
    sb = supabase_client()
    if not sb:
        return

    row = {
        "rid": intake.get("rid"),
        "plan": intake.get("plan"),
        "name": intake.get("name"),
        "email": intake.get("email"),
        "school": intake.get("school"),
        "sport": intake.get("sport"),
        "position": intake.get("position"),
        "season_status": intake.get("season_status"),
        "training_days_per_week": intake.get("training_days_per_week"),
        "intensity_1_to_10": intake.get("intensity_1_to_10"),
        "travel_frequency": intake.get("travel_frequency"),
        "goals": intake.get("goals"),
        "sleep_hours": intake.get("sleep_hours"),
        "sleep_quality_1_to_10": intake.get("sleep_quality_1_to_10"),
        "stress_1_to_10": intake.get("stress_1_to_10"),
        "soreness_1_to_10": intake.get("soreness_1_to_10"),
        "gi_sensitive": intake.get("gi_sensitive"),
        "caffeine_sensitive": intake.get("caffeine_sensitive"),
        "current_supplements": intake.get("current_supplements"),
        "avoid_ingredients": intake.get("avoid_ingredients"),
        "open_notes": intake.get("open_notes"),
        "ai_out": safe_json(ai_out),
        "created_at": datetime.utcnow().isoformat()
    }

    # Upsert by rid (idempotent)
    sb.table("audits").upsert(row).execute()

def supabase_save_order(order_row: dict):
    sb = supabase_client()
    if not sb:
        return

    # Insert once by stripe_session_id (idempotent-ish)
    sb.table("orders").upsert(order_row, on_conflict="stripe_session_id").execute()


# =========================================================
# FILE CHECKS + LOAD DATA
# =========================================================
require_file(PRODUCTS_CSV, "products.csv (data/products.csv)")
require_file(EXCLUSIONS_CSV, "exclusions.csv (data/exclusions.csv)")
require_file(LOGO_PATH, "logo (assets/ibex_logo.png)")

products = load_products()
exclusions = load_exclusions()

STRIPE_BASIC_LINK = st.secrets.get("STRIPE_BASIC_LINK", "")
STRIPE_PERF_LINK = st.secrets.get("STRIPE_PERF_LINK", "")

# =========================================================
# SESSION STATE
# =========================================================
if "ai_out" not in st.session_state:
    st.session_state.ai_out = None
if "last_plan" not in st.session_state:
    st.session_state.last_plan = "Basic"
if "last_rid" not in st.session_state:
    st.session_state.last_rid = None

# Draft RID: always exists, used for the *next* audit submission
if "draft_rid" not in st.session_state:
    st.session_state.draft_rid = new_rid()

# Store last intake for optional support / linking
if "last_intake" not in st.session_state:
    st.session_state.last_intake = None

# =========================================================
# TOP HEADER
# =========================================================
render_header()

# =========================================================
# ROUTING: CLEAN THANK YOU PAGE
# =========================================================
qp = get_query_params()
success = str(qp.get("success", ["false"])[0] if isinstance(qp.get("success"), list) else qp.get("success", "false")).lower() == "true"
session_id = (qp.get("session_id", [""])[0] if isinstance(qp.get("session_id"), list) else qp.get("session_id", "")) or ""

def render_thank_you_page():
    # Premium confirmation page. No sloppy Stripe debug.
    st.markdown(
        """
        <div class="ibx-card" style="padding:32px; border-radius:26px;">
          <div style="display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;">
            <div>
              <div style="font-size:30px; font-weight:950; color:#0f172a;">✅ Payment confirmed</div>
              <div class="ibx-muted" style="margin-top:6px; font-size:15px;">
                Thanks — you’re all set. We’ve emailed your receipt.
              </div>
            </div>
            <div>
              <span class="ibx-badge">Order received</span>
              <span class="ibx-badge">Secure checkout</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Try to save order to Supabase (silent)
    # If Stripe isn't configured, we still show clean page.
    detected_rid = None
    plan = None

    if session_id and stripe_enabled():
        try:
            sess, line_items = stripe_retrieve_checkout_session(session_id)

            # Try to pull audit id from custom fields (Payment Link)
            detected_rid = extract_audit_id_from_session(sess)

            # Pull basic order data (no UI display)
            customer_email = sess.get("customer_details", {}).get("email") or sess.get("customer_email")
            amount_total = sess.get("amount_total")
            currency = sess.get("currency")
            status = sess.get("payment_status") or sess.get("status")
            shipping_details = sess.get("shipping_details")  # inline object if collected
            payment_intent = sess.get("payment_intent")
            customer_id = sess.get("customer")
            subscription_id = sess.get("subscription")

            # Determine plan from line items / price description if possible
            # (Optional – you can also store plan via custom field in Payment Link)
            try:
                li_data = (line_items.get("data") or [])
                names = []
                for li in li_data:
                    desc = li.get("description")
                    if desc:
                        names.append(desc.lower())
                if any("performance" in n for n in names):
                    plan = "Performance"
                elif any("basic" in n for n in names):
                    plan = "Basic"
            except Exception:
                pass

            # Save to Supabase if configured
            sb = supabase_client()
            if sb:
                order_row = {
                    "stripe_session_id": session_id,
                    "stripe_payment_intent": payment_intent,
                    "stripe_customer_id": customer_id,
                    "stripe_subscription_id": subscription_id,
                    "email": customer_email,
                    "amount_total": amount_total,
                    "currency": currency,
                    "status": status,
                    "rid": detected_rid,     # may be None if they didn’t enter it
                    "plan": plan,
                    "shipping": safe_json(shipping_details),
                    "line_items": safe_json(line_items),
                    "created_at": datetime.utcnow().isoformat()
                }
                supabase_save_order(order_row)

        except Exception:
            # We intentionally DO NOT show technical errors on the customer page.
            pass

    # Show RID if we can (nice + clear)
    # Priority: detected from Stripe custom field -> last_rid in session state
    show_rid = detected_rid or st.session_state.get("last_rid")
    if show_rid:
        st.markdown(
            """
            <div class="ibx-card" style="border-radius:26px;">
              <div style="font-size:18px; font-weight:950; color:#0f172a;">IBEX Audit ID</div>
              <div class="ibx-muted" style="margin-top:4px;">Keep this for support and order matching.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.code(show_rid, language=None)

    st.markdown(
        """
        <div class="ibx-card" style="border-radius:26px;">
          <div style="font-size:20px; font-weight:950; color:#0f172a;">What happens next</div>
          <div class="ibx-divider"></div>
          <ul>
            <li class="ibx-muted">You’ll receive a Stripe confirmation email.</li>
            <li class="ibx-muted">If you provided your IBEX Audit ID at checkout, your order is automatically matched to your audit.</li>
            <li class="ibx-muted">We’ll email tracking when it ships.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        if st.button("Return to IBEX (start / view audit)"):
            # Clear success params so it doesn't loop
            set_query_params()
            st.rerun()
    with c2:
        st.markdown("<div class='ibx-muted' style='margin-top:10px;'>Need help? Email support@ibexperformance.com</div>", unsafe_allow_html=True)


# If redirected from Stripe, show Thank You page and stop
if success:
    render_thank_you_page()
    st.stop()

# =========================================================
# MAIN APP TABS
# =========================================================
tabs = st.tabs(["Audit", "Privacy", "FAQ"])


# =========================================================
# TAB: AUDIT
# =========================================================
with tabs[0]:
    # RESULTS TOP (no scrolling)
    if st.session_state.ai_out:
        ai_out = st.session_state.ai_out
        plan = st.session_state.last_plan
        rid = st.session_state.last_rid

        st.markdown(
            f"""
            <div class="ibx-card">
              <div style="display:flex; justify-content:space-between; align-items:flex-end; gap:12px; flex-wrap:wrap;">
                <div>
                  <div style="font-size:28px; font-weight:950; color:#0f172a;">Your {plan} System</div>
                  <div class="ibx-muted">Your IBEX Audit ID (copy + paste during checkout)</div>
                </div>
                <div>
                  <span class="ibx-badge">Instant audit</span>
                  <span class="ibx-badge">Plan-aware</span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Big obvious RID block
        st.code(rid, language=None)
        st.caption("Copy this ID and paste it into Stripe during checkout so we can match your order to your recommendations.")

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
        notes = ai_out.get("notes_for_athlete", [])
        if notes:
            st.markdown("<div class='ibx-card'>", unsafe_allow_html=True)
            for n in notes:
                st.write(f"• {n}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Checkout")
        st.caption("Choose the plan you selected. Paste your IBEX Audit ID into Stripe during checkout.")

        if plan == "Basic":
            if STRIPE_BASIC_LINK:
                st.link_button("Subscribe — IBEX Basic", STRIPE_BASIC_LINK)
            else:
                st.info("Set STRIPE_BASIC_LINK in Streamlit Secrets.")
        else:
            if STRIPE_PERF_LINK:
                st.link_button("Subscribe — IBEX Performance", STRIPE_PERF_LINK)
            else:
                st.info("Set STRIPE_PERF_LINK in Streamlit Secrets.")

        if st.button("Start a new audit"):
            st.session_state.ai_out = None
            st.session_state.last_rid = None
            st.session_state.last_intake = None
            st.session_state.draft_rid = new_rid()  # always new
            st.rerun()

    else:
        st.markdown(
            """
            <div class="ibx-card">
              <div style="font-size:28px; font-weight:950; color:#0f172a;">Performance Audit</div>
              <div class="ibx-muted" style="margin-top:6px;">
                Fill out the audit in the sidebar. Your results appear here instantly — no scrolling.
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # SIDEBAR FORM (FULL QUESTIONS)
    with st.sidebar:
        st.markdown("## IBEX Audit")
        st.caption("Plan → Audit → Instant system.")

        plan = st.radio(
            "Choose your plan",
            ["Basic", "Performance"],
            index=0 if st.session_state.last_plan == "Basic" else 1,
            horizontal=True
        )

        pc = PLAN_COPY[plan]
        st.markdown(f"### {pc['headline']}")
        st.write(pc["sub"])
        for b in pc["bullets"]:
            st.write(f"• {b}")
        st.caption(pc["note"])

        st.markdown("---")

        with st.form("audit_form"):
            st.markdown("### About you")
            name = st.text_input("Full name")
            email = st.text_input("Email")
            school = st.text_input("School")

            st.markdown("### Sport & training")
            sport = st.text_input("Sport")
            position = st.text_input("Position / Event")
            season_status = st.selectbox("Season status", ["In-season", "Pre-season", "Off-season"])
            training_days = st.slider("Training days/week", 0, 7, 5)
            intensity = st.slider("Training intensity (1–10)", 1, 10, 7)
            travel = st.selectbox("Travel frequency", ["Never", "Sometimes", "Often"])

            st.markdown("### Goals")
            goals = st.multiselect(
                "Select all that apply",
                ["strength","endurance","recovery","sleep","gut","joints","focus","general health"]
            )

            st.markdown("### Recovery & lifestyle")
            sleep_hours = st.number_input("Sleep hours/night", min_value=0.0, max_value=12.0, value=7.0, step=0.5)
            sleep_quality = st.slider("Sleep quality (1–10)", 1, 10, 6)
            stress = st.slider("Stress (1–10)", 1, 10, 6)
            soreness = st.slider("Soreness/Fatigue (1–10)", 1, 10, 6)
            gi_sensitive = st.checkbox("GI sensitive / stomach issues", value=False)
            caffeine_sensitive = st.checkbox("Caffeine sensitive", value=False)

            st.markdown("### Current stack / notes")
            current_supps = st.text_area("Supplements you already take (optional)", placeholder="Creatine, fish oil, whey…")
            avoid_ingredients = st.text_input("Ingredients to avoid (optional)", placeholder="e.g., caffeine")
            open_notes = st.text_area("Other context or concerns (optional)", placeholder="Anything that would help tailor the plan…")

            st.markdown("---")
            st.caption("By continuing, you agree to the Privacy Policy (see Privacy tab).")
            consent = st.checkbox("I understand this is not medical advice.", value=False)

            submitted = st.form_submit_button("Build my system")

        if submitted:
            if not consent:
                st.error("Please check the consent box to proceed.")
                st.stop()

            # Use the draft RID so it’s stable for checkout
            rid = st.session_state.draft_rid

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
                ai_out = run_ai(intake, shortlist, exclusions, plan)

            # Save audit to Supabase (optional, silent)
            try:
                supabase_save_audit(intake, ai_out)
            except Exception:
                pass

            st.session_state.ai_out = ai_out
            st.session_state.last_plan = plan
            st.session_state.last_rid = rid
            st.session_state.last_intake = intake

            # IMPORTANT: rotate to a fresh draft RID for next audit
            st.session_state.draft_rid = new_rid()

            st.rerun()


# =========================================================
# TAB: PRIVACY
# =========================================================
with tabs[1]:
    render_privacy_policy()

# =========================================================
# TAB: FAQ
# =========================================================
with tabs[2]:
    render_faq()









