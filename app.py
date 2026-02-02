import os
import json
import uuid
import base64
from datetime import date, datetime, timezone

import pandas as pd
import streamlit as st

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# =========================================================
# CONFIG
# =========================================================
APP_TITLE = "IBEX"
APP_TAGLINE = "Personalized performance systems for athletes"

PRODUCTS_CSV = "data/products.csv"
EXCLUSIONS_CSV = "data/exclusions.csv"

# Put your HIGH-RES logo here (512x512 or 1024x1024 png recommended)
LOGO_PATH = "assets/ibex_logo.png"


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title=f"{APP_TITLE} • Performance Audit",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PREMIUM STYLING (fix contrast + inputs/dropdowns/buttons)
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
  border-radius: 20px;
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
  font-weight: 900 !important;
  color:#ffffff !important;               /* force readable text */
  text-decoration: none !important;
}
.stButton button{
  background: var(--accent) !important;
  border: none !important;
}
.stButton button:hover{ opacity: 0.92; }

/* Link buttons (Stripe) */
.stLinkButton a{
  background: var(--accent2) !important;
  border: 1px solid rgba(17,24,39,0.15) !important;
  color: #fff !important;
}
.stLinkButton a:hover{ opacity:0.92; }

/* Reduce extra whitespace above */
.block-container{ padding-top: 1.0rem; }

/* ---------------------------------------------------------
   SIDEBAR INPUT “BLOCKS” + DROPDOWN TEXT COLOR (BaseWeb)
--------------------------------------------------------- */
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stTextArea textarea,
section[data-testid="stSidebar"] .stNumberInput input{
  background:#ffffff !important;
  color: var(--text) !important;
  border:1px solid rgba(229,231,235,0.35) !important;
  border-radius: 14px !important;
}

/* Placeholders (sidebar) */
section[data-testid="stSidebar"] .stTextInput input::placeholder,
section[data-testid="stSidebar"] .stTextArea textarea::placeholder,
section[data-testid="stSidebar"] .stNumberInput input::placeholder{
  color: var(--muted) !important;
  opacity: 1 !important;
}

/* Selectbox / Multiselect (BaseWeb Select) */
section[data-testid="stSidebar"] [data-baseweb="select"] > div{
  background:#ffffff !important;
  border:1px solid rgba(229,231,235,0.35) !important;
  border-radius: 14px !important;
}

/* Selected value + input text inside select */
section[data-testid="stSidebar"] [data-baseweb="select"] *{
  color: var(--text) !important;
}

/* Dropdown menu panel + options (portal) */
div[data-baseweb="popover"] *{
  color: var(--text) !important;
}
div[data-baseweb="menu"]{
  background:#ffffff !important;
  border:1px solid rgba(15,23,42,0.10) !important;
  border-radius: 14px !important;
  overflow:hidden !important;
}
div[data-baseweb="menu"] [role="option"]{
  background:#ffffff !important;
}
div[data-baseweb="menu"] [role="option"]:hover{
  background: rgba(15,23,42,0.06) !important;
}

/* Multiselect pills readable */
section[data-testid="stSidebar"] [data-baseweb="tag"]{
  background: rgba(239,68,68,0.10) !important;
  color: var(--text) !important;
  border-radius: 999px !important;
}

/* Sidebar slider labels */
section[data-testid="stSidebar"] .stSlider *{
  color: var(--sideText) !important;
}

/* Radio + checkbox text stays light in sidebar */
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stCheckbox label{
  color: var(--sideText) !important;
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
    if not os.path.exists(LOGO_PATH):
        return None
    if Image is None:
        return None
    try:
        return Image.open(LOGO_PATH)
    except Exception:
        return None


def render_header():
    logo = load_logo()
    c1, c2 = st.columns([1, 7], gap="large")
    with c1:
        if logo is not None:
            st.image(logo, width=140)
        else:
            st.markdown("<div class='ibx-card' style='text-align:center;'>IBEX</div>", unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"<div style='font-size:46px; font-weight:950; color:#0f172a; margin-top:2px;'>{APP_TITLE}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='ibx-muted' style='font-size:16px; margin-top:-8px;'>{APP_TAGLINE}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div style="margin-top:12px;">
              <span class="ibx-badge">Plan-aware AI</span>
              <span class="ibx-badge">Privacy-first</span>
              <span class="ibx-badge">Athlete-safe guardrails</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


@st.cache_data(show_spinner=False)
def load_products():
    df = pd.read_csv(PRODUCTS_CSV)
    df.columns = [c.strip() for c in df.columns]
    required = [
        "Product_ID", "Category", "Ingredient", "Brand", "Store", "Link",
        "Serving_Form", "Typical_Use", "Timing", "Avoid_If",
        "Third_Party_Tested", "NSF_Certified", "Price", "Est_Monthly_Cost", "Notes"
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


def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY in Streamlit Secrets.")
        st.stop()
    if OpenAI is None:
        st.error("openai package not installed. Ensure requirements.txt includes `openai`.")
        st.stop()
    return OpenAI(api_key=api_key)


def is_yes(val) -> bool:
    return str(val).strip().lower() in {"y", "yes", "true", "1"}


def qp_get(name: str, default: str = "") -> str:
    """
    Works with Streamlit's st.query_params which may return a list-like or string.
    """
    try:
        v = st.query_params.get(name)
        if v is None:
            return default
        if isinstance(v, (list, tuple)):
            return v[0] if v else default
        return str(v)
    except Exception:
        return default


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# =========================================================
# STRIPE (debug-safe + real error)
# =========================================================
def stripe_retrieve_session(session_id: str):
    try:
        import stripe
    except Exception as e:
        st.error("Stripe package not installed (missing `stripe` in requirements.txt).")
        st.exception(e)
        return None

    key = st.secrets.get("STRIPE_SECRET_KEY", "")
    if not key:
        st.error("Missing STRIPE_SECRET_KEY in Streamlit Secrets.")
        return None

    # Safe fingerprint so you can confirm Streamlit is using the key you set
    st.caption(f"Stripe key loaded: {key[:7]}…{key[-4:]} (len={len(key)})")
    st.caption(f"Session id: {session_id[:10]}…")

    stripe.api_key = key

    try:
        # NOTE: shipping_details is NOT expandable; it's already included on the session
        sess = stripe.checkout.Session.retrieve(
            session_id,
            expand=["customer", "subscription", "line_items"],
        )
        return sess
    except Exception as e:
        st.error("Stripe API call failed. Here is the real error:")
        st.exception(e)
        return None



# =========================================================
# SUPABASE (optional)
# =========================================================
def sb_client():
    """
    Secrets you can set:
      SUPABASE_URL
      SUPABASE_SERVICE_ROLE_KEY   (recommended for server-side inserts)
    """
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        return None

    try:
        from supabase import create_client
    except Exception:
        return None

    try:
        return create_client(url, key)
    except Exception:
        return None


def sb_upsert_audit(audit_row: dict):
    sb = sb_client()
    if sb is None:
        return False, "Supabase not configured (missing package or secrets)."
    try:
        # expects table: audits with primary key rid
        sb.table("audits").upsert(audit_row).execute()
        return True, ""
    except Exception as e:
        return False, str(e)


def sb_insert_order(order_row: dict):
    sb = sb_client()
    if sb is None:
        return False, "Supabase not configured (missing package or secrets)."
    try:
        # expects table: orders (id optional, created_at optional)
        sb.table("orders").insert(order_row).execute()
        return True, ""
    except Exception as e:
        return False, str(e)


# =========================================================
# PLAN DEFINITIONS (copy)
# =========================================================
PLAN_COPY = {
    "Basic": {
        "headline": "Foundations, done right.",
        "sub": "A clean, conservative system built from the essentials. Minimal complexity. Maximum consistency.",
        "bullets": [
            "Core performance stack only (the “boring” stuff that actually works)",
            "Prefers NSF Certified for Sport / third-party tested when available",
            "Designed for consistency, budgeting, and simplicity",
        ],
        "note": "Best for: most college athletes who want a safe, no-BS baseline.",
    },
    "Performance": {
        "headline": "Optimization mode.",
        "sub": "A deeper system with expanded options and conditional additions based on your audit.",
        "bullets": [
            "Expanded catalog (advanced recovery, sleep, gut, joint support as needed)",
            "More conditional logic: your schedule + training load + sensitivities",
            "Built for athletes chasing marginal gains (without sketchy stuff)",
        ],
        "note": "Best for: high volume training, in-season stress, or athletes who want every edge.",
    },
}

BASIC_CORE_CATEGORIES = {
    "Creatine", "Omega-3", "Magnesium", "Vitamin D", "Electrolytes", "Protein",
    "Multivitamin", "Zinc", "Vitamin C", "Probiotic", "Fiber", "Collagen", "Tart Cherry"
}


def filter_products_by_plan(products: pd.DataFrame, plan: str) -> pd.DataFrame:
    p = products.copy()
    p["Category_norm"] = p["Category"].astype(str).str.strip()
    if plan == "Basic":
        return p[p["Category_norm"].isin(BASIC_CORE_CATEGORIES)]
    return p


def shortlist_products(products: pd.DataFrame, goals: list[str], gi_sensitive: bool, caffeine_sensitive: bool, plan: str) -> pd.DataFrame:
    p = filter_products_by_plan(products, plan)

    # soft goal matching
    if goals:
        mask = False
        for g in goals:
            mask = mask | p["Typical_Use"].astype(str).str.contains(g, case=False, na=False)
        if mask is not False:
            p = p[mask]

    # sensitivities
    if gi_sensitive:
        p = p[~p["Avoid_If"].astype(str).str.contains("GI", case=False, na=False)]
    if caffeine_sensitive:
        p = p[~p["Avoid_If"].astype(str).str.contains("caffeine", case=False, na=False)]

    # sort for Basic: certified/tested first
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
        "Product_ID", "Category", "Ingredient", "Brand", "Store", "Serving_Form",
        "Typical_Use", "Timing", "Avoid_If", "Third_Party_Tested", "NSF_Certified", "Notes"
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

    content = resp.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except Exception:
        # best-effort JSON extraction
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start:end + 1])
        raise


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
    blocks = [("AM", "Morning"), ("PM", "Evening"), ("Training", "Training")]
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
                    st.markdown(f"- **{p.get('Ingredient', pid)}** — {p.get('Brand', '')}")
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
    <li><strong>Contact info</strong> (name/email if you provide it).</li>
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


def render_audit_id_card(rid: str):
    st.markdown(
        f"""
<div class="ibx-card" style="border:1px solid rgba(239,68,68,0.30); background:linear-gradient(180deg, rgba(239,68,68,0.06), rgba(255,255,255,1));">
  <div style="font-size:18px; font-weight:950; color:#0f172a;">IBEX Audit ID (copy + paste during checkout)</div>
  <div class="ibx-muted" style="margin-top:6px;">This links your payment to your personalized recommendations.</div>
  <div style="margin-top:14px; padding:16px; border-radius:16px; background:#0b1220; color:#ffffff; font-weight:900; font-size:18px; letter-spacing:0.4px; text-align:center;">
    {rid}
  </div>
</div>
""",
        unsafe_allow_html=True
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("Copy Audit ID"):
            st.toast("Audit ID copied? (If not, select and copy it manually.)", icon="✅")
    with c2:
        st.caption("Tip: Click the ID above → Cmd/Ctrl+C")


# =========================================================
# FILE REQUIREMENTS
# =========================================================
require_file(PRODUCTS_CSV, "products.csv (data/products.csv)")
require_file(EXCLUSIONS_CSV, "exclusions.csv (data/exclusions.csv)")
if os.path.exists(LOGO_PATH) is False:
    # Don't hard-stop if logo missing; app can still run.
    pass

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
if "last_intake" not in st.session_state:
    st.session_state.last_intake = None


# =========================================================
# TOP: HEADER
# =========================================================
render_header()

# =========================================================
# CONFIRMATION / THANK YOU PAGE (Stripe redirect target)
# If URL has ?success=true&session_id=...
# =========================================================
success = qp_get("success", "false").lower() == "true"
session_id = qp_get("session_id", "")

if success:
    st.markdown(
        """
<div class="ibx-card" style="padding:30px;">
  <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap;">
    <div style="font-size:22px; font-weight:950; color:#0f172a;">✅ Payment confirmed</div>
    <span class="ibx-badge">You’re all set</span>
  </div>
  <div class="ibx-muted" style="margin-top:6px;">
    Thanks — we received your payment. Below is your order confirmation.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='ibx-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:18px; font-weight:950; color:#0f172a;'>Order details</div>", unsafe_allow_html=True)
    st.markdown("<div class='ibx-divider'></div>", unsafe_allow_html=True)

    if session_id:
        sess = stripe_retrieve_session(session_id)
        if sess:
            customer_email = (sess.get("customer_details") or {}).get("email")
            shipping = sess.get("shipping_details")
            mode = sess.get("mode")
            amount_total = sess.get("amount_total")
            currency = (sess.get("currency") or "").upper()

            # Pull custom field / metadata if present (depends on how you configured Payment Link)
            # If you add a custom field in Stripe called "IBEX Audit ID", it'll show in custom_fields
            custom_fields = sess.get("custom_fields") or []
            audit_id_from_checkout = ""
            for f in custom_fields:
                try:
                    if (f.get("key") or "").lower() in {"ibex_audit_id", "audit_id", "ibex_auditid"}:
                        audit_id_from_checkout = (f.get("text") or {}).get("value", "")
                except Exception:
                    pass

            # Render summary
            st.markdown(
                f"""
<div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px;">
  <div>
    <div class="ibx-muted">Email</div>
    <div style="font-weight:900; color:#0f172a;">{customer_email or "—"}</div>
  </div>
  <div>
    <div class="ibx-muted">Amount</div>
    <div style="font-weight:900; color:#0f172a;">{(amount_total/100):.2f} {currency if currency else ""}</div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

            st.markdown("<div class='ibx-divider'></div>", unsafe_allow_html=True)

            st.markdown("<div class='ibx-muted'>Shipping</div>", unsafe_allow_html=True)
            if shipping:
                st.json(shipping)
            else:
                st.write("—")

            if audit_id_from_checkout:
                st.markdown("<div class='ibx-divider'></div>", unsafe_allow_html=True)
                st.markdown("<div class='ibx-muted'>IBEX Audit ID (from checkout)</div>", unsafe_allow_html=True)
                st.code(audit_id_from_checkout, language=None)

            # Save to Supabase (optional)
            order_row = {
                "created_at": now_iso(),
                "stripe_session_id": session_id,
                "stripe_mode": mode,
                "customer_email": customer_email,
                "shipping_details": shipping,
                "audit_id": audit_id_from_checkout or None,
                "amount_total": amount_total,
                "currency": currency or None,
            }
            ok, err = sb_insert_order(order_row)
            if ok:
                st.success("Saved order to Supabase.")
            else:
                st.caption("Supabase save skipped / failed (optional).")
                st.caption(err[:220] if err else "")

        else:
            st.warning("Could not load Stripe session details.")
            st.caption("If the error above says permission/No such session, your key permissions/account/mode don’t match the session.")
    else:
        st.warning("Missing session_id in URL. Your Stripe redirect should include session_id={CHECKOUT_SESSION_ID}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
<div class="ibx-card">
  <div style="font-size:18px; font-weight:950; color:#0f172a;">Next steps</div>
  <div class="ibx-divider"></div>
  <ul>
    <li>You’ll receive a Stripe confirmation email.</li>
    <li>If you entered your IBEX Audit ID at checkout, we’ll match your order to your recommendations automatically.</li>
    <li>We’ll email tracking when it ships.</li>
  </ul>
</div>
""",
        unsafe_allow_html=True,
    )

    if st.button("Return to IBEX (start / view audit)"):
        # If your Streamlit version supports it, clear query params
        try:
            st.query_params.clear()
        except Exception:
            pass
        st.rerun()

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
                  <div class="ibx-muted">Reference: generated from your audit</div>
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

        if rid:
            render_audit_id_card(rid)

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
        else:
            st.info("No additional notes.")

        st.subheader("Checkout")

        st.markdown(
            "<div class='ibx-card'>"
            "<div style='font-size:16px; font-weight:950; color:#0f172a;'>Before you pay</div>"
            "<div class='ibx-muted' style='margin-top:6px;'>Copy your <strong>IBEX Audit ID</strong> above and paste it into Stripe when asked. That’s how we match your order to your plan.</div>"
            "</div>",
            unsafe_allow_html=True
        )

        # If Basic, only show Basic link; else Performance link
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

        c1, c2 = st.columns([1, 2])
        with c1:
            if st.button("Start a new audit"):
                st.session_state.ai_out = None
                st.session_state.last_rid = None
                st.session_state.last_intake = None
                st.rerun()
        with c2:
            st.caption("You can run multiple audits. Each one generates a new Audit ID.")

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

    # SIDEBAR FORM (FULL QUESTION SET)
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
                ["strength", "endurance", "recovery", "sleep", "gut", "joints", "focus", "general health"]
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
            open_notes = st.text_area(
                "Any other concerns or context you want IBEX to know (optional)",
                placeholder="Injuries, travel schedule, diet restrictions, sleep issues, anything relevant…"
            )

            st.markdown("---")
            st.caption("By continuing, you agree to the Privacy Policy (see Privacy tab).")
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
                "open_notes": open_notes,
                "created_at": now_iso(),
            }

            shortlist = shortlist_products(products, goals, gi_sensitive, caffeine_sensitive, plan)

            with st.spinner("Generating your system…"):
                ai_out = run_ai(intake, shortlist, exclusions, plan)

            st.session_state.ai_out = ai_out
            st.session_state.last_plan = plan
            st.session_state.last_rid = rid
            st.session_state.last_intake = intake

            # Save audit + AI output to Supabase (optional)
            audit_row = {
                "rid": rid,
                "created_at": now_iso(),
                "plan": plan,
                "name": name,
                "email": email,
                "school": school,
                "sport": sport,
                "position": position,
                "season_status": season_status,
                "intake_json": intake,
                "ai_json": ai_out,
            }
            ok, err = sb_upsert_audit(audit_row)
            if not ok:
                # Don't block the user if Supabase isn't ready.
                st.toast("Audit saved locally for this session. Supabase save skipped/failed (optional).", icon="ℹ️")
                if err:
                    st.caption(err[:220])

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









