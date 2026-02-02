import os
import json
import uuid
import base64
from datetime import date

import pandas as pd
import streamlit as st

# Optional deps
try:
    from PIL import Image
except Exception:
    Image = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import stripe
except Exception:
    stripe = None

try:
    import requests
except Exception:
    requests = None


# =========================================================
# CONFIG / PATHS
# =========================================================
APP_TITLE = "IBEX"
APP_TAGLINE = "Personalized performance systems for athletes"

PRODUCTS_CSV = "data/products.csv"
EXCLUSIONS_CSV = "data/exclusions.csv"
LOGO_PATH = "assets/ibex_logo.png"   # 512x512 or 1024x1024 PNG recommended


# ---------------------------------------------------------
# PAGE CONFIG (favicon / engine tab icon)
# ---------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_TITLE} • Performance Audit",
    page_icon=LOGO_PATH,
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PREMIUM STYLING (Fix contrast + dropdowns + buttons)
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
.stTextInput input,
.stTextArea textarea,
.stNumberInput input{
  background:#fff !important;
  color:var(--text) !important;
  border:1px solid var(--border) !important;
  border-radius:14px !important;
}
.stTextInput input::placeholder,
.stTextArea textarea::placeholder,
.stNumberInput input::placeholder{
  color: var(--muted) !important;
  opacity: 1 !important;
}

/* Tabs */
button[data-baseweb="tab"]{
  color: var(--sub) !important;
  font-weight: 700 !important;
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

/* Buttons (regular buttons) */
div[data-testid="stButton"] button{
  border-radius: 14px !important;
  padding: 0.85rem 1.10rem !important;
  font-weight: 900 !important;
  border: none !important;
  background: var(--accent) !important;
  color: #ffffff !important;
}
div[data-testid="stButton"] button:hover{ opacity: 0.92; }

/* Link buttons (st.link_button) — Streamlit renders this differently */
div[data-testid="stLinkButton"] a{
  display:inline-flex !important;
  align-items:center !important;
  justify-content:center !important;
  border-radius: 14px !important;
  padding: 0.85rem 1.10rem !important;
  font-weight: 900 !important;
  text-decoration:none !important;
  background: var(--accent2) !important;
  border: 1px solid rgba(17,24,39,0.15) !important;
  color:#ffffff !important;
}
div[data-testid="stLinkButton"] a:hover{ opacity:0.92; }
div[data-testid="stLinkButton"] a *{ color:#ffffff !important; }

/* Reduce extra whitespace above */
.block-container{ padding-top: 1.0rem; }

/* ---------------------------------------------------------
   SIDEBAR INPUTS — keep text dark inside bubbles
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

/* BaseWeb Select (selectbox + multiselect) — selected value */
section[data-testid="stSidebar"] [data-baseweb="select"] > div{
  background:#ffffff !important;
  border:1px solid rgba(229,231,235,0.35) !important;
  border-radius: 14px !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] span,
section[data-testid="stSidebar"] [data-baseweb="select"] input{
  color: var(--text) !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] svg{
  color: var(--text) !important;
}

/* Dropdown menu (portal) */
div[data-baseweb="popover"]{
  z-index: 9999 !important;
}
div[data-baseweb="menu"]{
  background:#ffffff !important;
  border:1px solid rgba(15,23,42,0.10) !important;
  border-radius: 14px !important;
  overflow:hidden !important;
}
div[data-baseweb="menu"] *{
  color: var(--text) !important;
}
div[data-baseweb="menu"] [role="option"]{
  background:#ffffff !important;
}
div[data-baseweb="menu"] [role="option"]:hover{
  background: rgba(15,23,42,0.06) !important;
}

/* Radio + checkbox labels remain light in sidebar */
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
    if logo is not None:
        c1, c2 = st.columns([1, 7], gap="large")
        with c1:
            st.image(logo, width=140)
        with c2:
            st.markdown(
                f"<div style='font-size:44px; font-weight:900; color:#0f172a; margin-top:2px;'>{APP_TITLE}</div>",
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
              <div style="font-size:44px; font-weight:900; color:#0f172a;">{APP_TITLE}</div>
              <div class="ibx-muted" style="font-size:16px;">{APP_TAGLINE}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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

def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY in Streamlit Secrets.")
        st.stop()
    if OpenAI is None:
        st.error("openai package not installed. Ensure requirements.txt includes `openai`.")
        st.stop()
    return OpenAI(api_key=api_key)

def init_stripe():
    sk = st.secrets.get("STRIPE_SECRET_KEY")
    if not sk:
        return False
    if stripe is None:
        st.error("stripe package not installed. Add `stripe` to requirements.txt.")
        st.stop()
    stripe.api_key = sk
    return True

def is_yes(val) -> bool:
    return str(val).strip().lower() in {"y","yes","true","1"}

# -------- Supabase REST helpers (no extra libs needed) ----
def supabase_enabled() -> bool:
    return bool(st.secrets.get("SUPABASE_URL") and st.secrets.get("SUPABASE_SERVICE_ROLE_KEY") and requests is not None)

def sb_headers():
    key = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

def sb_url(table: str) -> str:
    base = st.secrets["SUPABASE_URL"].rstrip("/")
    return f"{base}/rest/v1/{table}"

def sb_insert(table: str, row: dict):
    if not supabase_enabled():
        return None
    r = requests.post(sb_url(table), headers=sb_headers(), data=json.dumps(row))
    if r.status_code >= 300:
        st.error(f"Supabase insert failed ({table}): {r.status_code}")
        st.code(r.text)
        st.stop()
    out = r.json()
    return out[0] if isinstance(out, list) and out else out

def sb_update(table: str, match: dict, updates: dict):
    if not supabase_enabled():
        return None
    # build ?col=eq.value&...
    params = "&".join([f"{k}=eq.{str(v)}" for k, v in match.items()])
    url = sb_url(table) + (("?" + params) if params else "")
    r = requests.patch(url, headers=sb_headers(), data=json.dumps(updates))
    if r.status_code >= 300:
        st.error(f"Supabase update failed ({table}): {r.status_code}")
        st.code(r.text)
        st.stop()
    out = r.json()
    return out[0] if isinstance(out, list) and out else out


# =========================================================
# PLAN DEFINITIONS (cleaner copy)
# =========================================================
PLAN_COPY = {
    "Basic": {
        "headline": "Foundations, done right.",
        "sub": "A conservative essentials-only system. Simple, consistent, athlete-safe.",
        "bullets": [
            "Core stack only (the boring stuff that works)",
            "Prefers NSF / third-party tested where available",
            "Built for consistency + budget discipline",
        ],
        "note": "Best for: most college athletes who want a safe baseline.",
    },
    "Performance": {
        "headline": "Optimization mode.",
        "sub": "Expanded catalog + conditional additions based on your audit signals.",
        "bullets": [
            "Deeper recovery/sleep/gut/joint options when needed",
            "More conditional logic based on workload + season phase",
            "Still avoids sketchy categories and stimulant blends",
        ],
        "note": "Best for: high volume training, in-season stress, or marginal gains.",
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

    # tested/certified first for Basic
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
        "Plan: PERFORMANCE. Expanded optimization. Add conditional items only if clearly supported by intake. Still conservative on risk."
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
    blocks = [("AM","Morning"), ("PM","Evening"), ("Training","Training")]
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

  <p><strong>IBEX</strong> (“we”) provides an athlete-focused performance audit and personalized supplement organization experience.</p>

  <h3>What we collect</h3>
  <ul>
    <li><strong>Audit inputs</strong> you choose to provide (training, sleep, stress, preferences).</li>
    <li><strong>Contact info</strong> (email, name if provided).</li>
    <li><strong>Checkout info</strong> (shipping + payment) processed by <strong>Stripe</strong>. We do not store your full card number.</li>
  </ul>

  <h3>How we use it</h3>
  <ul>
    <li>Generate your personalized system and schedule.</li>
    <li>Operate subscriptions and fulfill shipments (if purchased).</li>
    <li>Provide support and improve the product (prefer aggregated insights).</li>
  </ul>

  <h3>What we do NOT do</h3>
  <ul>
    <li>We do <strong>not</strong> sell your personal data.</li>
    <li>We do <strong>not</strong> share your data with third parties for their marketing.</li>
  </ul>

  <h3>AI processing</h3>
  <p>Your audit inputs are sent to an AI model to generate recommendations. IBEX is not a medical service and does not provide medical advice.</p>

  <h3>Retention & deletion</h3>
  <p>We retain data only as long as needed to provide the service and meet legal obligations. You can request deletion.</p>

  <div class="ibx-divider"></div>
  <div class="ibx-muted" style="font-size:12px;">MVP template. Not legal advice.</div>
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

  <h3>Basic vs Performance?</h3>
  <p class="ibx-muted"><strong>Basic</strong> = essentials-only baseline. <strong>Performance</strong> = broader catalog + conditional optimization.</p>

  <h3>Do you sell my data?</h3>
  <p class="ibx-muted">No.</p>
</div>
""",
        unsafe_allow_html=True
    )


# =========================================================
# REQUIRED FILES
# =========================================================
require_file(PRODUCTS_CSV, "products.csv (data/products.csv)")
require_file(EXCLUSIONS_CSV, "exclusions.csv (data/exclusions.csv)")
require_file(LOGO_PATH, "logo (assets/ibex_logo.png)")

products = load_products()
exclusions = load_exclusions()

STRIPE_BASIC_LINK = st.secrets.get("STRIPE_BASIC_LINK", "")
STRIPE_PERF_LINK = st.secrets.get("STRIPE_PERF_LINK", "")

# Session state
if "ai_out" not in st.session_state:
    st.session_state.ai_out = None
if "last_plan" not in st.session_state:
    st.session_state.last_plan = "Basic"
if "last_rid" not in st.session_state:
    st.session_state.last_rid = None
if "paid_banner" not in st.session_state:
    st.session_state.paid_banner = None


# =========================================================
# STRIPE RETURN HANDLER (session_id -> pull shipping/email)
# =========================================================
stripe_ok = init_stripe()
qp = st.query_params
success = str(qp.get("success", "false")).lower()
session_id = qp.get("session_id", None)

if success == "true" and session_id and stripe_ok:
    try:
        sess = stripe.checkout.Session.retrieve(session_id)
        # get email + shipping
        cust_email = None
        shipping = None

        if getattr(sess, "customer_details", None):
            cust_email = getattr(sess.customer_details, "email", None)

        if getattr(sess, "shipping_details", None):
            sd = sess.shipping_details
            shipping = {
                "name": getattr(sd, "name", None),
                "address": sd.address.to_dict() if getattr(sd, "address", None) else None
            }

        # pull custom field "IBEX Audit ID" from Payment Link
        rid = None
        custom_fields = getattr(sess, "custom_fields", None)
        if custom_fields:
            for f in custom_fields:
                label = (f.get("label") or "").strip().lower()
                if label == "ibex audit id":
                    # text field
                    text = f.get("text", {})
                    rid = (text.get("value") or "").strip()

        # Update Supabase order + audit
        if supabase_enabled():
            # Update orders by stripe_session_id if exists, otherwise by rid
            updated = None
            if rid:
                sb_update("orders", {"rid": rid}, {
                    "paid": True,
                    "stripe_session_id": session_id,
                    "stripe_customer_email": cust_email,
                    "shipping": shipping,
                })
                sb_update("audits", {"rid": rid}, {"status": "paid"})
                updated = True
            else:
                # if no RID provided, at least store the session and email somewhere
                sb_insert("orders", {
                    "rid": None,
                    "plan": None,
                    "paid": True,
                    "stripe_session_id": session_id,
                    "stripe_customer_email": cust_email,
                    "shipping": shipping,
                })
                updated = True

            if updated:
                st.session_state.paid_banner = f"✅ Payment received. Email: {cust_email or '—'}"

    except Exception as e:
        st.session_state.paid_banner = f"Payment detected but could not verify details. ({type(e).__name__})"


# =========================================================
# UI
# =========================================================
render_header()

if st.session_state.paid_banner:
    st.success(st.session_state.paid_banner)

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
                  <div class="ibx-muted">Checkout Code (IBEX Audit ID): <b>{rid}</b></div>
                </div>
                <div>
                  <span class="ibx-badge">Instant audit</span>
                  <span class="ibx-badge">Plan-aware</span>
                </div>
              </div>
              <div class="ibx-divider"></div>
              <div class="ibx-muted">
                At checkout, paste <b>{rid}</b> into the field <b>IBEX Audit ID</b> so we can match your order to your audit.
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

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

        # Show ONLY the plan the user selected
        if plan == "Basic":
            st.markdown("<div class='ibx-card'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:18px; font-weight:950; color:#0f172a;'>IBEX Basic</div>", unsafe_allow_html=True)
            st.markdown("<div class='ibx-muted'>Foundations, done right.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            if STRIPE_BASIC_LINK:
                st.link_button("Subscribe — IBEX Basic", STRIPE_BASIC_LINK)
            else:
                st.info("Set STRIPE_BASIC_LINK in Streamlit Secrets.")
        else:
            st.markdown("<div class='ibx-card'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:18px; font-weight:950; color:#0f172a;'>IBEX Performance</div>", unsafe_allow_html=True)
            st.markdown("<div class='ibx-muted'>Expanded catalog + conditional optimization.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            if STRIPE_PERF_LINK:
                st.link_button("Subscribe — IBEX Performance", STRIPE_PERF_LINK)
            else:
                st.info("Set STRIPE_PERF_LINK in Streamlit Secrets.")

        if st.button("Start a new audit"):
            st.session_state.ai_out = None
            st.session_state.last_rid = None
            st.session_state.paid_banner = None
            st.query_params.clear()
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

    # SIDEBAR FORM
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
                ai_out = run_ai(intake, shortlist, exclusions, plan)

            # ---- Save to Supabase (audit + unpaid order stub) ----
            if supabase_enabled():
                sb_insert("audits", {
                    "rid": rid,
                    "plan": plan,
                    "name": name,
                    "email": email,
                    "school": school,
                    "intake": intake,
                    "ai_out": ai_out,
                    "status": "created"
                })
                sb_insert("orders", {
                    "rid": rid,
                    "plan": plan,
                    "paid": False
                })

            st.session_state.ai_out = ai_out
            st.session_state.last_plan = plan
            st.session_state.last_rid = rid
            st.session_state.paid_banner = None
            st.query_params.clear()
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





