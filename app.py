import os
import json
import uuid
import base64
from datetime import date

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# NEW: Supabase
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
# PREMIUM STYLING
# FIXES:
#  - Dropdown menu options unreadable (BaseWeb menu portal)
# CHANGES:
#  - No checkbox gating UX changes here; that's in form logic below
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

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.stApp{ background: var(--bg); }
html, body, [class*="css"]{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }

h1,h2,h3,h4,h5{ color:var(--text) !important; letter-spacing:-0.2px; }
p,li,span,div,label{ color:var(--sub); }

section[data-testid="stSidebar"]{
  background: var(--side);
  border-right:1px solid var(--sideBorder);
}
section[data-testid="stSidebar"] *{
  color: var(--sideText) !important;
}
section[data-testid="stSidebar"] a{ color:#93c5fd !important; }

input, textarea, select {
  background:#fff !important;
  color:var(--text) !important;
  border:1px solid var(--border) !important;
  border-radius:12px !important;
}

button[data-baseweb="tab"]{
  color: var(--sub) !important;
  font-weight: 600;
  border-radius: 12px 12px 0 0 !important;
}
button[data-baseweb="tab"][aria-selected="true"]{
  color: var(--accent) !important;
  border-bottom: 3px solid var(--accent) !important;
}

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

.stButton button, .stLinkButton a{
  border-radius: 14px !important;
  padding: 0.78rem 1.05rem !important;
  font-weight: 800 !important;
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

.block-container{ padding-top: 1.0rem; }

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

/* Sidebar select container */
section[data-testid="stSidebar"] [data-baseweb="select"] > div{
  background:#ffffff !important;
  border:1px solid rgba(229,231,235,0.35) !important;
  border-radius: 14px !important;
}

/* Selected value inside select */
section[data-testid="stSidebar"] [data-baseweb="select"] *{
  color: var(--text) !important;
}

/* Caret icon */
section[data-testid="stSidebar"] [data-baseweb="select"] svg{
  color: var(--text) !important;
}

/* =========================================================
   FIX: Dropdown menu options unreadable
   BaseWeb renders menu in a portal (popover/menu) OUTSIDE sidebar.
   Force high-contrast text + white background for the menu.
========================================================= */

/* Popover container background */
div[data-baseweb="popover"]{
  background: transparent !important;
}

/* Menu panel */
div[data-baseweb="menu"]{
  background:#ffffff !important;
  border:1px solid rgba(15,23,42,0.12) !important;
  border-radius: 14px !important;
  overflow:hidden !important;
}

/* Menu items text */
div[data-baseweb="menu"] *{
  color:#0f172a !important;     /* FORCE readable text */
}

/* Option hover */
div[data-baseweb="menu"] [role="option"]:hover{
  background: rgba(15,23,42,0.06) !important;
}

/* Slider text */
section[data-testid="stSidebar"] .stSlider *{
  color: var(--sideText) !important;
}
section[data-testid="stSidebar"] [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] [data-testid="stTickBarMax"]{
  color: var(--sideText) !important;
}

/* Radio text */
section[data-testid="stSidebar"] .stRadio label{
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
    return str(val).strip().lower() in {"y","yes","true","1"}

# =========================================================
# Supabase client + save function
# =========================================================
@st.cache_resource(show_spinner=False)
def get_supabase():
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        st.error("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in Streamlit Secrets.")
        st.stop()

    if create_client is None:
        st.error("supabase package not installed. Add `supabase` to requirements.txt.")
        st.stop()

    return create_client(url, key)

def save_to_supabase(rid: str, intake: dict, ai_out: dict):
    sb = get_supabase()
    payload = {
        "audit_id": rid,
        "email": (intake.get("email") or "").strip() or None,
        "athlete_name": (intake.get("name") or "").strip() or None,
        "survey": intake,
        "ai_result": ai_out,
        "status": "created",
    }

    res = sb.table("recommendations").insert(payload).execute()
    if hasattr(res, "error") and res.error:
        raise RuntimeError(str(res.error))
    return res.data[0]["id"] if res.data else None

# =========================================================
# PREMIUM AUDIT ID CARD
# =========================================================
def display_audit_id(rid: str):
    if not rid:
        return
    display_id = "IBEX-" + rid.replace("-", "")[:10].upper()

    html = f"""
    <div style="
        background:#ffffff;
        border:1px solid rgba(15,23,42,0.10);
        border-radius:20px;
        padding:22px 22px 18px 22px;
        box-shadow: 0 18px 45px rgba(2, 6, 23, 0.08);
        margin: 4px 0 18px 0;
    ">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:14px; flex-wrap:wrap;">
        <div>
          <div style="font-size:13px; letter-spacing:0.12em; font-weight:800; color:#64748b;">
            IBEX AUDIT ID
          </div>
          <div style="margin-top:6px; font-size:22px; font-weight:900; color:#0f172a; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;">
            {display_id}
          </div>
          <div style="margin-top:10px; font-size:13px; color:#64748b;">
            Copy this and paste it into Stripe during checkout so we can match your order to your recommendations.
          </div>
        </div>

        <div style="display:flex; flex-direction:column; gap:10px; min-width:240px;">
          <button id="ibexCopyBtn" style="
              background:#ef4444; color:#ffffff; border:none;
              border-radius:14px; padding:12px 14px;
              font-weight:900; cursor:pointer;
              box-shadow: 0 10px 25px rgba(239,68,68,0.20);
          ">
            Copy Audit ID
          </button>

          <div id="ibexCopyNote" style="font-size:12px; color:#64748b; text-align:center; min-height:16px;"></div>
        </div>
      </div>

      <div style="margin-top:14px; padding:10px 12px; border-radius:14px; background:rgba(15,23,42,0.03); border:1px solid rgba(15,23,42,0.06);">
        <div style="font-size:11px; color:#64748b; font-weight:800; letter-spacing:0.10em;">FULL ID (internal)</div>
        <div style="margin-top:6px; font-size:12px; color:#0f172a; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; word-break:break-all;">
          {rid}
        </div>
      </div>
    </div>

    <script>
      const txt = "{display_id}";
      const btn = document.getElementById("ibexCopyBtn");
      const note = document.getElementById("ibexCopyNote");

      btn.addEventListener("click", async () => {{
        try {{
          await navigator.clipboard.writeText(txt);
          note.textContent = "Copied ✓";
          note.style.color = "#16a34a";
          setTimeout(() => {{
            note.textContent = "";
            note.style.color = "#64748b";
          }}, 1400);
        }} catch (e) {{
          note.textContent = "Copy failed — select & copy manually";
          note.style.color = "#b45309";
        }}
      }});
    </script>
    """
    components.html(html, height=210)

# =========================================================
# HEADER
# =========================================================
def render_header():
    logo = load_logo()
    if logo is not None:
        c1, c2 = st.columns([1, 7], gap="large")
        with c1:
            st.image(logo, width=130)
        with c2:
            st.markdown(
                f"<div style='font-size:44px; font-weight:850; color:#0f172a; margin-top:2px;'>{APP_TITLE}</div>",
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
              <div style="font-size:44px; font-weight:850; color:#0f172a;">{APP_TITLE}</div>
              <div class="ibx-muted" style="font-size:16px;">{APP_TAGLINE}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# DATA LOADERS
# =========================================================
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

# =========================================================
# PLAN DEFINITIONS
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
                <div style="margin-top:12px; font-size:18px; font-weight:800; color:#0f172a;">
                  {p['Ingredient']}
                </div>
                <div class="ibx-muted" style="margin-top:2px;">
                  {p['Brand']} • {p['Serving_Form']} • {p['Store']}
                </div>
                <div class="ibx-divider"></div>
                <div style="font-weight:800; color:#0f172a;">Why this</div>
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
            st.markdown(f"<div style='font-size:18px; font-weight:900; color:#0f172a;'>{title}</div>", unsafe_allow_html=True)
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
    support_email = st.secrets.get("SUPPORT_EMAIL", "support@ibexsupplements.com")
    st.markdown(
        f"""
<div class="ibx-card">
  <div style="font-size:26px; font-weight:900; color:#0f172a;">Privacy Policy</div>
  <div class="ibx-muted" style="margin-top:4px;">Effective: {eff}</div>
  <div class="ibx-divider"></div>
  <p>... (unchanged) ...</p>
</div>
""",
        unsafe_allow_html=True
    )

def render_faq():
    support_email = st.secrets.get("SUPPORT_EMAIL", "support@ibexsupplements.com")
    st.markdown(
        f"""
<div class="ibx-card">
  <div style="font-size:26px; font-weight:900; color:#0f172a;">FAQ</div>
  <div class="ibx-divider"></div>
  <p>... (unchanged) ...</p>
</div>
""",
        unsafe_allow_html=True
    )

# =========================================================
# APP START
# =========================================================
def require_file(path: str, friendly: str):
    if not os.path.exists(path):
        st.error(f"Missing {friendly}: `{path}`")
        st.info("Fix: upload the file to your GitHub repo in the correct folder, then reboot the app.")
        st.stop()

require_file(PRODUCTS_CSV, "products.csv (data/products.csv)")
require_file(EXCLUSIONS_CSV, "exclusions.csv (data/exclusions.csv)")
require_file(LOGO_PATH, "logo (assets/ibex_logo.png)")

products = load_products()
exclusions = load_exclusions()

STRIPE_BASIC_LINK = st.secrets.get("STRIPE_BASIC_LINK", "")
STRIPE_PERF_LINK = st.secrets.get("STRIPE_PERF_LINK", "")

if "ai_out" not in st.session_state:
    st.session_state.ai_out = None
if "last_plan" not in st.session_state:
    st.session_state.last_plan = "Basic"
if "last_rid" not in st.session_state:
    st.session_state.last_rid = None

render_header()

tabs = st.tabs(["Audit", "Privacy", "FAQ"])

# =========================================================
# TAB: AUDIT
# =========================================================
with tabs[0]:
    if st.session_state.ai_out:
        ai_out = st.session_state.ai_out
        plan = st.session_state.last_plan

        st.markdown(
            f"""
            <div class="ibx-card">
              <div style="display:flex; justify-content:space-between; align-items:flex-end; gap:12px; flex-wrap:wrap;">
                <div>
                  <div style="font-size:28px; font-weight:950; color:#0f172a;">Your {plan} System</div>
                  <div class="ibx-muted">Reference ID: {st.session_state.last_rid}</div>
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

        display_audit_id(st.session_state.last_rid)

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
        st.caption("Copy your IBEX Audit ID above and paste it into Stripe during checkout.")

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

            # ✅ CHANGE: remove checkbox gating (trust barrier)
            st.caption("Not medical advice. For details, see the Privacy tab.")

            submitted = st.form_submit_button("Build my system")

        if submitted:
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

            # ✅ Keep DB save (but don't block results)
            try:
                row_id = save_to_supabase(rid, intake, ai_out)
                st.sidebar.success("Saved ✅")
            except Exception as e:
                st.sidebar.error("Save failed (DB)")
                st.sidebar.code(str(e))

            st.session_state.ai_out = ai_out
            st.session_state.last_plan = plan
            st.session_state.last_rid = rid
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








