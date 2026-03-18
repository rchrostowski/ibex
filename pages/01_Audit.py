"""
IBEX — Performance Audit App
─────────────────────────────
This is pages/01_Audit.py (renamed from app.py)
Place alongside pages/00_Home.py in the pages/ folder.
"""

import os
import json
import uuid
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

try:
    from supabase import create_client
except Exception:
    create_client = None

# =========================================================
# CONFIG
# =========================================================
APP_TITLE    = "IBEX"
APP_TAGLINE  = "Personalized performance systems for D1 athletes"
PRODUCTS_CSV = "data/products.csv"
EXCLUSIONS_CSV = "data/exclusions.csv"
LOGO_PATH    = "assets/ibex_logo.png"

st.set_page_config(
    page_title=f"{APP_TITLE} • Performance Audit",
    page_icon=LOGO_PATH,
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# DARK NAVY / GOLD THEME  (matches landing page)
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Barlow:wght@300;400;600;700&family=Barlow+Condensed:wght@400;500;600;700;800&display=swap');

:root {
  --black:  #0a0a0f;
  --navy:   #0b1220;
  --navy2:  #132033;
  --off:    #f0ede6;
  --gold:   #c9a84c;
  --gold2:  #e8c97a;
  --muted:  rgba(240,237,230,0.52);
  --border: rgba(201,168,76,0.18);
  --red:    #ef4444;
}

/* ── GLOBAL ── */
#MainMenu, footer, header { visibility: hidden; }
html, body, [class*="css"] {
  font-family: 'Barlow', sans-serif !important;
  font-weight: 300;
}
.stApp { background: var(--navy) !important; }
.block-container { padding-top: 1.2rem !important; }

h1,h2,h3,h4,h5 { color: var(--off) !important; font-family: 'Bebas Neue', sans-serif !important; letter-spacing: 0.03em; }
p, li, span, div, label { color: var(--muted); }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
  background: var(--black) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--off) !important; }
section[data-testid="stSidebar"] .stMarkdown p { color: var(--muted) !important; }
section[data-testid="stSidebar"] a { color: var(--gold) !important; }

/* sidebar inputs */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
  background: var(--navy) !important;
  color: var(--off) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}
section[data-testid="stSidebar"] input::placeholder,
section[data-testid="stSidebar"] textarea::placeholder {
  color: var(--muted) !important; opacity: 1 !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: var(--navy) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] * { color: var(--off) !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] svg { color: var(--gold) !important; }

/* dropdown menu */
div[data-baseweb="popover"] { background: transparent !important; }
div[data-baseweb="menu"] {
  background: var(--navy2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}
div[data-baseweb="menu"] * { color: var(--off) !important; }
div[data-baseweb="menu"] [role="option"]:hover { background: rgba(201,168,76,0.1) !important; }

/* slider */
section[data-testid="stSidebar"] .stSlider * { color: var(--off) !important; }
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
  background: var(--gold) !important;
  border-color: var(--gold) !important;
}
[data-testid="stSlider"] div[class*="StyledSliderBar"] { background: var(--gold) !important; }

/* radio */
section[data-testid="stSidebar"] .stRadio label { color: var(--off) !important; }
section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] div {
  border-color: var(--gold) !important;
}
section[data-testid="stSidebar"] .stRadio [aria-checked="true"] div {
  background: var(--gold) !important;
  border-color: var(--gold) !important;
}

/* checkbox */
section[data-testid="stSidebar"] .stCheckbox label { color: var(--off) !important; }
section[data-testid="stSidebar"] .stCheckbox [data-baseweb="checkbox"] div {
  border-color: var(--gold) !important;
  background: transparent !important;
}
section[data-testid="stSidebar"] .stCheckbox [aria-checked="true"] div {
  background: var(--gold) !important;
}

/* number input */
section[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
  background: var(--navy) !important;
  border: 1px solid var(--border) !important;
  color: var(--off) !important;
}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
  background: var(--navy2) !important;
  border-color: var(--border) !important;
  color: var(--gold) !important;
}

/* multiselect */
section[data-testid="stSidebar"] [data-baseweb="tag"] {
  background: rgba(201,168,76,0.15) !important;
  border-color: var(--gold) !important;
}
section[data-testid="stSidebar"] [data-baseweb="tag"] span { color: var(--gold) !important; }

/* ── BUTTONS ── */
.stButton button {
  background: var(--gold) !important;
  color: var(--black) !important;
  border: none !important;
  border-radius: 4px !important;
  font-family: 'Barlow Condensed', sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  padding: 0.7rem 1.4rem !important;
  transition: background 0.2s !important;
}
.stButton button:hover { background: var(--gold2) !important; }

.stLinkButton a {
  background: var(--navy2) !important;
  color: var(--gold) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  font-family: 'Barlow Condensed', sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  padding: 0.7rem 1.4rem !important;
}
.stLinkButton a:hover { border-color: var(--gold) !important; }

/* form submit button */
[data-testid="stFormSubmitButton"] button {
  background: var(--gold) !important;
  color: var(--black) !important;
  border: none !important;
  border-radius: 4px !important;
  font-family: 'Barlow Condensed', sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  padding: 0.8rem 1.6rem !important;
  width: 100% !important;
}

/* ── TABS ── */
button[data-baseweb="tab"] {
  font-family: 'Barlow Condensed', sans-serif !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  font-weight: 600 !important;
  color: var(--muted) !important;
  background: transparent !important;
  border-bottom: 2px solid transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--gold) !important;
  border-bottom: 2px solid var(--gold) !important;
}
[data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0.5rem !important;
}
[data-baseweb="tab-panel"] { background: transparent !important; }

/* ── CUSTOM CARDS ── */
.ibx-card {
  background: var(--black);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 24px;
  margin-bottom: 16px;
  transition: border-color 0.3s;
}
.ibx-card:hover { border-color: rgba(201,168,76,0.35); }

.ibx-badge {
  display: inline-block;
  padding: 4px 10px;
  border: 1px solid var(--border);
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 0.62rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--gold) !important;
  margin-right: 6px;
  margin-bottom: 4px;
}
.ibx-badge.green { border-color: rgba(74,222,128,0.35); color: #4ade80 !important; }
.ibx-badge.blue  { border-color: rgba(125,211,252,0.35); color: #7dd3fc !important; }

.ibx-divider {
  height: 1px;
  background: var(--border);
  margin: 14px 0;
}
.ibx-muted { color: var(--muted) !important; }

.ibx-label {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 0.62rem;
  letter-spacing: 0.35em;
  text-transform: uppercase;
  color: var(--gold) !important;
  margin-bottom: 0.4rem;
}
.ibx-title {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 2rem;
  color: var(--off) !important;
  letter-spacing: 0.03em;
  line-height: 1;
}

/* ── ALERTS ── */
.stWarning { background: rgba(201,168,76,0.08) !important; border-color: var(--gold) !important; border-radius: 4px !important; }
.stWarning * { color: var(--gold) !important; }
.stInfo { background: rgba(11,18,32,0.8) !important; border-color: var(--border) !important; border-radius: 4px !important; }
.stInfo * { color: var(--muted) !important; }
.stSuccess { background: rgba(74,222,128,0.08) !important; border-color: rgba(74,222,128,0.3) !important; border-radius: 4px !important; }
.stError { background: rgba(239,68,68,0.08) !important; border-color: rgba(239,68,68,0.3) !important; border-radius: 4px !important; }

/* ── CHAT ── */
[data-testid="stChatMessage"] {
  background: var(--black) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  margin-bottom: 8px !important;
}
[data-testid="stChatMessage"] * { color: var(--off) !important; }
[data-testid="stChatInputContainer"] {
  background: var(--black) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
}
[data-testid="stChatInputContainer"] * { color: var(--off) !important; }
[data-testid="stChatInputContainer"] textarea { background: transparent !important; }

/* ── FAQ ACCORDION ── */
.ibx-faq details {
  background: var(--black);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 14px 16px;
  margin: 8px 0;
  transition: border-color 0.3s;
}
.ibx-faq details[open] { border-color: rgba(201,168,76,0.4); }
.ibx-faq summary {
  list-style: none; cursor: pointer;
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
  font-family: 'Barlow Condensed', sans-serif;
  font-weight: 700; font-size: 15px; letter-spacing: 0.03em;
  color: var(--off) !important; outline: none;
}
.ibx-faq summary::-webkit-details-marker { display: none; }
.ibx-faq .qhint { color: var(--muted) !important; font-size: 11px; font-weight: 400; margin-top: 2px; }
.ibx-faq .answer { margin-top: 12px; color: var(--muted) !important; line-height: 1.65; font-size: 14px; }
.ibx-faq .chev {
  width: 30px; height: 30px; border-radius: 4px;
  display: flex; align-items: center; justify-content: center;
  background: rgba(201,168,76,0.08); border: 1px solid var(--border);
  flex-shrink: 0; color: var(--gold) !important;
}
.ibx-faq details[open] .chev { background: rgba(201,168,76,0.15); border-color: rgba(201,168,76,0.4); }
.ibx-faq .pill {
  display: inline-block; padding: 4px 8px; border-radius: 2px;
  background: rgba(201,168,76,0.08); border: 1px solid var(--border);
  color: var(--gold) !important; font-size: 11px; font-weight: 700;
  font-family: 'Barlow Condensed', sans-serif; letter-spacing: 0.1em;
  text-transform: uppercase; margin-right: 6px;
}

/* ── MAIN AREA BACKGROUND ── */
.main .block-container { background: var(--navy) !important; }

/* ── SPINNER ── */
[data-testid="stSpinner"] * { color: var(--gold) !important; }

/* sidebar caption/small text */
section[data-testid="stSidebar"] .stCaption p { color: var(--muted) !important; font-size: 0.75rem !important; }
section[data-testid="stSidebar"] hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS  (unchanged from original)
# =========================================================
def require_file(path, friendly):
    if not os.path.exists(path):
        st.error(f"Missing {friendly}: `{path}`")
        st.info("Fix: upload the file to your GitHub repo in the correct folder, then reboot the app.")
        st.stop()

def load_logo():
    if not os.path.exists(LOGO_PATH): return None
    if Image is None: return None
    try: return Image.open(LOGO_PATH)
    except: return None

def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY in Streamlit Secrets.")
        st.stop()
    if OpenAI is None:
        st.error("openai package not installed.")
        st.stop()
    return OpenAI(api_key=api_key)

def is_yes(val):
    return str(val).strip().lower() in {"y","yes","true","1"}

def parse_money(val):
    try:
        s = str(val).strip().replace("$","").replace(",","")
        if not s: return 0.0
        x = float(s)
        return 0.0 if x != x else x
    except: return 0.0

def norm_key(val):
    s = str(val or "").strip().lower()
    return " ".join(s.split())

# =========================================================
# SUPABASE
# =========================================================
@st.cache_resource(show_spinner=False)
def get_supabase():
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        st.error("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY.")
        st.stop()
    if create_client is None:
        st.error("supabase package not installed.")
        st.stop()
    return create_client(url, key)

def save_to_supabase(rid, intake, ai_out):
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
    if hasattr(res,"error") and res.error: raise RuntimeError(str(res.error))
    return res.data[0]["id"] if res.data else None

# =========================================================
# EVIDENCE
# =========================================================
def get_evidence_link(row):
    ev = str(row.get("Evidence_Link","") or "").strip()
    return ev if ev else str(row.get("Link","") or "").strip()

def evidence_enabled():
    return str(st.secrets.get("EVIDENCE_LINKS_ENABLED","true")).strip().lower() in {"1","true","yes","y"}

# =========================================================
# PLAN DEFINITIONS
# =========================================================
PLAN_COPY = {
    "Basic": {
        "headline": "Foundations, done right.",
        "sub": "A clean, conservative system built from the essentials. Minimal complexity. Maximum consistency.",
        "bullets": [
            "Core performance stack only (the stuff that actually works)",
            "Prefers NSF Certified for Sport / third-party tested",
            "Designed for consistency and simplicity",
        ],
        "note": "Best for: most college athletes who want a safe, no-BS baseline.",
    },
    "Performance": {
        "headline": "Optimization mode.",
        "sub": "A deeper system with expanded options based on your full audit.",
        "bullets": [
            "Expanded catalog (advanced recovery, sleep, gut, joint support)",
            "More conditional logic: schedule + training load + sensitivities",
            "Built for athletes chasing marginal gains (without sketchy stuff)",
        ],
        "note": "Best for: high-volume training, in-season stress, or athletes who want every edge.",
    },
}

BASIC_CORE_CATEGORIES = {
    "Creatine","Omega-3","Magnesium","Vitamin D","Electrolytes","Protein",
    "Multivitamin","Zinc","Vitamin C","Probiotic","Fiber","Collagen","Tart Cherry"
}

PLAN_LIMITS = {
    "Basic":       {"max_units":5,"supp_budget":39.0,"max_am":3,"max_pm":2,"max_training":2},
    "Performance": {"max_units":8,"supp_budget":69.0,"max_am":3,"max_pm":3,"max_training":2},
}

def item_units(monthly_cost):
    return 2 if monthly_cost >= 20.0 else 1

# =========================================================
# AUDIT ID CARD
# =========================================================
def display_audit_id(rid):
    if not rid: return
    display_id = "IBEX-" + rid.replace("-","")[:10].upper()
    html = f"""
    <div style="
        background:#0a0a0f;
        border:1px solid rgba(201,168,76,0.35);
        padding:20px 22px 16px;
        margin:4px 0 18px;
    ">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:14px;flex-wrap:wrap;">
        <div>
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.62rem;letter-spacing:0.35em;text-transform:uppercase;color:#c9a84c;">
            IBEX AUDIT ID
          </div>
          <div style="margin-top:6px;font-size:1.5rem;font-weight:900;color:#f0ede6;font-family:ui-monospace,monospace;">
            {display_id}
          </div>
          <div style="margin-top:8px;font-size:0.8rem;color:rgba(240,237,230,0.5);">
            Copy this and paste it into Stripe during checkout so we can match your order.
          </div>
        </div>
        <div style="display:flex;flex-direction:column;gap:8px;min-width:200px;">
          <button id="ibexCopyBtn" style="
            background:#c9a84c;color:#0a0a0f;border:none;
            padding:10px 14px;font-family:'Barlow Condensed',sans-serif;
            font-size:0.72rem;letter-spacing:0.2em;text-transform:uppercase;font-weight:700;
            cursor:pointer;
          ">Copy Audit ID</button>
          <div id="ibexCopyNote" style="font-size:0.72rem;color:rgba(240,237,230,0.5);text-align:center;min-height:14px;"></div>
        </div>
      </div>
      <div style="margin-top:12px;padding:8px 10px;background:rgba(201,168,76,0.06);border:1px solid rgba(201,168,76,0.12);">
        <div style="font-size:0.58rem;color:rgba(240,237,230,0.4);font-family:'Barlow Condensed',sans-serif;letter-spacing:0.15em;text-transform:uppercase;">FULL ID</div>
        <div style="margin-top:4px;font-size:0.72rem;color:rgba(240,237,230,0.6);font-family:ui-monospace,monospace;word-break:break-all;">{rid}</div>
      </div>
    </div>
    <script>
      const btn=document.getElementById("ibexCopyBtn");
      const note=document.getElementById("ibexCopyNote");
      btn.addEventListener("click",async()=>{{
        try{{
          await navigator.clipboard.writeText("{display_id}");
          note.textContent="Copied ✓"; note.style.color="#c9a84c";
          setTimeout(()=>{{note.textContent="";note.style.color="rgba(240,237,230,0.5)"}},1400);
        }}catch(e){{note.textContent="Select & copy manually";}}
      }});
    </script>
    """
    components.html(html, height=200)

# =========================================================
# HEADER
# =========================================================
def render_header():
    logo = load_logo()
    st.markdown("""
    <div style="border-bottom:1px solid rgba(201,168,76,0.18);padding-bottom:1.2rem;margin-bottom:1.2rem;">
    """, unsafe_allow_html=True)

    if logo is not None:
        c1, c2 = st.columns([1,8], gap="medium")
        with c1:
            st.image(logo, width=80)
        with c2:
            st.markdown(f"""
            <div style="padding-top:4px;">
              <div style="font-family:'Bebas Neue',sans-serif;font-size:2.8rem;color:#f0ede6;letter-spacing:0.05em;line-height:1;">{APP_TITLE}</div>
              <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.7rem;letter-spacing:0.3em;text-transform:uppercase;color:#c9a84c;margin-top:2px;">{APP_TAGLINE}</div>
              <div style="margin-top:10px;">
                <span class="ibx-badge green">NCAA Compliant</span>
                <span class="ibx-badge blue">NSF Tested</span>
                <span class="ibx-badge">Evidence-Linked</span>
                <span class="ibx-badge">AI-Personalized</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div>
          <div style="font-family:'Bebas Neue',sans-serif;font-size:2.8rem;color:#f0ede6;letter-spacing:0.05em;">{APP_TITLE}</div>
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.7rem;letter-spacing:0.3em;text-transform:uppercase;color:#c9a84c;">{APP_TAGLINE}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# DATA LOADERS  (unchanged)
# =========================================================
@st.cache_data(show_spinner=False)
def load_products():
    df = pd.read_csv(PRODUCTS_CSV)
    df.columns = [c.strip() for c in df.columns]
    required = ["Product_ID","Category","Ingredient","Brand","Store","Link",
                "Serving_Form","Typical_Use","Timing","Avoid_If",
                "Third_Party_Tested","NSF_Certified","Price","Est_Monthly_Cost","Notes"]
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f"products.csv missing columns: {missing}")
    if "Evidence_Link"    not in df.columns: df["Evidence_Link"]    = ""
    if "NCAA_Risk_Tier"   not in df.columns: df["NCAA_Risk_Tier"]   = ""
    if "Athlete_Safe_OK"  not in df.columns: df["Athlete_Safe_OK"]  = ""
    return df

@st.cache_data(show_spinner=False)
def load_exclusions():
    df = pd.read_csv(EXCLUSIONS_CSV)
    df.columns = [c.strip() for c in df.columns]
    if "Excluded_Category_or_Ingredient" not in df.columns or "Reason" not in df.columns:
        raise ValueError("exclusions.csv must have: Excluded_Category_or_Ingredient, Reason")
    return df

# =========================================================
# FILTERS  (unchanged)
# =========================================================
def filter_products_by_plan(products, plan):
    p = products.copy()
    p["Category_norm"] = p["Category"].astype(str).str.strip()
    if plan == "Basic":
        return p[p["Category_norm"].isin(BASIC_CORE_CATEGORIES)]
    return p

def filter_ncaa_safe(products, plan):
    p = products.copy()
    if "Athlete_Safe_OK" in p.columns:
        p = p[p["Athlete_Safe_OK"].astype(str).str.strip().str.upper().isin({"Y","YES","TRUE","1"})]
    if "NCAA_Risk_Tier" in p.columns:
        tier = p["NCAA_Risk_Tier"].astype(str).str.strip().str.lower()
        if plan == "Basic":
            p = p[tier.eq("green") | tier.eq("")]
        else:
            p = p[tier.isin({"green","yellow"}) | tier.eq("")]
    return p

def shortlist_products(products, goals, gi_sensitive, caffeine_sensitive, plan):
    products = filter_ncaa_safe(products, plan)
    p = filter_products_by_plan(products, plan)
    if goals:
        mask = False
        for g in goals:
            mask = mask | p["Typical_Use"].astype(str).str.contains(g, case=False, na=False)
        if mask is not False: p = p[mask]
    if gi_sensitive:
        p = p[~p["Avoid_If"].astype(str).str.contains("GI", case=False, na=False)]
    if caffeine_sensitive:
        p = p[~p["Avoid_If"].astype(str).str.contains("caffeine", case=False, na=False)]
    if plan == "Basic":
        p = p.assign(
            nsf=p["NSF_Certified"].apply(is_yes),
            tpt=p["Third_Party_Tested"].apply(lambda x: str(x).strip().lower() in {"y","yes","true","1","unknown"})
        ).sort_values(["nsf","tpt"], ascending=[False,False]).drop(columns=["nsf","tpt"])
    if len(p) < 25:
        p = filter_products_by_plan(products, plan).copy()
    cap = 55 if plan == "Basic" else 85
    return p.head(cap)

# =========================================================
# AI  (unchanged)
# =========================================================
def run_ai(intake, products_shortlist, exclusions, plan):
    client = get_openai_client()
    approved_products = products_shortlist[[
        "Product_ID","Category","Ingredient","Brand","Store","Link","Evidence_Link",
        "Serving_Form","Typical_Use","Timing","Avoid_If",
        "Third_Party_Tested","NSF_Certified","Notes","Est_Monthly_Cost"
    ]].to_dict(orient="records")

    output_schema = {
        "flags":["string"],"consult_professional":"boolean",
        "included_product_ids":["IBX-0001"],"excluded_product_ids":["IBX-0002"],
        "schedule":{"AM":["IBX-0001"],"PM":["IBX-0003"],"Training":["IBX-0004"]},
        "reasons":{"IBX-0001":"short non-medical reason"},
        "notes_for_athlete":["bullet"]
    }

    plan_rules = (
        "Plan: BASIC. Conservative and foundational. Keep stack simple. Prefer NSF/third-party tested."
        if plan == "Basic" else
        "Plan: PERFORMANCE. Expanded optimization. May add conditional advanced items if clearly supported."
    )
    lim = PLAN_LIMITS.get(plan, {"max_units":6,"supp_budget":50.0,"max_am":3,"max_pm":3,"max_training":2})

    system_prompt = (
        "You are IBEX, an assistant that organizes a personalized supplement system for athletes. "
        "You are NOT a medical provider. Do NOT diagnose, treat, or make medical claims. "
        "Only select products from approved_products. Never select anything matching the exclusions list. "
        "Evidence links are per product in approved_products as Evidence_Link. "
        "Do NOT invent papers, DOIs, authors, or citations. "
        "If intake mentions serious symptoms, medications, or a medical condition, set consult_professional=true. "
        f"{plan_rules} "
        f"STACK CAPS: max {lim['max_units']} units (Est_Monthly_Cost >= $20 = 2 units). "
        f"AM ≤ {lim['max_am']}, PM ≤ {lim['max_pm']}, Training ≤ {lim['max_training']}. "
        "CRITICAL: No duplicate ingredients. One product per ingredient (case-insensitive). "
        "Return ONLY valid JSON matching output_format schema."
    )

    payload = {"plan":plan,"intake":intake,"approved_products":approved_products,
               "exclusions":exclusions.to_dict(orient="records"),"output_format":output_schema}

    model = st.secrets.get("OPENAI_MODEL","gpt-4.1-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user","content":json.dumps(payload)}],
        temperature=0.2
    )
    content = resp.choices[0].message.content.strip()
    try: return json.loads(content)
    except:
        s, e = content.find("{"), content.rfind("}")
        if s != -1 and e != -1 and e > s: return json.loads(content[s:e+1])
        raise

# =========================================================
# ENFORCE CAPS  (unchanged)
# =========================================================
def enforce_caps(ai_out, plan, products_df):
    lim = PLAN_LIMITS.get(plan, {"max_units":6,"supp_budget":50.0,"max_am":3,"max_pm":3,"max_training":2})
    included = ai_out.get("included_product_ids",[]) or []
    schedule = ai_out.get("schedule",{}) or {}
    reasons  = ai_out.get("reasons",{}) or {}
    notes    = ai_out.get("notes_for_athlete",[]) or []
    if not included: return ai_out

    prod_map    = products_df.set_index("Product_ID").to_dict(orient="index")
    model_order = {pid:i for i,pid in enumerate(included)}

    rows = []
    for pid in included:
        p    = prod_map.get(pid,{})
        cat  = str(p.get("Category","") or "").strip()
        ing  = str(p.get("Ingredient","") or "").strip()
        est  = parse_money(p.get("Est_Monthly_Cost",0))
        nsf  = is_yes(p.get("NSF_Certified",""))
        tpt  = str(p.get("Third_Party_Tested","")).strip().lower() in {"y","yes","true","1","unknown"}
        core = cat in BASIC_CORE_CATEGORIES
        dup_key = norm_key(ing) if norm_key(ing) else f"cat::{norm_key(cat)}"
        rows.append({"pid":pid,"cat":cat,"ing":ing,"est":float(est or 0.0),
                     "nsf":bool(nsf),"tpt":bool(tpt),"core":bool(core),
                     "order":model_order.get(pid,9999),"dup_key":dup_key})

    rows.sort(key=lambda r:(not r["core"],not r["nsf"],not r["tpt"],r["order"]))

    kept,seen_keys,dropped = [],[],[]
    for r in rows:
        if r["dup_key"] in seen_keys: dropped.append(r); continue
        kept.append(r); seen_keys.add(r["dup_key"])

    if dropped:
        msg = "We removed duplicate forms of the same supplement to keep your stack clean."
        if msg not in notes: notes = [msg]+notes

    picked,used_units = [],0
    for r in kept:
        units = item_units(r["est"])
        if used_units+units > lim["max_units"]: continue
        picked.append(r["pid"]); used_units += units

    if not picked:
        for r in kept:
            units = item_units(r["est"])
            if used_units+units > lim["max_units"]: continue
            picked.append(r["pid"]); used_units += units
            if used_units >= lim["max_units"]: break

    picked_set = set(picked)
    def trim_bucket(items, maxn):
        out=[]
        for pid in (items or []):
            if pid in picked_set and pid not in out: out.append(pid)
            if len(out)>=maxn: break
        return out

    new_schedule = {
        "AM":       trim_bucket(schedule.get("AM",[]),      lim["max_am"]),
        "PM":       trim_bucket(schedule.get("PM",[]),      lim["max_pm"]),
        "Training": trim_bucket(schedule.get("Training",[]),lim["max_training"]),
    }
    new_reasons = {pid:reasons.get(pid,"") for pid in picked if pid in reasons}

    if len(picked) < len(included):
        msg = "To keep the stack practical, we capped the number of recommended items for your plan."
        if msg not in notes: notes = [msg]+notes

    ai_out["included_product_ids"] = picked
    ai_out["schedule"]             = new_schedule
    ai_out["reasons"]              = new_reasons
    ai_out["notes_for_athlete"]    = notes
    ai_out["meta_caps"]            = {"plan":plan,"units_selected":used_units,"max_units":lim["max_units"]}
    return ai_out

# =========================================================
# PRODUCT CARDS  — dark theme
# =========================================================
def render_products(product_ids, products_df, reasons):
    prod_map = products_df.set_index("Product_ID").to_dict(orient="index")
    cols = st.columns(3, gap="large")

    for i, pid in enumerate(product_ids):
        p = prod_map.get(pid)
        if not p: continue
        ev = get_evidence_link(p)

        with cols[i % 3]:
            nsf_badge  = '<span class="ibx-badge blue">NSF ✓</span>'  if is_yes(p.get("NSF_Certified",""))   else ""
            ncaa_badge = '<span class="ibx-badge green">NCAA ✓</span>'

            st.markdown(f"""
            <div class="ibx-card">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:6px;">
                <span class="ibx-badge">{p.get('Category','')}</span>
                <span class="ibx-badge">{p.get('Timing','')}</span>
              </div>
              <div style="margin-top:10px;">
                {ncaa_badge}{nsf_badge}
              </div>
              <div style="margin-top:12px;font-family:'Bebas Neue',sans-serif;font-size:1.5rem;color:#f0ede6;letter-spacing:0.03em;line-height:1.1;">
                {p.get('Ingredient','')}
              </div>
              <div style="font-size:0.78rem;color:rgba(240,237,230,0.45);margin-top:2px;">{p.get('Serving_Form','')}</div>
              <div class="ibx-divider"></div>
              <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.65rem;letter-spacing:0.2em;text-transform:uppercase;color:#c9a84c;">Why this</div>
              <div style="font-size:0.85rem;color:rgba(240,237,230,0.65);margin-top:4px;line-height:1.6;">
                {reasons.get(pid,'Personalized to your audit')}
              </div>
            </div>
            """, unsafe_allow_html=True)

            if evidence_enabled():
                if ev:
                    st.link_button("Open the linked study →", ev)
                else:
                    st.caption("No evidence link attached for this item yet.")

# =========================================================
# SCHEDULE  — dark theme
# =========================================================
def render_schedule(schedule, products_df):
    prod_map = products_df.set_index("Product_ID").to_dict(orient="index")
    blocks   = [("AM","Morning","Foundation window"),("PM","Evening","Recovery window"),("Training","Training","Performance window")]
    cols     = st.columns(3, gap="large")

    for i, (key, title, sub) in enumerate(blocks):
        with cols[i]:
            items = schedule.get(key,[]) if isinstance(schedule,dict) else []
            items_html = ""
            if not items:
                items_html = '<div style="color:rgba(240,237,230,0.3);font-size:0.85rem;">—</div>'
            else:
                for pid in items:
                    p = prod_map.get(pid,{})
                    items_html += f"""
                    <div style="display:flex;justify-content:space-between;padding:0.4rem 0;border-bottom:1px solid rgba(201,168,76,0.08);font-size:0.85rem;">
                      <span style="color:rgba(240,237,230,0.8);">{p.get('Ingredient',pid)}</span>
                      <span style="color:#c9a84c;font-size:0.72rem;font-family:'Barlow Condensed',sans-serif;">{p.get('Serving_Form','')}</span>
                    </div>"""

            st.markdown(f"""
            <div class="ibx-card">
              <div style="font-family:'Bebas Neue',sans-serif;font-size:1.6rem;color:#c9a84c;line-height:1;">{title}</div>
              <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.62rem;letter-spacing:0.2em;text-transform:uppercase;color:rgba(240,237,230,0.35);margin-bottom:12px;">{sub}</div>
              {items_html}
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# PRIVACY  (unchanged logic, restyled)
# =========================================================
def render_privacy_policy():
    eff = date.today().strftime("%B %d, %Y")
    support_email = st.secrets.get("SUPPORT_EMAIL","support@ibexsupplements.com")
    st.markdown(f"""
    <div class="ibx-card" style="max-width:860px;">
      <div class="ibx-label">Legal</div>
      <div class="ibx-title" style="margin-bottom:0.5rem;">Privacy Policy</div>
      <div style="font-size:0.78rem;color:rgba(240,237,230,0.35);margin-bottom:1.5rem;">Effective: {eff}</div>
      <div class="ibx-divider"></div>
      <div style="font-size:0.88rem;line-height:1.8;color:rgba(240,237,230,0.6);">

        <p><b style="color:#f0ede6;">IBEX</b> ("we," "us," or "our") provides a performance audit and supplement planning experience for athletes.</p>

        <p style="margin-top:1.2rem;font-family:'Barlow Condensed',sans-serif;font-size:1rem;font-weight:700;letter-spacing:0.05em;color:#f0ede6;">1) What we collect</p>
        <ul style="margin-left:1.2rem;">
          <li>Contact info: name and email (optional).</li>
          <li>Training & lifestyle inputs: sport, season, training frequency, goals, sleep, stress, soreness, sensitivities.</li>
          <li>App usage data: audit reference ID and basic logs.</li>
        </ul>

        <p style="margin-top:1.2rem;font-family:'Barlow Condensed',sans-serif;font-size:1rem;font-weight:700;letter-spacing:0.05em;color:#f0ede6;">2) What we do not collect</p>
        <ul style="margin-left:1.2rem;">
          <li>We do not require student IDs or social security numbers.</li>
          <li>We do not sell personal information.</li>
        </ul>

        <p style="margin-top:1.2rem;font-family:'Barlow Condensed',sans-serif;font-size:1rem;font-weight:700;letter-spacing:0.05em;color:#f0ede6;">3) How we use your information</p>
        <ul style="margin-left:1.2rem;">
          <li>Generate your recommended system and timing schedule.</li>
          <li>Power the Ask IBEX chat using your audit as context.</li>
          <li>Improve reliability and safety controls.</li>
        </ul>

        <p style="margin-top:1.2rem;font-family:'Barlow Condensed',sans-serif;font-size:1rem;font-weight:700;letter-spacing:0.05em;color:#f0ede6;">4) AI processing</p>
        <p>Your inputs may be sent to an AI provider to generate structured output. We instruct the model to avoid medical diagnosis and to not invent research citations.</p>

        <p style="margin-top:1.2rem;font-family:'Barlow Condensed',sans-serif;font-size:1rem;font-weight:700;letter-spacing:0.05em;color:#f0ede6;">5) NCAA / sport compliance notice</p>
        <p>IBEX is athlete-safe oriented, but no supplement can be guaranteed compliant for any league or test. Always confirm with your athletic department.</p>

        <p style="margin-top:1.2rem;font-family:'Barlow Condensed',sans-serif;font-size:1rem;font-weight:700;letter-spacing:0.05em;color:#f0ede6;">6–12) Full details</p>
        <p>Data is stored securely. We share only with service providers needed to run IBEX. You may request deletion by emailing <b style="color:#c9a84c;">{support_email}</b> with your IBEX Audit ID.</p>

      </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# FAQ  (restyled)
# =========================================================
def render_faq():
    support_email = st.secrets.get("SUPPORT_EMAIL","support@ibexsupplements.com")
    st.markdown(f"""
    <div class="ibx-card" style="max-width:860px;">
      <div class="ibx-label">Help</div>
      <div class="ibx-title" style="margin-bottom:1.2rem;">FAQ</div>
      <div class="ibx-divider"></div>
      <div class="ibx-faq">

        <details open>
          <summary><div>What is IBEX?<div class="qhint">Audit → system → schedule</div></div><div class="chev">⌄</div></summary>
          <div class="answer">IBEX builds a personalized supplement system for D1 athletes. You answer questions about training, recovery, and goals — IBEX generates a plan using only items from a curated, NCAA-safe catalog.</div>
        </details>

        <details>
          <summary><div>Is IBEX medical advice?<div class="qhint">No diagnosis, no treatment</div></div><div class="chev">⌄</div></summary>
          <div class="answer">No. IBEX is not a medical provider and does not diagnose or treat conditions. Consult a qualified professional if you have symptoms, medications, or a medical condition.</div>
        </details>

        <details>
          <summary><div>How does IBEX choose supplements?<div class="qhint">AI constrained to your allowed list</div></div><div class="chev">⌄</div></summary>
          <div class="answer">IBEX uses your audit inputs and matches them to items in the catalog. The AI only selects from the approved products — it cannot invent or add items outside the catalog.</div>
        </details>

        <details>
          <summary><div>Why do you show Evidence links?<div class="qhint">Sources only when they exist</div></div><div class="chev">⌄</div></summary>
          <div class="answer">IBEX only shows evidence links that are attached per product in the catalog. If an item has no link, IBEX says so. No invented citations — ever.</div>
        </details>

        <details>
          <summary><div>Can I ask questions about my plan?<div class="qhint">Yes — use Ask IBEX</div></div><div class="chev">⌄</div></summary>
          <div class="answer">Yes. Use the Ask IBEX tab to ask about timing, stacking, tradeoffs, travel, and season adjustments. The chat is grounded in your audit and your recommended items.</div>
        </details>

        <details>
          <summary><div>Is this safe for NCAA athletes?<div class="qhint">Designed for it — but always verify</div></div><div class="chev">⌄</div></summary>
          <div class="answer">
            IBEX is designed to be athlete-safe, but no supplement can be guaranteed compliant for every league or test. Always check with your athletic department.
            <div style="margin-top:10px;">
              <span class="pill">Prefer third-party tested</span>
              <span class="pill">Avoid blends</span>
              <span class="pill">Keep lot numbers</span>
            </div>
          </div>
        </details>

        <details>
          <summary><div>How do I delete my data?<div class="qhint">Email support with your Audit ID</div></div><div class="chev">⌄</div></summary>
          <div class="answer">Email <b style="color:#c9a84c;">{support_email}</b> with your IBEX Audit ID and request deletion.</div>
        </details>

      </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# CHAT HELPERS  (unchanged)
# =========================================================
def build_chat_context(intake, ai_out, products_df):
    prod_map = products_df.set_index("Product_ID").to_dict(orient="index")
    included = ai_out.get("included_product_ids",[]) or []
    schedule = ai_out.get("schedule",{}) or {}
    reasons  = ai_out.get("reasons",{}) or {}
    items = []
    for pid in included:
        p = prod_map.get(pid,{})
        items.append({"Product_ID":pid,"Category":p.get("Category",""),
                      "Ingredient":p.get("Ingredient",""),"Timing":p.get("Timing",""),
                      "Serving_Form":p.get("Serving_Form",""),"Reason":reasons.get(pid,""),
                      "Evidence_Link":get_evidence_link(p),"Notes":p.get("Notes","")})
    return {"intake":intake,"recommendations":items,"schedule":schedule,
            "notes_for_athlete":ai_out.get("notes_for_athlete",[]),
            "flags":ai_out.get("flags",[]),
            "consult_professional":bool(ai_out.get("consult_professional",False))}

def run_chat_answer(messages, context):
    client = get_openai_client()
    model  = st.secrets.get("OPENAI_CHAT_MODEL", st.secrets.get("OPENAI_MODEL","gpt-4.1-mini"))
    system = (
        "You are IBEX Chat, an athlete-safe assistant. "
        "You are NOT a medical provider. Do not diagnose or provide medical treatment advice. "
        "Use only the provided context. DO NOT invent studies, authors, DOIs, or citations. "
        "If a recommended item has Evidence_Link, you may reference it. "
        "Keep answers practical, short, and athlete-friendly."
    )
    full = [{"role":"system","content":system},
            {"role":"user","content":"CONTEXT:\n"+json.dumps(context)}]
    for m in messages:
        if m.get("role") in {"user","assistant"}:
            full.append({"role":m["role"],"content":m.get("content","")})
    resp = client.chat.completions.create(model=model, messages=full, temperature=0.2)
    return resp.choices[0].message.content.strip()

# =========================================================
# APP START
# =========================================================
require_file(PRODUCTS_CSV,   "products.csv (data/products.csv)")
require_file(EXCLUSIONS_CSV, "exclusions.csv (data/exclusions.csv)")
require_file(LOGO_PATH,      "logo (assets/ibex_logo.png)")

products   = load_products()
exclusions = load_exclusions()

STRIPE_BASIC_LINK = st.secrets.get("STRIPE_BASIC_LINK","")
STRIPE_PERF_LINK  = st.secrets.get("STRIPE_PERF_LINK","")

if "ai_out"          not in st.session_state: st.session_state.ai_out          = None
if "last_plan"       not in st.session_state: st.session_state.last_plan       = "Basic"
if "last_rid"        not in st.session_state: st.session_state.last_rid        = None
if "last_intake"     not in st.session_state: st.session_state.last_intake     = None
if "chat_messages"   not in st.session_state: st.session_state.chat_messages   = []

render_header()

tabs = st.tabs(["Audit", "Ask IBEX", "Privacy", "FAQ"])

# =========================================================
# TAB 0 — AUDIT
# =========================================================
with tabs[0]:

    if st.session_state.ai_out:
        ai_out = st.session_state.ai_out
        plan   = st.session_state.last_plan

        st.markdown(f"""
        <div class="ibx-card" style="display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:12px;">
          <div>
            <div class="ibx-label">Your Results</div>
            <div class="ibx-title">Your {plan} System</div>
            <div style="font-size:0.78rem;color:rgba(240,237,230,0.35);margin-top:4px;">Ref: {st.session_state.last_rid}</div>
          </div>
          <div>
            <span class="ibx-badge green">NCAA Compliant</span>
            <span class="ibx-badge">AI-Personalized</span>
            <span class="ibx-badge">Evidence-Linked</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        display_audit_id(st.session_state.last_rid)

        if ai_out.get("consult_professional", False):
            st.warning("Based on what you shared, we recommend consulting a qualified professional. We kept this stack conservative.")
        flags = ai_out.get("flags",[])
        if flags:
            st.caption("Signals detected: " + ", ".join(flags))

        st.markdown('<div class="ibx-label" style="margin-top:1rem;">Recommended Stack</div>', unsafe_allow_html=True)
        render_products(ai_out.get("included_product_ids",[]), products, ai_out.get("reasons",{}))

        st.markdown('<div class="ibx-label" style="margin-top:1rem;">Daily Schedule</div>', unsafe_allow_html=True)
        render_schedule(ai_out.get("schedule",{}), products)

        notes = ai_out.get("notes_for_athlete",[])
        if notes:
            st.markdown('<div class="ibx-label" style="margin-top:1rem;">Notes</div>', unsafe_allow_html=True)
            st.markdown('<div class="ibx-card">', unsafe_allow_html=True)
            for n in notes:
                st.write(f"→ {n}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Checkout
        st.markdown("""
        <div style="margin-top:2rem;" class="ibx-label">Checkout</div>
        <div class="ibx-card">
          <div style="font-size:0.82rem;color:rgba(240,237,230,0.5);margin-bottom:1rem;">
            Copy your IBEX Audit ID above and paste it into Stripe during checkout so we can match your order.
          </div>
        """, unsafe_allow_html=True)

        if plan == "Basic":
            st.markdown("""
            <div style="font-family:'Bebas Neue',sans-serif;font-size:1.8rem;color:#f0ede6;">IBEX Basic — $100/mo</div>
            <div style="font-size:0.82rem;color:rgba(240,237,230,0.45);margin-bottom:1rem;">Foundations, done right. Free shipping.</div>
            """, unsafe_allow_html=True)
            if STRIPE_BASIC_LINK:
                st.link_button("Subscribe — IBEX Basic →", STRIPE_BASIC_LINK)
            else:
                st.info("Set STRIPE_BASIC_LINK in Streamlit Secrets.")
        else:
            st.markdown("""
            <div style="font-family:'Bebas Neue',sans-serif;font-size:1.8rem;color:#f0ede6;">IBEX Performance — $130/mo</div>
            <div style="font-size:0.82rem;color:rgba(240,237,230,0.45);margin-bottom:1rem;">Optimization mode. Free priority shipping.</div>
            """, unsafe_allow_html=True)
            if STRIPE_PERF_LINK:
                st.link_button("Subscribe — IBEX Performance →", STRIPE_PERF_LINK)
            else:
                st.info("Set STRIPE_PERF_LINK in Streamlit Secrets.")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        if st.button("← Start a new audit"):
            st.session_state.ai_out        = None
            st.session_state.last_rid      = None
            st.session_state.last_intake   = None
            st.session_state.chat_messages = []
            st.rerun()

    else:
        st.markdown("""
        <div class="ibx-card">
          <div class="ibx-label">Get Started</div>
          <div class="ibx-title">Performance Audit</div>
          <div style="font-size:0.88rem;color:rgba(240,237,230,0.5);margin-top:8px;line-height:1.7;">
            Fill out the audit in the sidebar. Your personalized stack appears here instantly.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── SIDEBAR FORM ──
    with st.sidebar:
        st.markdown("""
        <div style="font-family:'Bebas Neue',sans-serif;font-size:1.8rem;color:#c9a84c;letter-spacing:0.08em;margin-bottom:0.2rem;">IBEX Audit</div>
        <div style="font-size:0.72rem;color:rgba(240,237,230,0.4);font-family:'Barlow Condensed',sans-serif;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:1rem;">Plan → Audit → Instant system</div>
        """, unsafe_allow_html=True)

        plan = st.radio("Choose your plan", ["Basic","Performance"],
                        index=0 if st.session_state.last_plan == "Basic" else 1,
                        horizontal=True)

        pc = PLAN_COPY[plan]
        st.markdown(f"**{pc['headline']}**")
        st.write(pc["sub"])
        for b in pc["bullets"]: st.write(f"→ {b}")
        st.caption(pc["note"])
        st.markdown("---")

        with st.form("audit_form"):
            st.markdown("**About you**")
            name   = st.text_input("Full name")
            email  = st.text_input("Email")
            school = st.text_input("School")

            st.markdown("**Sport & training**")
            sport        = st.text_input("Sport")
            position     = st.text_input("Position / Event")
            season_status= st.selectbox("Season status",["In-season","Pre-season","Off-season"])
            training_days= st.slider("Training days/week", 0, 7, 5)
            intensity    = st.slider("Training intensity (1–10)", 1, 10, 7)
            travel       = st.selectbox("Travel frequency",["Never","Sometimes","Often"])

            st.markdown("**Goals**")
            goals = st.multiselect("Select all that apply",
                ["strength","endurance","recovery","sleep","gut","joints","focus","general health"])

            st.markdown("**Recovery & lifestyle**")
            sleep_hours   = st.number_input("Sleep hours/night", min_value=0.0, max_value=12.0, value=7.0, step=0.5)
            sleep_quality = st.slider("Sleep quality (1–10)", 1, 10, 6)
            stress        = st.slider("Stress (1–10)", 1, 10, 6)
            soreness      = st.slider("Soreness/Fatigue (1–10)", 1, 10, 6)
            gi_sensitive  = st.checkbox("GI sensitive / stomach issues", value=False)
            caffeine_sens = st.checkbox("Caffeine sensitive", value=False)

            st.markdown("**Current stack / notes**")
            current_supps     = st.text_area("Supplements you already take (optional)", placeholder="Creatine, fish oil, whey…")
            avoid_ingredients = st.text_input("Ingredients to avoid (optional)", placeholder="e.g., caffeine")
            open_notes        = st.text_area("Other context or concerns (optional)", placeholder="Anything that would help tailor the plan…")

            st.markdown("---")
            st.caption("Not medical advice. For details, see the Privacy tab.")
            submitted = st.form_submit_button("Build My System")

        if submitted:
            rid = str(uuid.uuid4())
            intake = {
                "rid":rid,"plan":plan,"name":name,"email":email,"school":school,
                "sport":sport,"position":position,"season_status":season_status,
                "training_days_per_week":training_days,"intensity_1_to_10":intensity,
                "travel_frequency":travel,"goals":goals,"sleep_hours":sleep_hours,
                "sleep_quality_1_to_10":sleep_quality,"stress_1_to_10":stress,
                "soreness_1_to_10":soreness,"gi_sensitive":gi_sensitive,
                "caffeine_sensitive":caffeine_sens,"current_supplements":current_supps,
                "avoid_ingredients":avoid_ingredients,"open_notes":open_notes,
            }
            shortlist = shortlist_products(products, goals, gi_sensitive, caffeine_sens, plan)
            with st.spinner("Building your system…"):
                ai_out = run_ai(intake, shortlist, exclusions, plan)
            ai_out = enforce_caps(ai_out, plan, products)
            try:
                _ = save_to_supabase(rid, intake, ai_out)
                st.sidebar.success("Saved ✓")
            except Exception as e:
                st.sidebar.error("Save failed")
                st.sidebar.code(str(e))
            st.session_state.ai_out        = ai_out
            st.session_state.last_plan     = plan
            st.session_state.last_rid      = rid
            st.session_state.last_intake   = intake
            st.session_state.chat_messages = []
            st.rerun()

# =========================================================
# TAB 1 — ASK IBEX
# =========================================================
with tabs[1]:
    st.markdown("""
    <div class="ibx-card">
      <div class="ibx-label">AI Chat</div>
      <div class="ibx-title">Ask IBEX</div>
      <div style="font-size:0.85rem;color:rgba(240,237,230,0.5);margin-top:6px;line-height:1.7;">
        Ask questions about your stack, timing, and tradeoffs. Evidence links shown only when attached per item.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.ai_out or not st.session_state.last_intake:
        st.info("Run your audit first — your personalized chat will appear here.")
    else:
        for m in st.session_state.chat_messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        prompt = st.chat_input('Ask a question (e.g., "Why creatine?" "Can I take these together?" "What if I travel?")')
        if prompt:
            st.session_state.chat_messages.append({"role":"user","content":prompt})
            with st.chat_message("user"): st.markdown(prompt)
            context = build_chat_context(st.session_state.last_intake, st.session_state.ai_out, products)
            with st.spinner("IBEX is thinking…"):
                answer = run_chat_answer(st.session_state.chat_messages, context)
            st.session_state.chat_messages.append({"role":"assistant","content":answer})
            with st.chat_message("assistant"): st.markdown(answer)

# =========================================================
# TAB 2 — PRIVACY
# =========================================================
with tabs[2]:
    render_privacy_policy()

# =========================================================
# TAB 3 — FAQ
# =========================================================
with tabs[3]:
    render_faq()
