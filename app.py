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

LOGO_PATH = "assets/ibex_logo.png"

st.set_page_config(
    page_title=f"{APP_TITLE} • Performance Audit",
    page_icon=LOGO_PATH,
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PREMIUM STYLING
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

/* FIX: Dropdown menu options unreadable */
div[data-baseweb="popover"]{
  background: transparent !important;
}
div[data-baseweb="menu"]{
  background:#ffffff !important;
  border:1px solid rgba(15,23,42,0.12) !important;
  border-radius: 14px !important;
  overflow:hidden !important;
}
div[data-baseweb="menu"] *{
  color:#0f172a !important;
}
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

/* =========================
   FAQ Accordion
========================= */
.ibx-faq details{
  background:#ffffff;
  border:1px solid rgba(15,23,42,0.10);
  border-radius:16px;
  padding: 14px 16px;
  margin: 10px 0;
  box-shadow: 0 10px 24px rgba(2,6,23,0.06);
}
.ibx-faq details[open]{
  border-color: rgba(239,68,68,0.35);
  box-shadow: 0 14px 30px rgba(239,68,68,0.10);
}
.ibx-faq summary{
  list-style:none;
  cursor:pointer;
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:12px;
  font-weight:900;
  color:#0f172a;
  font-size:16px;
  outline:none;
}
.ibx-faq summary::-webkit-details-marker{ display:none; }
.ibx-faq .qhint{
  color:#64748b;
  font-weight:700;
  font-size:12px;
  margin-top: 2px;
}
.ibx-faq .answer{
  margin-top: 12px;
  color:#334155;
  line-height:1.65;
  font-size:15px;
}
.ibx-faq .answer ul{ margin: 8px 0 0 18px; }
.ibx-faq .chev{
  width:34px;
  height:34px;
  border-radius:12px;
  display:flex;
  align-items:center;
  justify-content:center;
  background: rgba(15,23,42,0.04);
  border: 1px solid rgba(15,23,42,0.08);
  flex: 0 0 auto;
}
.ibx-faq details[open] .chev{
  background: rgba(239,68,68,0.08);
  border-color: rgba(239,68,68,0.18);
}
.ibx-faq .pill{
  display:inline-block;
  padding: 6px 10px;
  border-radius:999px;
  background: rgba(15,23,42,0.04);
  border:1px solid rgba(15,23,42,0.08);
  color:#334155;
  font-size:12px;
  font-weight:800;
  margin-right:8px;
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
    return str(val).strip().lower() in {"y", "yes", "true", "1"}


def parse_money(val) -> float:
    try:
        s = str(val).strip()
        if not s:
            return 0.0
        s = s.replace("$", "").replace(",", "")
        x = float(s)
        if x != x:  # NaN guard
            return 0.0
        return x
    except Exception:
        return 0.0


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
# Evidence link (paper-per-product)
# =========================================================
def get_evidence_link(row: dict) -> str:
    ev = str(row.get("Evidence_Link", "") or "").strip()
    if ev:
        return ev
    return str(row.get("Link", "") or "").strip()


def evidence_enabled() -> bool:
    return str(st.secrets.get("EVIDENCE_LINKS_ENABLED", "true")).strip().lower() in {"1", "true", "yes", "y"}


# =========================================================
# PLAN DEFINITIONS + INTERNAL PROFIT-PROTECTED LIMITS
# (NOT shown in UI)
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

# Internal only. You can change these later without touching UI.
PLAN_LIMITS = {
    "Basic": {"max_units": 5, "supp_budget": 39.0, "max_am": 3, "max_pm": 2, "max_training": 2},
    "Performance": {"max_units": 8, "supp_budget": 69.0, "max_am": 3, "max_pm": 3, "max_training": 2},
}


def item_units(monthly_cost: float) -> int:
    return 2 if monthly_cost >= 20.0 else 1


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
                  <span class="ibx-badge">Evidence-linked</span>
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
        "Product_ID", "Category", "Ingredient", "Brand", "Store", "Link",
        "Serving_Form", "Typical_Use", "Timing", "Avoid_If",
        "Third_Party_Tested", "NSF_Certified", "Price", "Est_Monthly_Cost", "Notes"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"products.csv missing columns: {missing}")

    if "Evidence_Link" not in df.columns:
        df["Evidence_Link"] = ""
    if "NCAA_Risk_Tier" not in df.columns:
        df["NCAA_Risk_Tier"] = ""
    if "Athlete_Safe_OK" not in df.columns:
        df["Athlete_Safe_OK"] = ""

    return df


@st.cache_data(show_spinner=False)
def load_exclusions():
    df = pd.read_csv(EXCLUSIONS_CSV)
    df.columns = [c.strip() for c in df.columns]
    if "Excluded_Category_or_Ingredient" not in df.columns or "Reason" not in df.columns:
        raise ValueError("exclusions.csv must have columns: Excluded_Category_or_Ingredient, Reason")
    return df


# =========================================================
# FILTERS
# =========================================================
def filter_products_by_plan(products: pd.DataFrame, plan: str) -> pd.DataFrame:
    p = products.copy()
    p["Category_norm"] = p["Category"].astype(str).str.strip()
    if plan == "Basic":
        return p[p["Category_norm"].isin(BASIC_CORE_CATEGORIES)]
    return p


def filter_ncaa_safe(products: pd.DataFrame, plan: str) -> pd.DataFrame:
    p = products.copy()

    if "Athlete_Safe_OK" in p.columns:
        p = p[p["Athlete_Safe_OK"].astype(str).str.strip().str.upper().isin({"Y", "YES", "TRUE", "1"})]

    if "NCAA_Risk_Tier" in p.columns:
        tier = p["NCAA_Risk_Tier"].astype(str).str.strip().str.lower()
        if plan == "Basic":
            p = p[tier.eq("green") | tier.eq("")]
        else:
            p = p[tier.isin({"green", "yellow"}) | tier.eq("")]

    return p


def shortlist_products(products: pd.DataFrame, goals: list[str], gi_sensitive: bool, caffeine_sensitive: bool, plan: str) -> pd.DataFrame:
    products = filter_ncaa_safe(products, plan)
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


# =========================================================
# AI
# =========================================================
def run_ai(intake: dict, products_shortlist: pd.DataFrame, exclusions: pd.DataFrame, plan: str) -> dict:
    client = get_openai_client()

    approved_products = products_shortlist[[
        "Product_ID", "Category", "Ingredient", "Brand", "Store", "Link",
        "Evidence_Link",
        "Serving_Form", "Typical_Use", "Timing", "Avoid_If",
        "Third_Party_Tested", "NSF_Certified", "Notes",
        "Est_Monthly_Cost"
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

    lim = PLAN_LIMITS.get(plan, {"max_units": 6, "supp_budget": 50.0, "max_am": 3, "max_pm": 3, "max_training": 2})

    system_prompt = (
        "You are IBEX, an assistant that organizes a personalized supplement system for athletes. "
        "You are NOT a medical provider. Do NOT diagnose, treat, or make medical claims. "
        "Only select products from approved_products. "
        "Never select anything that matches the exclusions list. "
        "IMPORTANT: Evidence links are provided per product in approved_products as Evidence_Link (or Link fallback). "
        "Do NOT invent papers, DOIs, authors, or citations. If evidence is missing, do not pretend it exists. "
        "If intake mentions serious symptoms, medications, or a medical condition, set consult_professional=true and keep recommendations conservative. "
        f"{plan_rules} "
        "STACK CAPS (must follow): "
        f"Cap the stack to {lim['max_units']} units max, where any item with Est_Monthly_Cost >= $20 counts as 2 units. "
        f"Schedule caps: AM ≤ {lim['max_am']}, PM ≤ {lim['max_pm']}, Training ≤ {lim['max_training']}. "
        "If more items could help, include them as optional suggestions in notes_for_athlete instead of adding them to included_product_ids. "
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
            return json.loads(content[start:end + 1])
        raise


# =========================================================
# ENFORCE STACK CAPS (HARD) — last line of defense
# =========================================================
def enforce_caps(ai_out: dict, plan: str, products_df: pd.DataFrame) -> dict:
    lim = PLAN_LIMITS.get(plan, {"max_units": 6, "supp_budget": 50.0, "max_am": 3, "max_pm": 3, "max_training": 2})

    included = ai_out.get("included_product_ids", []) or []
    schedule = ai_out.get("schedule", {}) or {}
    reasons = ai_out.get("reasons", {}) or {}
    notes = ai_out.get("notes_for_athlete", []) or []

    if not included:
        return ai_out

    prod_map = products_df.set_index("Product_ID").to_dict(orient="index")
    model_order = {pid: i for i, pid in enumerate(included)}

    rows = []
    for pid in included:
        p = prod_map.get(pid, {})
        cat = str(p.get("Category", "") or "").strip()
        est = parse_money(p.get("Est_Monthly_Cost", 0))
        nsf = is_yes(p.get("NSF_Certified", ""))
        tpt = str(p.get("Third_Party_Tested", "")).strip().lower() in {"y", "yes", "true", "1", "unknown"}
        core = cat in BASIC_CORE_CATEGORIES
        rows.append({
            "pid": pid,
            "cat": cat,
            "est": est,
            "nsf": nsf,
            "tpt": tpt,
            "core": core,
            "order": model_order.get(pid, 9999)
        })

    rows.sort(key=lambda r: (not r["core"], not r["nsf"], not r["tpt"], r["order"]))

    picked = []
    used_units = 0

    for r in rows:
        cost = float(r["est"] or 0.0)
        units = item_units(cost)
        if used_units + units > lim["max_units"]:
            continue
        picked.append(r["pid"])
        used_units += units

    if not picked:
        for r in rows:
            cost = float(r["est"] or 0.0)
            units = item_units(cost)
            if used_units + units > lim["max_units"]:
                continue
            picked.append(r["pid"])
            used_units += units
            if used_units >= lim["max_units"]:
                break

    picked_set = set(picked)

    def trim_bucket(items, maxn):
        out = []
        for pid in (items or []):
            if pid in picked_set and pid not in out:
                out.append(pid)
            if len(out) >= maxn:
                break
        return out

    new_schedule = {
        "AM": trim_bucket(schedule.get("AM", []), lim["max_am"]),
        "PM": trim_bucket(schedule.get("PM", []), lim["max_pm"]),
        "Training": trim_bucket(schedule.get("Training", []), lim["max_training"]),
    }

    new_reasons = {pid: reasons.get(pid, "") for pid in picked if pid in reasons}

    # IMPORTANT: no dollar amounts shown to users
    if len(picked) < len(included):
        msg = "To keep the stack practical, we capped the number of recommended items for your plan."
        if msg not in notes:
            notes = [msg] + notes

    ai_out["included_product_ids"] = picked
    ai_out["schedule"] = new_schedule
    ai_out["reasons"] = new_reasons
    ai_out["notes_for_athlete"] = notes

    # keep internal meta for you (not displayed anywhere)
    ai_out["meta_caps"] = {"plan": plan, "units_selected": used_units, "max_units": lim["max_units"]}

    return ai_out


# =========================================================
# UI RENDERING (NO BRAND / NO STORE SHOWN)
# =========================================================
def render_products(product_ids: list[str], products_df: pd.DataFrame, reasons: dict):
    prod_map = products_df.set_index("Product_ID").to_dict(orient="index")
    cols = st.columns(3, gap="large")

    for i, pid in enumerate(product_ids):
        p = prod_map.get(pid)
        if not p:
            continue

        ev = get_evidence_link(p)

        with cols[i % 3]:
            st.markdown("<div class='ibx-card'>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap;">
                  <span class="ibx-badge">{p.get('Category','')}</span>
                  <span class="ibx-badge">{p.get('Timing','')}</span>
                </div>

                <div style="margin-top:12px; font-size:18px; font-weight:800; color:#0f172a;">
                  {p.get('Ingredient','')}
                </div>

                <div class="ibx-muted" style="margin-top:2px;">
                  {p.get('Serving_Form','')}
                </div>

                <div class="ibx-divider"></div>

                <div style="font-weight:800; color:#0f172a;">Why this</div>
                <div class="ibx-muted" style="margin-top:4px;">
                  {reasons.get(pid, "Personalized to your audit")}
                </div>
                """,
                unsafe_allow_html=True
            )

            if evidence_enabled():
                st.markdown("<div class='ibx-divider'></div>", unsafe_allow_html=True)
                st.markdown("<div style='font-weight:800; color:#0f172a;'>Evidence</div>", unsafe_allow_html=True)

                if ev:
                    st.link_button("Open the linked study", ev)
                else:
                    st.caption("No evidence link attached for this item yet.")

            st.markdown("</div>", unsafe_allow_html=True)


def render_schedule(schedule: dict, products_df: pd.DataFrame):
    prod_map = products_df.set_index("Product_ID").to_dict(orient="index")
    blocks = [("AM", "Morning"), ("PM", "Evening"), ("Training", "Training")]
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
                    st.markdown(f"- **{p.get('Ingredient', pid)}**")
            st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PRIVACY POLICY (FIXED — no raw HTML tags showing)
# =========================================================
def render_privacy_policy():
    eff = date.today().strftime("%B %d, %Y")
    support_email = st.secrets.get("SUPPORT_EMAIL", "support@ibexsupplements.com")

    st.markdown(
        f"""
<div class="ibx-card">
  <div style="font-size:30px; font-weight:950; color:#0f172a;">Privacy Policy</div>
  <div class="ibx-muted" style="margin-top:6px;">Effective: {eff}</div>
  <div class="ibx-divider"></div>

  <div style="font-size:16px; line-height:1.7; color:#334155;">
    <p><b>IBEX</b> (“we,” “us,” or “our”) provides a performance audit and supplement planning experience for athletes.
    This policy explains what we collect, why we collect it, how it’s used, and the choices you have.</p>

    <div style="margin-top:18px; font-weight:950; color:#0f172a; font-size:18px;">1) What we collect</div>
    <ul>
      <li><b>Contact info</b>: name and email (optional fields may be left blank).</li>
      <li><b>Training & lifestyle inputs</b>: sport, season status, training frequency, goals, sleep, stress, soreness, sensitivities, and notes you enter.</li>
      <li><b>App usage data</b>: audit reference ID and basic logs needed to operate the service.</li>
    </ul>

    <div style="margin-top:18px; font-weight:950; color:#0f172a; font-size:18px;">2) What we do not collect</div>
    <ul>
      <li>We do not require student/team IDs, department logins, or social security numbers.</li>
      <li>We do not sell personal information.</li>
    </ul>

    <div style="margin-top:18px; font-weight:950; color:#0f172a; font-size:18px;">3) How we use your information</div>
    <ul>
      <li>Generate your recommended system and timing schedule.</li>
      <li>Power the “Ask IBEX” chat using your audit + recommended items as context.</li>
      <li>Show evidence links only when attached per product in our catalog.</li>
      <li>Improve reliability and safety controls (aggregate analysis, not ad targeting).</li>
    </ul>

    <div style="margin-top:18px; font-weight:950; color:#0f172a; font-size:18px;">4) AI processing</div>
    <p>Your audit inputs and a curated list of allowed catalog items may be sent to an AI provider to generate structured output.
    We instruct the model to avoid medical diagnosis/treatment and to <b>not invent research citations</b>.
    If evidence is missing for an item, we do not pretend it exists.</p>

    <div style="margin-top:18px; font-weight:950; color:#0f172a; font-size:18px;">5) NCAA / sport compliance notice</div>
    <p>IBEX is athlete-safe oriented, but <b>no supplement can be guaranteed compliant</b> for any league or test.
    Rules change and contamination can occur. Always confirm with your athletic department.</p>

    <div style="margin-top:18px; font-weight:950; color:#0f172a; font-size:18px;">6) Where data is stored</div>
    <p>Audit records may be stored in a secure database and may include your inputs, outputs, and reference ID.</p>

    <div style="margin-top:18px; font-weight:950; color:#0f172a; font-size:18px;">7) Sharing</div>
    <ul>
      <li><b>Service providers</b>: hosting, database, and AI processing providers as needed to run IBEX.</li>
      <li><b>Legal</b>: if required by law or to protect users and the service.</li>
    </ul>

    <div style="margin-top:18px; font-weight:950; color:#0f172a; font-size:18px;">8) Retention</div>
    <p>We retain audit data as long as needed to provide the service and improve safety.
    You can request deletion (see Section 10).</p>

    <div style="margin-top:18px; font-weight:950; color:#0f172a; font-size:18px;">9) Security</div>
    <p>We use reasonable safeguards (access controls and encryption in transit where supported). No method is 100% secure.</p>

    <div style="margin-top:18px; font-weight:950; color:#0f172a; font-size:18px;">10) Your choices (access / delete)</div>
    <p>To request access to or deletion of your audit data, email <b>{support_email}</b> with your IBEX Audit ID.
    If you don’t have it, include the email you used (if provided) and approximate date/time.</p>

    <div style="margin-top:18px; font-weight:950; color:#0f172a; font-size:18px;">11) Children</div>
    <p>IBEX is intended for users who can legally consent to data processing in their jurisdiction.</p>

    <div style="margin-top:18px; font-weight:950; color:#0f172a; font-size:18px;">12) Changes</div>
    <p>We may update this policy. The effective date above reflects the latest revision.</p>

    <div style="margin-top:18px; font-weight:950; color:#0f172a; font-size:18px;">Contact</div>
    <p>Email: <b>{support_email}</b></p>
  </div>
</div>
""",
        unsafe_allow_html=True
    )


# =========================================================
# FAQ (PRETTY ACCORDION)
# =========================================================
def render_faq():
    support_email = st.secrets.get("SUPPORT_EMAIL", "support@ibexsupplements.com")

    st.markdown(
        f"""
<div class="ibx-card">
  <div style="font-size:30px; font-weight:950; color:#0f172a;">FAQ</div>
  <div class="ibx-muted" style="margin-top:6px;">Answers for athletes — simple, direct, and transparent.</div>
  <div class="ibx-divider"></div>

  <div class="ibx-faq">

    <details open>
      <summary>
        <div>
          What is IBEX?
          <div class="qhint">Audit → system → schedule</div>
        </div>
        <div class="chev">⌄</div>
      </summary>
      <div class="answer">
        IBEX is a performance audit that helps athletes build a simple supplement system and timing schedule.
        You answer questions about training, recovery, and goals, and IBEX generates a plan using only items from a curated catalog.
      </div>
    </details>

    <details>
      <summary>
        <div>
          Is IBEX medical advice?
          <div class="qhint">No diagnosis, no treatment</div>
        </div>
        <div class="chev">⌄</div>
      </summary>
      <div class="answer">
        No. IBEX is not a medical provider and does not diagnose or treat conditions.
        If you have symptoms, take medications, or have a medical condition, consult a qualified professional.
      </div>
    </details>

    <details>
      <summary>
        <div>
          How does IBEX choose supplements?
          <div class="qhint">AI constrained to your allowed list</div>
        </div>
        <div class="chev">⌄</div>
      </summary>
      <div class="answer">
        IBEX uses your audit inputs and matches them to items in the catalog.
        The AI is constrained to choose only from the approved products provided for your session.
      </div>
    </details>

    <details>
      <summary>
        <div>
          Why do you show “Evidence” links?
          <div class="qhint">Trust barrier → sources only when they exist</div>
        </div>
        <div class="chev">⌄</div>
      </summary>
      <div class="answer">
        Athletes are smart — and trust matters. IBEX only shows evidence links that are attached per product in the catalog.
        If an item has no evidence link, IBEX will say so. No invented citations.
      </div>
    </details>

    <details>
      <summary>
        <div>
          Can I ask questions about my plan?
          <div class="qhint">Yes — use Ask IBEX</div>
        </div>
        <div class="chev">⌄</div>
      </summary>
      <div class="answer">
        Yes. Use the <b>Ask IBEX</b> tab to ask questions about timing, stacking, tradeoffs, and travel routines.
        The chat is grounded in your audit + your recommended items.
      </div>
    </details>

    <details>
      <summary>
        <div>
          Do you show brands or stores?
          <div class="qhint">Ingredient-first recommendations</div>
        </div>
        <div class="chev">⌄</div>
      </summary>
      <div class="answer">
        By default, IBEX does <b>not</b> show brands or retailers in athlete-facing recommendation cards.
        The focus is the ingredient, form, timing, and why it was chosen.
      </div>
    </details>

    <details>
      <summary>
        <div>
          Is this safe for NCAA athletes?
          <div class="qhint">No guarantees — but strong guardrails</div>
        </div>
        <div class="chev">⌄</div>
      </summary>
      <div class="answer">
        IBEX is designed to be athlete-safe oriented, but no supplement can be guaranteed compliant for every league or test.
        Rules change and contamination risk exists. Always check with your athletic department.
        <div style="margin-top:10px;">
          <span class="pill">Prefer third-party tested</span>
          <span class="pill">Avoid blends & “hardcore” products</span>
          <span class="pill">Keep packaging / lot numbers</span>
        </div>
      </div>
    </details>

    <details>
      <summary>
        <div>
          Why are recommendations capped?
          <div class="qhint">Practical + simple system</div>
        </div>
        <div class="chev">⌄</div>
      </summary>
      <div class="answer">
        IBEX caps recommendations to keep your stack realistic and easy to follow.
        If more items could help, IBEX lists them as optional in notes instead of forcing a huge stack.
      </div>
    </details>

    <details>
      <summary>
        <div>
          How do I delete my data?
          <div class="qhint">Email support with your Audit ID</div>
        </div>
        <div class="chev">⌄</div>
      </summary>
      <div class="answer">
        Email <b>{support_email}</b> with your IBEX Audit ID and request deletion.
        If you don’t have the ID, include the email you used (if provided) and approximate date/time.
      </div>
    </details>

    <details>
      <summary>
        <div>
          Support
          <div class="qhint">We respond fast</div>
        </div>
        <div class="chev">⌄</div>
      </summary>
      <div class="answer">
        Email: <b>{support_email}</b>
      </div>
    </details>

  </div>
</div>
""",
        unsafe_allow_html=True
    )


# =========================================================
# AI CHAT
# =========================================================
def build_chat_context(intake: dict, ai_out: dict, products_df: pd.DataFrame) -> dict:
    prod_map = products_df.set_index("Product_ID").to_dict(orient="index")
    included = ai_out.get("included_product_ids", []) or []
    schedule = ai_out.get("schedule", {}) or {}
    reasons = ai_out.get("reasons", {}) or {}

    items = []
    for pid in included:
        p = prod_map.get(pid, {})
        items.append({
            "Product_ID": pid,
            "Category": p.get("Category", ""),
            "Ingredient": p.get("Ingredient", ""),
            "Timing": p.get("Timing", ""),
            "Serving_Form": p.get("Serving_Form", ""),
            "Reason": reasons.get(pid, ""),
            "Evidence_Link": get_evidence_link(p),
            "Notes": p.get("Notes", "")
        })

    return {
        "intake": intake,
        "recommendations": items,
        "schedule": schedule,
        "notes_for_athlete": ai_out.get("notes_for_athlete", []),
        "flags": ai_out.get("flags", []),
        "consult_professional": bool(ai_out.get("consult_professional", False))
    }


def run_chat_answer(messages: list[dict], context: dict) -> str:
    client = get_openai_client()
    model = st.secrets.get("OPENAI_CHAT_MODEL", st.secrets.get("OPENAI_MODEL", "gpt-4.1-mini"))

    system = (
        "You are IBEX Chat, an athlete-safe assistant. "
        "You are NOT a medical provider. Do not diagnose or provide medical treatment advice. "
        "Use only the provided context (audit + recommended items + evidence links). "
        "DO NOT invent studies, authors, DOIs, or citations. "
        "If a recommended item has Evidence_Link, you may reference it directly. "
        "If Evidence_Link is missing, say you don't have a linked study for that item yet. "
        "Keep answers practical, short, and athlete-friendly. "
        "If user asks medical questions, advise consulting a qualified professional."
    )

    full = [{"role": "system", "content": system}]
    full.append({"role": "user", "content": "CONTEXT:\n" + json.dumps(context)})

    for m in messages:
        if m.get("role") in {"user", "assistant"}:
            full.append({"role": m["role"], "content": m.get("content", "")})

    resp = client.chat.completions.create(
        model=model,
        messages=full,
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()


# =========================================================
# APP START
# =========================================================
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
if "last_intake" not in st.session_state:
    st.session_state.last_intake = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

render_header()

tabs = st.tabs(["Audit", "Ask IBEX", "Privacy", "FAQ"])

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
                  <span class="ibx-badge">Evidence-linked</span>
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
            st.session_state.last_intake = None
            st.session_state.chat_messages = []
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
            open_notes = st.text_area("Other context or concerns (optional)", placeholder="Anything that would help tailor the plan…")

            st.markdown("---")
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

            ai_out = enforce_caps(ai_out, plan, products)

            try:
                _ = save_to_supabase(rid, intake, ai_out)
                st.sidebar.success("Saved ✅")
            except Exception as e:
                st.sidebar.error("Save failed (DB)")
                st.sidebar.code(str(e))

            st.session_state.ai_out = ai_out
            st.session_state.last_plan = plan
            st.session_state.last_rid = rid
            st.session_state.last_intake = intake
            st.session_state.chat_messages = []
            st.rerun()

# =========================================================
# TAB: ASK IBEX
# =========================================================
with tabs[1]:
    st.markdown(
        """
        <div class="ibx-card">
          <div style="font-size:28px; font-weight:950; color:#0f172a;">Ask IBEX</div>
          <div class="ibx-muted" style="margin-top:6px;">
            Ask questions about your stack, timing, and tradeoffs. Evidence links are shown only when attached per item.
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state.ai_out or not st.session_state.last_intake:
        st.info("Run an audit first. Then your personalized chat will appear here.")
    else:
        for m in st.session_state.chat_messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        prompt = st.chat_input("Ask a question (e.g., “Why creatine?” “Can I take these together?” “What if I travel?”)")
        if prompt:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            context = build_chat_context(st.session_state.last_intake, st.session_state.ai_out, products)

            with st.spinner("IBEX is thinking…"):
                answer = run_chat_answer(st.session_state.chat_messages, context)

            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

# =========================================================
# TAB: PRIVACY
# =========================================================
with tabs[2]:
    render_privacy_policy()

# =========================================================
# TAB: FAQ
# =========================================================
with tabs[3]:
    render_faq()










