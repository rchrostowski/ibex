import os
import json
import uuid
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
  --text:#0b1220;
  --sub:#24324a;
  --muted:#64748b;
  --border:rgba(15,23,42,0.10);
  --accent:#ef4444;
  --accent2:#111827;

  --side:#0b1220;
  --sideBorder:#132033;
  --sideText:#e5e7eb;

  --successBg:#ecfdf5;
  --successBorder:#10b98133;
  --successText:#065f46;

  --warnBg:#fffbeb;
  --warnBorder:#f59e0b33;
  --warnText:#92400e;
}

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.stApp{ background: var(--bg); }
html, body, [class*="css"]{
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
}

h1,h2,h3,h4,h5{ color:var(--text) !important; letter-spacing:-0.2px; }
p,li,span,div,label{ color:var(--sub); }

/* Sidebar */
section[data-testid="stSidebar"]{
  background: var(--side);
  border-right:1px solid var(--sideBorder);
}
section[data-testid="stSidebar"] *{ color: var(--sideText) !important; }

/* Premium cards */
.ibx-card{
  background: var(--card);
  border:1px solid var(--border);
  border-radius: 22px;
  padding: 26px;
  box-shadow: 0 22px 55px rgba(2, 6, 23, 0.08);
  margin-bottom: 16px;
}
.ibx-muted{ color: var(--muted) !important; }
.ibx-divider{
  height:1px;
  background: rgba(15,23,42,0.08);
  margin: 14px 0;
}
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

/* Inputs (main) */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input{
  background:#ffffff !important;
  color: var(--text) !important;
  border:1px solid rgba(15,23,42,0.12) !important;
  border-radius: 14px !important;
}
.stTextInput input::placeholder,
.stTextArea textarea::placeholder,
.stNumberInput input::placeholder{
  color: var(--muted) !important;
  opacity: 1 !important;
}

/* BaseWeb select */
div[data-baseweb="select"] > div{
  background:#ffffff !important;
  border:1px solid rgba(15,23,42,0.12) !important;
  border-radius: 14px !important;
}
div[data-baseweb="select"] *{ color: var(--text) !important; }
div[data-baseweb="select"] svg{ color: var(--text) !important; }

/* Dropdown menu */
div[data-baseweb="popover"] *{ color: var(--text) !important; }
div[data-baseweb="menu"]{
  background:#ffffff !important;
  border:1px solid rgba(15,23,42,0.10) !important;
  border-radius: 14px !important;
  overflow:hidden !important;
}
div[data-baseweb="menu"] [role="option"]{ background:#ffffff !important; }
div[data-baseweb="menu"] [role="option"]:hover{ background: rgba(15,23,42,0.06) !important; }

/* Sidebar inputs white bg */
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stTextArea textarea,
section[data-testid="stSidebar"] .stNumberInput input{
  background:#ffffff !important;
  color: var(--text) !important;
  border:1px solid rgba(229,231,235,0.35) !important;
  border-radius: 14px !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div{
  background:#ffffff !important;
  border:1px solid rgba(229,231,235,0.35) !important;
  border-radius: 14px !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] *{ color: var(--text) !important; }

/* Buttons */
.stButton button{
  border-radius: 14px !important;
  padding: 0.78rem 1.05rem !important;
  font-weight: 900 !important;
  background: var(--accent) !important;
  border: none !important;
}
.stButton button, .stButton button *{ color:#ffffff !important; }
.stButton button:hover{ opacity: 0.92; }

/* Link buttons */
div[data-testid="stLinkButton"] a{
  display:inline-flex !important;
  align-items:center !important;
  justify-content:center !important;
  border-radius: 14px !important;
  padding: 0.78rem 1.05rem !important;
  font-weight: 900 !important;
  background: var(--accent2) !important;
  border: 1px solid rgba(17,24,39,0.15) !important;
  text-decoration: none !important;
}
div[data-testid="stLinkButton"] a, div[data-testid="stLinkButton"] a *{ color:#ffffff !important; }
div[data-testid="stLinkButton"] a:hover{ opacity:0.92; }

/* Status banners */
.ibx-banner-success{
  background: var(--successBg);
  border:1px solid var(--successBorder);
  color: var(--successText);
  padding: 14px 16px;
  border-radius: 16px;
  font-weight: 850;
}
.ibx-banner-warn{
  background: var(--warnBg);
  border:1px solid var(--warnBorder);
  color: var(--warnText);
  padding: 14px 16px;
  border-radius: 16px;
  font-weight: 750;
}

/* Copy box */
.ibx-copy{
  margin-top:12px;
  font-size:20px;
  font-weight:950;
  background:#0b1220;
  border:1px solid rgba(255,255,255,0.10);
  border-radius:16px;
  padding:14px;
  color:#ffffff;
  text-align:center;
  letter-spacing:0.6px;
  user-select: all;
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
        st.stop()


def load_logo():
    if not os.path.exists(LOGO_PATH) or Image is None:
        return None
    try:
        return Image.open(LOGO_PATH)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_products():
    df = pd.read_csv(PRODUCTS_CSV)
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def load_exclusions():
    df = pd.read_csv(EXCLUSIONS_CSV)
    df.columns = [c.strip() for c in df.columns]
    return df


def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY in Streamlit Secrets.")
        st.stop()
    if OpenAI is None:
        st.error("Missing `openai` package. Add `openai` to requirements.txt.")
        st.stop()
    return OpenAI(api_key=api_key)


def get_supabase():
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        return None, "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY"
    if create_client is None:
        return None, "Supabase package not installed (add `supabase` to requirements.txt)"
    try:
        return create_client(url, key), None
    except Exception as e:
        return None, f"Supabase init failed: {e}"


def get_stripe():
    sk = st.secrets.get("STRIPE_SECRET_KEY")
    if not sk:
        return None, "Missing STRIPE_SECRET_KEY in Streamlit Secrets"
    if stripe is None:
        return None, "Stripe package not installed (add `stripe` to requirements.txt)"
    try:
        stripe.api_key = sk
        return stripe, None
    except Exception as e:
        return None, f"Stripe init failed: {e}"


def qp_get(key: str):
    try:
        v = st.query_params.get(key)
    except Exception:
        v = None
    if isinstance(v, list):
        return v[0] if v else None
    return v


def render_header():
    logo = load_logo()
    c1, c2 = st.columns([1, 7], gap="large")
    with c1:
        if logo is not None:
            st.image(logo, width=140)
    with c2:
        st.markdown(
            f"<div style='font-size:44px; font-weight:950; color:#0b1220; margin-top:2px;'>{APP_TITLE}</div>",
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


def audit_id_box(rid: str):
    st.markdown(
        f"""
<div class="ibx-card" style="border:1px solid rgba(239,68,68,0.35); background: linear-gradient(180deg,#fff, #fff5f5);">
  <div style="font-size:18px; font-weight:950; color:#0b1220;">IBEX Audit ID (copy + paste during checkout)</div>
  <div class="ibx-muted" style="margin-top:6px;">
    This links your payment to your personalized recommendations.
  </div>
  <div class="ibx-copy">{rid}</div>
  <div style="margin-top:12px; display:flex; gap:10px; flex-wrap:wrap; justify-content:center;">
    <button onclick="navigator.clipboard.writeText('{rid}')" style="
      padding:10px 18px; border-radius:12px; background:#ef4444; color:white;
      font-weight:950; border:none; cursor:pointer;
    ">Copy Audit ID</button>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# =========================================================
# PLAN COPY
# =========================================================
PLAN_COPY = {
    "Basic": {
        "headline": "Foundations, done right.",
        "sub": "A conservative essentials-only system that prioritizes consistency, safety, and simplicity.",
        "bullets": [
            "Core stack that covers 80% of outcomes",
            "Prefers NSF Certified for Sport / third-party tested when available",
            "Built for routine, budget control, and low friction",
        ],
        "note": "Best for: most athletes who want a clean baseline that’s hard to mess up.",
    },
    "Performance": {
        "headline": "Optimization mode.",
        "sub": "A deeper system with conditional add-ons based on your training load, recovery, and constraints.",
        "bullets": [
            "Expanded options (sleep, gut, joints, recovery) only when justified",
            "More conditional logic + tighter timing recommendations",
            "Still avoids risky / sketchy categories",
        ],
        "note": "Best for: high training volume, in-season stress, or athletes chasing marginal gains.",
    },
}


def run_ai(intake: dict, products_shortlist: pd.DataFrame, exclusions: pd.DataFrame, plan: str) -> dict:
    client = get_openai_client()

    approved_products = products_shortlist.to_dict(orient="records")
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
        "Plan: BASIC. Keep stack simple and foundational. Prefer NSF/third-party tested. Avoid niche/experimental items."
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
            return json.loads(content[start:end + 1])
        raise


def render_privacy_policy():
    eff = date.today().strftime("%B %d, %Y")
    st.markdown(
        f"""
<div class="ibx-card">
  <div style="font-size:26px; font-weight:950; color:#0b1220;">Privacy Policy</div>
  <div class="ibx-muted" style="margin-top:4px;">Effective: {eff}</div>
  <div class="ibx-divider"></div>
  <p><strong>IBEX</strong> does not sell your personal data. Stripe processes payments and shipping details.</p>
  <p class="ibx-muted" style="font-size:12px;">Template for MVP use, not legal advice.</p>
</div>
""",
        unsafe_allow_html=True
    )


def render_faq():
    st.markdown(
        """
<div class="ibx-card">
  <div style="font-size:26px; font-weight:950; color:#0b1220;">FAQ</div>
  <div class="ibx-divider"></div>
  <h3>Is this medical advice?</h3>
  <p class="ibx-muted">No. Consult a qualified professional for medical concerns.</p>
</div>
""",
        unsafe_allow_html=True
    )


# =========================================================
# FILE CHECKS + DATA
# =========================================================
require_file(PRODUCTS_CSV, "products.csv (data/products.csv)")
require_file(EXCLUSIONS_CSV, "exclusions.csv (data/exclusions.csv)")
require_file(LOGO_PATH, "logo (assets/ibex_logo.png)")

products = load_products()
exclusions = load_exclusions()

STRIPE_BASIC_LINK = st.secrets.get("STRIPE_BASIC_LINK", "")
STRIPE_PERF_LINK = st.secrets.get("STRIPE_PERF_LINK", "")


# =========================================================
# THANK YOU / STRIPE RETURN HANDLER (PREMIUM)
# =========================================================
success = qp_get("success")
session_id = qp_get("session_id")

if success == "true" and session_id:
    render_header()

    st.markdown(
        """
        <div class="ibx-banner-success">✅ Payment confirmed — you’re all set.</div>
        """,
        unsafe_allow_html=True,
    )

    supa, supa_err = get_supabase()
    stp, stripe_err = get_stripe()

    # Fetch Stripe session
    session_obj = None
    line_items = None

    if stp is None:
        st.markdown(
            f"<div class='ibx-card'><div class='ibx-banner-warn'>Stripe not connected: {stripe_err}</div>"
            f"<div class='ibx-muted' style='margin-top:10px;'>Add STRIPE_SECRET_KEY to Secrets and `stripe` to requirements.txt to show customer + shipping + items here.</div></div>",
            unsafe_allow_html=True
        )
    else:
        try:
            session_obj = stripe.checkout.Session.retrieve(
                session_id,
                expand=["line_items", "customer_details", "shipping_details"]
            )
            line_items = session_obj.get("line_items", {})
        except Exception as e:
            st.markdown(
                f"<div class='ibx-card'><div class='ibx-banner-warn'>Could not load Stripe session.</div>"
                f"<div class='ibx-muted' style='margin-top:10px;'>Check that STRIPE_SECRET_KEY is the correct mode (test vs live).</div></div>",
                unsafe_allow_html=True
            )

    # Display premium confirmation content
    if session_obj:
        cust = session_obj.get("customer_details") or {}
        ship = session_obj.get("shipping_details") or {}

        # Attempt to find audit id
        audit_id = None
        try:
            custom_fields = session_obj.get("custom_fields") or []
            for cf in custom_fields:
                label = (cf.get("label") or {}).get("custom") or ""
                if "audit" in label.lower():
                    audit_id = (cf.get("text") or {}).get("value")
        except Exception:
            pass

        if not audit_id:
            md = session_obj.get("metadata") or {}
            audit_id = md.get("ibex_audit_id")

        # Guess plan
        plan_guess = None
        if line_items and line_items.get("data"):
            desc = (line_items["data"][0].get("description") or "").lower()
            if "performance" in desc:
                plan_guess = "Performance"
            elif "basic" in desc:
                plan_guess = "Basic"

        # Save order to Supabase
        if supa is None:
            st.markdown(
                f"<div class='ibx-card'><div class='ibx-banner-warn'>Supabase not connected: {supa_err}</div>"
                f"<div class='ibx-muted' style='margin-top:10px;'>Add `supabase` to requirements.txt and set SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY.</div></div>",
                unsafe_allow_html=True
            )
        else:
            try:
                supa.table("orders").upsert({
                    "stripe_session_id": session_id,
                    "stripe_payment_intent": session_obj.get("payment_intent"),
                    "stripe_customer_email": cust.get("email"),
                    "stripe_customer_name": cust.get("name"),
                    "stripe_shipping": ship,
                    "stripe_line_items": line_items,
                    "plan": plan_guess,
                    "audit_id": audit_id,
                    "raw_session": session_obj,
                }).execute()

                if audit_id:
                    try:
                        supa.table("audits").update({"status": "paid"}).eq("id", audit_id).execute()
                    except Exception:
                        pass
            except Exception:
                st.markdown(
                    "<div class='ibx-card'><div class='ibx-banner-warn'>Could not save order to Supabase.</div>"
                    "<div class='ibx-muted' style='margin-top:10px;'>Check your tables exist: audits + orders, and that you used the SERVICE ROLE key.</div></div>",
                    unsafe_allow_html=True
                )

        # Premium confirmation layout
        left, right = st.columns([1.2, 0.8], gap="large")

        with left:
            st.markdown("<div class='ibx-card'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:26px; font-weight:950; color:#0b1220;'>Thank you.</div>", unsafe_allow_html=True)
            st.markdown("<div class='ibx-muted' style='margin-top:6px;'>We received your order. Here’s what happens next.</div>", unsafe_allow_html=True)
            st.markdown("<div class='ibx-divider'></div>", unsafe_allow_html=True)

            st.markdown("**Next steps**")
            st.write("• You’ll receive a Stripe confirmation email.")
            st.write("• We’ll match your order to your audit using your IBEX Audit ID.")
            st.write("• You’ll get tracking when it ships.")

            st.markdown("<div class='ibx-divider'></div>", unsafe_allow_html=True)

            st.markdown("**Order details**")
            st.write("**Email:**", cust.get("email", "—"))
            st.write("**Name:**", cust.get("name", "—"))
            if ship:
                addr = (ship.get("address") or {})
                pretty_addr = ", ".join([x for x in [
                    addr.get("line1"), addr.get("line2"), addr.get("city"),
                    addr.get("state"), addr.get("postal_code"), addr.get("country")
                ] if x])
                st.write("**Shipping:**", ship.get("name", "—"))
                st.write(pretty_addr if pretty_addr else "—")

            if line_items and line_items.get("data"):
                st.markdown("**Items**")
                for li in line_items["data"]:
                    st.write(f"• {li.get('description','Item')} × {li.get('quantity',1)}")

            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown("<div class='ibx-card'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:18px; font-weight:950; color:#0b1220;'>Important</div>", unsafe_allow_html=True)
            st.markdown("<div class='ibx-muted' style='margin-top:6px;'>Keep this ID. It links your payment to your personalized recommendations.</div>", unsafe_allow_html=True)

            if audit_id:
                st.markdown(f"<div class='ibx-copy'>{audit_id}</div>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div style="margin-top:12px; display:flex; justify-content:center;">
                      <button onclick="navigator.clipboard.writeText('{audit_id}')" style="
                        padding:10px 18px; border-radius:12px; background:#ef4444; color:white;
                        font-weight:950; border:none; cursor:pointer;
                      ">Copy Audit ID</button>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div class='ibx-banner-warn' style='margin-top:12px;'>No IBEX Audit ID found on this payment.</div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<div class='ibx-muted' style='margin-top:10px;'>Fix: in Stripe Payment Link → collect customer info → add required custom field named <b>IBEX Audit ID</b>.</div>",
                    unsafe_allow_html=True
                )

            st.markdown("<div class='ibx-divider'></div>", unsafe_allow_html=True)
            st.markdown("**Need help?**")
            st.markdown("<div class='ibx-muted'>Email support@ibexperformance.com (replace with your email).</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
        if st.button("Return to IBEX"):
            try:
                st.query_params.clear()
            except Exception:
                pass
            st.rerun()

    else:
        # If stripe session didn't load, still show a clean page
        st.markdown(
            """
            <div class="ibx-card">
              <div style="font-size:24px; font-weight:950; color:#0b1220;">Thank you.</div>
              <div class="ibx-muted" style="margin-top:6px;">
                Your payment completed, but we couldn’t load the order details yet.
              </div>
              <div class="ibx-divider"></div>
              <div class="ibx-muted">Fix the Stripe/Supabase warnings above, then refresh this page.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.stop()


# =========================================================
# MAIN APP (AUDIT)
# =========================================================
if "ai_out" not in st.session_state:
    st.session_state.ai_out = None
if "last_plan" not in st.session_state:
    st.session_state.last_plan = "Basic"
if "last_rid" not in st.session_state:
    st.session_state.last_rid = None

render_header()
tabs = st.tabs(["Audit", "Privacy", "FAQ"])

with tabs[0]:
    st.markdown(
        """
        <div class="ibx-card">
          <div style="font-size:28px; font-weight:950; color:#0b1220;">Performance Audit</div>
          <div class="ibx-muted" style="margin-top:6px;">
            Fill out the audit in the sidebar. Results appear instantly.
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.ai_out and st.session_state.last_rid:
        audit_id_box(st.session_state.last_rid)

        st.markdown("<div class='ibx-card'>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:18px; font-weight:950; color:#0b1220;'>Checkout</div>", unsafe_allow_html=True)
        st.markdown("<div class='ibx-muted' style='margin-top:6px;'>Copy your IBEX Audit ID above and paste it into Stripe during checkout.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.last_plan == "Basic":
            if STRIPE_BASIC_LINK:
                st.link_button("Subscribe — IBEX Basic", STRIPE_BASIC_LINK)
            else:
                st.markdown("<div class='ibx-banner-warn'>Missing STRIPE_BASIC_LINK in Secrets.</div>", unsafe_allow_html=True)
        else:
            if STRIPE_PERF_LINK:
                st.link_button("Subscribe — IBEX Performance", STRIPE_PERF_LINK)
            else:
                st.markdown("<div class='ibx-banner-warn'>Missing STRIPE_PERF_LINK in Secrets.</div>", unsafe_allow_html=True)

        if st.button("Start a new audit"):
            st.session_state.ai_out = None
            st.session_state.last_rid = None
            st.rerun()

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

            st.markdown("### Goals")
            goals = st.multiselect(
                "Select all that apply",
                ["strength", "endurance", "recovery", "sleep", "gut", "joints", "focus", "general health"]
            )

            st.markdown("### Notes")
            open_notes = st.text_area("Other context (optional)")

            st.markdown("---")
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
                "goals": goals,
                "open_notes": open_notes
            }

            # Keep this simple for now
            shortlist = products.head(60)

            with st.spinner("Generating your system…"):
                ai_out = run_ai(intake, shortlist, exclusions, plan)

            # Save audit to Supabase (clean warning if missing)
            supa, supa_err = get_supabase()
            if supa is not None:
                try:
                    supa.table("audits").insert({
                        "id": rid,
                        "plan": plan,
                        "intake": intake,
                        "ai_out": ai_out,
                        "status": "created",
                    }).execute()
                except Exception:
                    st.markdown(
                        "<div class='ibx-card'><div class='ibx-banner-warn'>Could not save audit to Supabase.</div></div>",
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    f"<div class='ibx-card'><div class='ibx-banner-warn'>Supabase not connected: {supa_err}</div></div>",
                    unsafe_allow_html=True
                )

            st.session_state.ai_out = ai_out
            st.session_state.last_plan = plan
            st.session_state.last_rid = rid
            st.rerun()

with tabs[1]:
    render_privacy_policy()

with tabs[2]:
    render_faq()








