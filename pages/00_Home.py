"""
IBEX Landing Page
─────────────────
Instructions:
  1. Create a folder called  pages/  in your repo root (same level as app.py)
  2. Put this file inside it:  pages/00_Home.py
  3. Rename your existing app.py → pages/01_Audit.py
     (Streamlit sorts pages alphabetically, so 00 comes first)
  4. Push to GitHub — Streamlit Cloud will auto-detect the multi-page structure.

That's it. No other changes needed.
"""

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="IBEX — Precision Supplements for D1 Athletes",
    page_icon="assets/ibex_logo.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide all Streamlit default UI chrome
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─── Audit page URL — update if your repo slug differs ───
AUDIT_URL = "https://ibexsupplements.streamlit.app/Audit"

PAGE_HTML = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Barlow:wght@300;400;600;700&family=Barlow+Condensed:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}
:root {{
  --black:  #0a0a0f;
  --navy:   #0b1220;
  --off:    #f0ede6;
  --gold:   #c9a84c;
  --gold2:  #e8c97a;
  --muted:  rgba(240,237,230,0.52);
  --border: rgba(201,168,76,0.18);
}}
html {{ scroll-behavior:smooth; }}
body {{
  background:var(--black); color:var(--off);
  font-family:'Barlow',sans-serif; font-weight:300;
  overflow-x:hidden;
}}

/* ── NAV ── */
nav {{
  position:sticky; top:0; z-index:999;
  display:flex; align-items:center; justify-content:space-between;
  padding:0.85rem 3rem;
  background:rgba(10,10,15,0.96); backdrop-filter:blur(16px);
  border-bottom:1px solid var(--border);
}}
.nav-brand {{ display:flex; align-items:center; gap:0.7rem; text-decoration:none; }}
.nav-wordmark {{ font-family:'Bebas Neue',sans-serif; font-size:1.75rem; letter-spacing:0.1em; color:var(--gold); }}
.nav-links {{ display:flex; gap:2rem; list-style:none; }}
.nav-links a {{
  font-family:'Barlow Condensed',sans-serif;
  font-size:0.7rem; letter-spacing:0.2em; text-transform:uppercase;
  color:var(--off); opacity:0.6; text-decoration:none; transition:opacity .2s,color .2s;
}}
.nav-links a:hover {{ opacity:1; color:var(--gold); }}
.nav-cta {{
  font-family:'Barlow Condensed',sans-serif;
  font-size:0.7rem; letter-spacing:0.2em; text-transform:uppercase; font-weight:700;
  background:var(--gold); color:var(--black);
  padding:0.5rem 1.2rem; text-decoration:none; transition:background .2s;
}}
.nav-cta:hover {{ background:var(--gold2); }}

/* ── HERO ── */
#hero {{
  min-height:92vh; display:grid; grid-template-columns:55% 45%;
  overflow:hidden; position:relative;
}}
.hero-divider {{
  position:absolute; top:0; left:55%; width:1px; height:100%;
  background:var(--border); z-index:3;
}}
.hero-left {{
  display:flex; flex-direction:column; justify-content:center;
  padding:5rem 3rem 4rem 3.5rem; position:relative; z-index:2;
}}
.hero-eyebrow {{
  font-family:'Barlow Condensed',sans-serif;
  font-size:0.65rem; letter-spacing:0.4em; text-transform:uppercase;
  color:var(--gold); margin-bottom:1.6rem;
  opacity:0; animation:fadeUp .7s .1s forwards;
}}
.hero-h1 {{
  font-family:'Bebas Neue',sans-serif;
  font-size:clamp(4rem,8vw,8rem); line-height:.9; letter-spacing:.02em;
  margin-bottom:1.4rem; opacity:0; animation:fadeUp .7s .25s forwards;
}}
.hero-h1 em {{ color:var(--gold); font-style:normal; }}
.hero-sub {{
  font-size:.98rem; line-height:1.75; color:var(--muted);
  max-width:420px; margin-bottom:2.2rem;
  opacity:0; animation:fadeUp .7s .4s forwards;
}}
.hero-actions {{
  display:flex; gap:1rem; align-items:center; flex-wrap:wrap;
  opacity:0; animation:fadeUp .7s .55s forwards;
}}
.btn-gold {{
  font-family:'Barlow Condensed',sans-serif;
  font-size:0.76rem; letter-spacing:.2em; text-transform:uppercase; font-weight:700;
  background:var(--gold); color:var(--black);
  padding:.9rem 1.9rem; text-decoration:none; display:inline-block;
  transition:background .2s,transform .15s;
}}
.btn-gold:hover {{ background:var(--gold2); transform:translateY(-1px); }}
.btn-ghost {{
  font-family:'Barlow Condensed',sans-serif;
  font-size:0.76rem; letter-spacing:.2em; text-transform:uppercase;
  color:var(--off); opacity:.55; text-decoration:none;
  border-bottom:1px solid currentColor; padding-bottom:2px; transition:opacity .2s;
}}
.btn-ghost:hover {{ opacity:1; }}
.hero-trust {{
  margin-top:2rem; display:flex; gap:1.2rem; flex-wrap:wrap;
  opacity:0; animation:fadeUp .7s .7s forwards;
}}
.trust-pill {{
  font-family:'Barlow Condensed',sans-serif;
  font-size:0.62rem; letter-spacing:.15em; text-transform:uppercase;
  color:var(--muted); border:1px solid rgba(201,168,76,.2); padding:.3rem .75rem;
}}
.hero-right {{
  position:relative; overflow:hidden;
  display:flex; align-items:center; justify-content:center;
}}
.hero-bg-mark {{
  position:absolute; right:-5%; top:50%; transform:translateY(-50%);
  font-family:'Bebas Neue',sans-serif; font-size:32vw;
  color:rgba(201,168,76,.03); pointer-events:none; line-height:1;
}}
.hero-stat-grid {{
  position:relative; z-index:1;
  display:grid; grid-template-columns:1fr 1fr; gap:1px;
  border:1px solid var(--border);
  opacity:0; animation:fadeIn 1s .6s forwards;
  width:min(340px,38vw);
}}
.stat-cell {{
  background:rgba(201,168,76,.04); padding:2rem 1.5rem;
  border:1px solid rgba(201,168,76,.07); transition:background .3s;
}}
.stat-cell:hover {{ background:rgba(201,168,76,.09); }}
.stat-num {{ font-family:'Bebas Neue',sans-serif; font-size:3.2rem; color:var(--gold); line-height:1; margin-bottom:.3rem; }}
.stat-label {{ font-family:'Barlow Condensed',sans-serif; font-size:0.65rem; letter-spacing:.2em; text-transform:uppercase; color:var(--muted); }}

/* ── TICKER ── */
.ticker {{ background:var(--gold); overflow:hidden; padding:.6rem 0; white-space:nowrap; }}
.ticker-inner {{ display:inline-flex; animation:ticker 30s linear infinite; }}
.t-item {{
  font-family:'Barlow Condensed',sans-serif;
  font-size:0.7rem; letter-spacing:.25em; text-transform:uppercase;
  color:var(--black); font-weight:700; padding:0 2rem;
}}
.t-dot {{ color:rgba(0,0,0,.25); }}
@keyframes ticker {{ 0%{{transform:translateX(0)}} 100%{{transform:translateX(-50%)}} }}

/* ── SECTIONS ── */
.section {{ padding:7rem 3.5rem; }}
.section-inner {{ max-width:1200px; margin:0 auto; }}
.section-label {{
  font-family:'Barlow Condensed',sans-serif;
  font-size:0.63rem; letter-spacing:.4em; text-transform:uppercase;
  color:var(--gold); margin-bottom:.7rem;
}}
.section-title {{
  font-family:'Bebas Neue',sans-serif;
  font-size:clamp(2.6rem,5vw,4.8rem); line-height:.93; margin-bottom:3rem;
}}
.section-sub {{
  font-size:.9rem; line-height:1.8; color:var(--muted);
  max-width:600px; margin-bottom:2.5rem;
}}

/* ── HOW ── */
#how {{ background:var(--navy); }}
.steps {{ display:grid; grid-template-columns:repeat(3,1fr); border:1px solid var(--border); }}
.step {{ padding:2.5rem 2rem; border-right:1px solid var(--border); transition:background .3s; }}
.step:last-child {{ border-right:none; }}
.step:hover {{ background:rgba(201,168,76,.04); }}
.step-num {{ font-family:'Bebas Neue',sans-serif; font-size:4rem; color:rgba(201,168,76,.1); line-height:1; margin-bottom:.7rem; }}
.step-title {{ font-family:'Barlow Condensed',sans-serif; font-size:1.1rem; font-weight:700; letter-spacing:.05em; text-transform:uppercase; color:var(--off); margin-bottom:.6rem; }}
.step-body {{ font-size:.86rem; line-height:1.75; color:var(--muted); }}

/* ── CATALOG ── */
#catalog {{ background:var(--black); }}
.cat-filter {{ display:flex; gap:.5rem; flex-wrap:wrap; margin-bottom:2rem; }}
.cat-btn {{
  font-family:'Barlow Condensed',sans-serif;
  font-size:0.65rem; letter-spacing:.18em; text-transform:uppercase; font-weight:600;
  padding:.35rem .85rem; border:1px solid var(--border);
  background:transparent; color:var(--muted); cursor:pointer; transition:all .2s;
}}
.cat-btn:hover,.cat-btn.active {{ background:var(--gold); color:var(--black); border-color:var(--gold); }}
.supp-grid {{
  display:grid; grid-template-columns:repeat(auto-fill,minmax(270px,1fr));
  gap:1px; background:var(--border); border:1px solid var(--border);
}}
.supp-card {{ background:var(--black); padding:1.6rem; transition:background .25s; }}
.supp-card:hover {{ background:rgba(201,168,76,.04); }}
.supp-card.hidden {{ display:none; }}
.supp-header {{ display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:.7rem; }}
.supp-cat {{ font-family:'Barlow Condensed',sans-serif; font-size:.58rem; letter-spacing:.2em; text-transform:uppercase; color:var(--gold); }}
.supp-badges {{ display:flex; gap:.25rem; flex-wrap:wrap; }}
.badge {{ font-family:'Barlow Condensed',sans-serif; font-size:.52rem; letter-spacing:.1em; text-transform:uppercase; font-weight:700; padding:.18rem .45rem; border:1px solid rgba(201,168,76,.25); color:var(--gold); }}
.badge.ncaa {{ border-color:rgba(34,197,94,.35); color:#4ade80; }}
.badge.nsf  {{ border-color:rgba(96,165,250,.35); color:#7dd3fc; }}
.supp-name {{ font-family:'Barlow Condensed',sans-serif; font-size:1.1rem; font-weight:700; color:var(--off); margin-bottom:.25rem; }}
.supp-dose {{ font-size:.76rem; color:var(--muted); margin-bottom:.5rem; }}
.supp-use  {{ font-size:.8rem; line-height:1.65; color:rgba(240,237,230,.58); }}
.supp-timing {{ margin-top:.7rem; font-family:'Barlow Condensed',sans-serif; font-size:.6rem; letter-spacing:.15em; text-transform:uppercase; color:rgba(201,168,76,.6); }}

/* ── AI DEMO ── */
#ai {{ background:var(--navy); }}
.ai-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:4rem; align-items:start; }}
.ai-tagline {{ font-family:'Bebas Neue',sans-serif; font-size:clamp(2.4rem,4vw,4rem); line-height:.93; margin-bottom:1.2rem; }}
.ai-tagline em {{ color:var(--gold); font-style:normal; }}
.ai-body {{ font-size:.93rem; line-height:1.8; color:var(--muted); margin-bottom:1.6rem; }}
.feature-list {{ list-style:none; margin-bottom:2rem; }}
.feature-list li {{
  font-size:.86rem; line-height:1.7; padding:.45rem 0;
  border-bottom:1px solid rgba(201,168,76,.08);
  color:rgba(240,237,230,.7); display:flex; gap:.7rem; align-items:flex-start;
}}
.feature-list li::before {{ content:'→'; color:var(--gold); flex-shrink:0; }}
.ai-demo {{ background:rgba(201,168,76,.04); border:1px solid var(--border); padding:1.8rem; }}
.ai-demo-label {{ font-family:'Barlow Condensed',sans-serif; font-size:.62rem; letter-spacing:.3em; text-transform:uppercase; color:var(--gold); margin-bottom:1rem; }}
.demo-inputs {{ display:flex; flex-direction:column; gap:.6rem; margin-bottom:1rem; }}
.demo-select {{
  background:rgba(11,18,32,.9); border:1px solid rgba(201,168,76,.2);
  color:var(--off); padding:.6rem .9rem;
  font-family:'Barlow',sans-serif; font-size:.85rem;
  width:100%; appearance:none; cursor:pointer;
}}
.demo-select:focus {{ outline:none; border-color:var(--gold); }}
.demo-run-btn {{
  font-family:'Barlow Condensed',sans-serif;
  font-size:.75rem; letter-spacing:.2em; text-transform:uppercase; font-weight:700;
  background:var(--gold); color:var(--black);
  border:none; padding:.8rem 1.5rem; cursor:pointer; width:100%; transition:background .2s;
}}
.demo-run-btn:hover {{ background:var(--gold2); }}
.demo-output {{ background:var(--navy); border:1px solid rgba(201,168,76,.12); padding:1.2rem; min-height:160px; margin-top:.8rem; }}
.demo-output-label {{ font-family:'Barlow Condensed',sans-serif; font-size:.6rem; letter-spacing:.25em; text-transform:uppercase; color:var(--gold); margin-bottom:.8rem; }}
.demo-pill {{
  display:flex; justify-content:space-between; align-items:center;
  padding:.45rem 0; border-bottom:1px solid rgba(201,168,76,.08);
  font-size:.82rem; color:rgba(240,237,230,.8);
}}
.demo-pill:last-child {{ border-bottom:none; }}
.demo-pill span {{ color:var(--gold); font-size:.72rem; }}
.demo-placeholder {{ font-size:.82rem; color:var(--muted); text-align:center; padding:2rem 0; }}

/* ── SCHEDULE ── */
#schedule {{ background:var(--black); }}
.sched-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:1.5rem; }}
.sched-block {{ border:1px solid var(--border); padding:2rem; transition:background .3s; }}
.sched-block:hover {{ background:rgba(201,168,76,.03); }}
.sched-time {{ font-family:'Bebas Neue',sans-serif; font-size:1.8rem; color:var(--gold); margin-bottom:.15rem; }}
.sched-sub {{ font-family:'Barlow Condensed',sans-serif; font-size:.63rem; letter-spacing:.2em; text-transform:uppercase; color:var(--muted); margin-bottom:1.2rem; }}
.sched-item {{ display:flex; justify-content:space-between; padding:.45rem 0; border-bottom:1px solid rgba(201,168,76,.07); font-size:.84rem; color:rgba(240,237,230,.75); }}
.sched-item:last-child {{ border-bottom:none; }}
.sched-dose {{ color:var(--gold); font-size:.72rem; }}

/* ── PLANS ── */
#plans {{ background:var(--navy); position:relative; overflow:hidden; }}
#plans::before {{
  content:'IBEX'; position:absolute; right:-.02em; bottom:-.1em;
  font-family:'Bebas Neue',sans-serif; font-size:22vw;
  color:rgba(201,168,76,.025); pointer-events:none; line-height:1;
}}
#plans .section-inner {{ position:relative; z-index:1; }}
.plan-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; }}
.plan-card {{ border:1px solid var(--border); padding:2.8rem; position:relative; transition:border-color .3s; }}
.plan-card:hover {{ border-color:rgba(201,168,76,.45); }}
.plan-card.featured {{ border-color:var(--gold); background:rgba(201,168,76,.03); }}
.featured-tag {{
  position:absolute; top:-1px; right:2rem;
  background:var(--gold); color:var(--black);
  font-family:'Barlow Condensed',sans-serif;
  font-size:.62rem; letter-spacing:.2em; text-transform:uppercase; font-weight:700;
  padding:.28rem .75rem;
}}
.plan-tier {{ font-family:'Barlow Condensed',sans-serif; font-size:.68rem; letter-spacing:.3em; text-transform:uppercase; color:var(--gold); margin-bottom:.7rem; }}
.plan-headline {{ font-family:'Barlow Condensed',sans-serif; font-size:1rem; font-weight:600; color:var(--muted); margin-bottom:1.4rem; font-style:italic; }}
.plan-price {{ font-family:'Bebas Neue',sans-serif; font-size:4.5rem; line-height:1; margin-bottom:.2rem; }}
.plan-period {{ font-size:.78rem; color:var(--muted); margin-bottom:2rem; }}
.plan-features {{ list-style:none; margin-bottom:2rem; }}
.plan-features li {{
  font-size:.85rem; line-height:1.65; padding:.5rem 0;
  border-bottom:1px solid rgba(201,168,76,.08);
  color:rgba(240,237,230,.7); display:flex; gap:.65rem;
}}
.plan-features li::before {{ content:'→'; color:var(--gold); flex-shrink:0; }}
.btn-plan {{
  font-family:'Barlow Condensed',sans-serif;
  font-size:.76rem; letter-spacing:.2em; text-transform:uppercase; font-weight:700;
  padding:.85rem 2rem; border:1px solid var(--gold);
  color:var(--gold); background:transparent; cursor:pointer;
  text-decoration:none; display:block; text-align:center; transition:all .2s;
}}
.btn-plan:hover {{ background:var(--gold); color:var(--black); }}
.plan-card.featured .btn-plan {{ background:var(--gold); color:var(--black); }}
.plan-card.featured .btn-plan:hover {{ background:var(--gold2); }}

/* ── PROOF ── */
#proof {{ background:var(--black); }}
.proof-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:1.5rem; }}
.proof-card {{ border:1px solid rgba(201,168,76,.1); padding:2.2rem; transition:all .3s; }}
.proof-card:hover {{ border-color:rgba(201,168,76,.3); background:rgba(201,168,76,.02); }}
.proof-stars {{ color:var(--gold); font-size:.8rem; margin-bottom:1rem; letter-spacing:.1em; }}
.proof-quote {{ font-size:.9rem; line-height:1.75; color:rgba(240,237,230,.65); font-style:italic; margin-bottom:1.2rem; }}
.proof-author {{ font-family:'Barlow Condensed',sans-serif; font-size:.68rem; letter-spacing:.2em; text-transform:uppercase; color:var(--gold); }}
.proof-sport {{ font-size:.74rem; color:rgba(240,237,230,.3); margin-top:.15rem; }}

/* ── FAQ ── */
#faq {{ background:var(--navy); }}
.faq-wrap {{ max-width:820px; }}
.faq-item {{ border-bottom:1px solid rgba(201,168,76,.12); }}
.faq-q {{
  display:flex; justify-content:space-between; align-items:center;
  padding:1.5rem 0; cursor:pointer; user-select:none;
  font-family:'Barlow Condensed',sans-serif;
  font-size:1.05rem; font-weight:700; letter-spacing:.02em; color:var(--off); gap:1rem;
}}
.faq-icon {{ color:var(--gold); font-size:1.2rem; flex-shrink:0; transition:transform .3s; }}
.faq-a {{
  max-height:0; overflow:hidden; font-size:.87rem; line-height:1.8; color:var(--muted);
  transition:max-height .4s ease,padding .3s;
}}
.faq-item.open .faq-a {{ max-height:280px; padding-bottom:1.4rem; }}
.faq-item.open .faq-icon {{ transform:rotate(45deg); }}

/* ── CTA ── */
#cta {{
  background:var(--black); padding:9rem 3.5rem; text-align:center;
  position:relative; overflow:hidden;
}}
#cta::before {{
  content:''; position:absolute; inset:0;
  background:radial-gradient(ellipse 55% 55% at 50% 50%,rgba(201,168,76,.07) 0%,transparent 70%);
  pointer-events:none;
}}
.cta-title {{
  font-family:'Bebas Neue',sans-serif;
  font-size:clamp(3.5rem,7vw,7rem); line-height:.93;
  margin-bottom:1.3rem; position:relative; z-index:1;
}}
.cta-title em {{ color:var(--gold); font-style:normal; }}
.cta-sub {{ font-size:1rem; color:var(--muted); margin-bottom:2.5rem; position:relative; z-index:1; }}

/* ── FOOTER ── */
footer {{
  padding:2rem 3.5rem; border-top:1px solid var(--border);
  display:flex; justify-content:space-between; align-items:center; gap:1rem; flex-wrap:wrap;
}}
.footer-wordmark {{ font-family:'Bebas Neue',sans-serif; font-size:1.4rem; letter-spacing:.1em; color:var(--gold); }}
.footer-copy {{ font-size:.72rem; color:rgba(240,237,230,.22); letter-spacing:.05em; }}
.footer-links {{ display:flex; gap:1.5rem; flex-wrap:wrap; }}
.footer-links a {{
  font-family:'Barlow Condensed',sans-serif;
  font-size:.68rem; letter-spacing:.15em; text-transform:uppercase;
  color:rgba(240,237,230,.28); text-decoration:none; transition:color .2s;
}}
.footer-links a:hover {{ color:var(--gold); }}
.footer-disclaimer {{
  width:100%; font-size:.68rem; color:rgba(240,237,230,.2); line-height:1.6;
  border-top:1px solid rgba(201,168,76,.07); padding-top:1rem; margin-top:.5rem;
}}

/* ── ANIMATIONS ── */
@keyframes fadeUp {{ from{{opacity:0;transform:translateY(18px)}} to{{opacity:1;transform:translateY(0)}} }}
@keyframes fadeIn  {{ from{{opacity:0}} to{{opacity:1}} }}
.reveal {{
  opacity:0; transform:translateY(20px);
  transition:opacity .65s ease,transform .65s ease;
}}
.reveal.visible {{ opacity:1; transform:translateY(0); }}

/* ── RESPONSIVE ── */
@media(max-width:900px){{
  nav {{ padding:.9rem 1.5rem; }}
  .nav-links {{ display:none; }}
  #hero {{ grid-template-columns:1fr; min-height:auto; }}
  .hero-right,.hero-divider {{ display:none; }}
  .hero-left {{ padding:4rem 1.5rem 3rem; }}
  .section {{ padding:5rem 1.5rem; }}
  .steps,.sched-grid,.plan-grid,.proof-grid {{ grid-template-columns:1fr; }}
  .step {{ border-right:none; border-bottom:1px solid var(--border); }}
  .step:last-child {{ border-bottom:none; }}
  .ai-grid {{ grid-template-columns:1fr; }}
  footer {{ flex-direction:column; text-align:center; }}
}}
</style>
</head>
<body>

<!-- NAV -->
<nav>
  <a class="nav-brand" href="#">
    <span class="nav-wordmark">IBEX</span>
  </a>
  <ul class="nav-links">
    <li><a href="#how">How It Works</a></li>
    <li><a href="#catalog">The Stack</a></li>
    <li><a href="#ai">The AI</a></li>
    <li><a href="#plans">Plans</a></li>
    <li><a href="#faq">FAQ</a></li>
  </ul>
  <a class="nav-cta" href="{AUDIT_URL}" target="_top">Build My Stack — Free</a>
</nav>

<!-- HERO -->
<section id="hero">
  <div class="hero-left">
    <div class="hero-eyebrow">D1 Athlete Performance Systems · NCAA Compliant</div>
    <h1 class="hero-h1">YOUR STACK.<br>YOUR <em>SPORT.</em><br>YOUR EDGE.</h1>
    <p class="hero-sub">IBEX uses AI to build a personalized supplement protocol for D1 athletes — tuned to your sport, position, training load, and recovery. Shipped monthly. Cancel anytime.</p>
    <div class="hero-actions">
      <a class="btn-gold" href="{AUDIT_URL}" target="_top">Run My Free Audit →</a>
      <a class="btn-ghost" href="#plans">View Plans</a>
    </div>
    <div class="hero-trust">
      <span class="trust-pill">NCAA Compliant</span>
      <span class="trust-pill">Third-Party Tested</span>
      <span class="trust-pill">Evidence-Linked</span>
      <span class="trust-pill">Cancel Anytime</span>
    </div>
  </div>
  <div class="hero-divider"></div>
  <div class="hero-right">
    <div class="hero-bg-mark">I</div>
    <div class="hero-stat-grid">
      <div class="stat-cell"><div class="stat-num">D1</div><div class="stat-label">Athletes Only</div></div>
      <div class="stat-cell"><div class="stat-num">AI</div><div class="stat-label">Personalized Stack</div></div>
      <div class="stat-cell"><div class="stat-num">30</div><div class="stat-label">Day Supply</div></div>
      <div class="stat-cell"><div class="stat-num">$100</div><div class="stat-label">Starting / Mo</div></div>
    </div>
  </div>
</section>

<!-- TICKER -->
<div class="ticker">
  <div class="ticker-inner">
    <span class="t-item">AI-Personalized Stack</span><span class="t-dot"> ✦ </span>
    <span class="t-item">NCAA Compliant</span><span class="t-dot"> ✦ </span>
    <span class="t-item">Free Audit Tool</span><span class="t-dot"> ✦ </span>
    <span class="t-item">Third-Party Tested</span><span class="t-dot"> ✦ </span>
    <span class="t-item">Shipped Monthly</span><span class="t-dot"> ✦ </span>
    <span class="t-item">In-Season & Off-Season Protocols</span><span class="t-dot"> ✦ </span>
    <span class="t-item">Evidence-Linked Ingredients</span><span class="t-dot"> ✦ </span>
    <span class="t-item">Cancel Anytime</span><span class="t-dot"> ✦ </span>
    <span class="t-item">AI-Personalized Stack</span><span class="t-dot"> ✦ </span>
    <span class="t-item">NCAA Compliant</span><span class="t-dot"> ✦ </span>
    <span class="t-item">Free Audit Tool</span><span class="t-dot"> ✦ </span>
    <span class="t-item">Third-Party Tested</span><span class="t-dot"> ✦ </span>
    <span class="t-item">Shipped Monthly</span><span class="t-dot"> ✦ </span>
    <span class="t-item">In-Season & Off-Season Protocols</span><span class="t-dot"> ✦ </span>
    <span class="t-item">Evidence-Linked Ingredients</span><span class="t-dot"> ✦ </span>
    <span class="t-item">Cancel Anytime</span><span class="t-dot"> ✦ </span>
  </div>
</div>

<!-- HOW IT WORKS -->
<section id="how" class="section">
  <div class="section-inner">
    <div class="section-label reveal">The Process</div>
    <h2 class="section-title reveal">FROM AUDIT<br>TO DOORSTEP</h2>
    <div class="steps">
      <div class="step reveal">
        <div class="step-num">01</div>
        <div class="step-title">Run Your Free Audit</div>
        <p class="step-body">Answer ~15 questions: sport, position, season status, training load, sleep, stress, soreness, goals, and sensitivities. 3 minutes. No card needed.</p>
      </div>
      <div class="step reveal">
        <div class="step-num">02</div>
        <div class="step-title">AI Builds Your Protocol</div>
        <p class="step-body">Our AI picks from an NCAA-safe curated catalog and builds your stack with exact doses and a daily AM / PM / Training-window schedule.</p>
      </div>
      <div class="step reveal">
        <div class="step-num">03</div>
        <div class="step-title">Ask IBEX. Ship. Perform.</div>
        <p class="step-body">Your personalized chat answers questions about your stack. Subscribe, paste your Audit ID at checkout, and your 30-day supply ships to your door.</p>
      </div>
    </div>
  </div>
</section>

<!-- SUPPLEMENT CATALOG -->
<section id="catalog" class="section">
  <div class="section-inner">
    <div class="section-label reveal">The Catalog</div>
    <h2 class="section-title reveal">WHAT'S IN<br>YOUR STACK</h2>
    <p class="section-sub reveal">Every ingredient is NCAA-compliant and evidence-backed. Nothing trendy. Nothing sketchy. Just the ingredients that actually move the needle for D1 athletes.</p>
    <div class="cat-filter reveal">
      <button class="cat-btn active" data-cat="all">All</button>
      <button class="cat-btn" data-cat="strength">Strength</button>
      <button class="cat-btn" data-cat="recovery">Recovery</button>
      <button class="cat-btn" data-cat="endurance">Endurance</button>
      <button class="cat-btn" data-cat="sleep">Sleep</button>
      <button class="cat-btn" data-cat="foundation">Foundation</button>
      <button class="cat-btn" data-cat="gut">Gut & Joint</button>
    </div>
    <div class="supp-grid" id="suppGrid">

      <div class="supp-card" data-cat="strength">
        <div class="supp-header"><span class="supp-cat">Strength</span><div class="supp-badges"><span class="badge ncaa">NCAA ✓</span><span class="badge nsf">NSF</span></div></div>
        <div class="supp-name">Creatine Monohydrate</div>
        <div class="supp-dose">5g / day · Powder</div>
        <div class="supp-use">The most-studied performance supplement ever. Increases phosphocreatine for explosive power, sprint capacity, and faster recovery between sets.</div>
        <div class="supp-timing">⏱ Post-training or AM</div>
      </div>

      <div class="supp-card" data-cat="strength">
        <div class="supp-header"><span class="supp-cat">Strength / Endurance</span><div class="supp-badges"><span class="badge ncaa">NCAA ✓</span></div></div>
        <div class="supp-name">Beta-Alanine</div>
        <div class="supp-dose">3.2g / day · Capsule or Powder</div>
        <div class="supp-use">Buffers muscle acid during repeated high-intensity efforts. Best for sports with multiple bouts — wrestling, lacrosse, rowing, basketball.</div>
        <div class="supp-timing">⏱ Pre-training</div>
      </div>

      <div class="supp-card" data-cat="strength">
        <div class="supp-header"><span class="supp-cat">Strength</span><div class="supp-badges"><span class="badge ncaa">NCAA ✓</span><span class="badge nsf">NSF</span></div></div>
        <div class="supp-name">Whey Protein</div>
        <div class="supp-dose">25–40g · Powder</div>
        <div class="supp-use">Fast-digesting complete protein to kick-start muscle protein synthesis. Essential for athletes with high volume and tight recovery windows.</div>
        <div class="supp-timing">⏱ Within 30 min post-training</div>
      </div>

      <div class="supp-card" data-cat="recovery">
        <div class="supp-header"><span class="supp-cat">Recovery</span><div class="supp-badges"><span class="badge ncaa">NCAA ✓</span><span class="badge nsf">NSF</span></div></div>
        <div class="supp-name">Tart Cherry Extract</div>
        <div class="supp-dose">480mg / day · Capsule</div>
        <div class="supp-use">Reduces exercise-induced soreness and inflammation. Particularly effective for back-to-back game days and high-volume training blocks.</div>
        <div class="supp-timing">⏱ AM + PM</div>
      </div>

      <div class="supp-card" data-cat="recovery">
        <div class="supp-header"><span class="supp-cat">Recovery / Sleep</span><div class="supp-badges"><span class="badge ncaa">NCAA ✓</span><span class="badge nsf">NSF</span></div></div>
        <div class="supp-name">Magnesium Glycinate</div>
        <div class="supp-dose">400mg / night · Capsule</div>
        <div class="supp-use">Supports muscle relaxation, sleep quality, and protein synthesis. Most athletes are deficient — especially those with high sweat rates.</div>
        <div class="supp-timing">⏱ 30 min before bed</div>
      </div>

      <div class="supp-card" data-cat="recovery">
        <div class="supp-header"><span class="supp-cat">Recovery</span><div class="supp-badges"><span class="badge ncaa">NCAA ✓</span><span class="badge nsf">NSF</span></div></div>
        <div class="supp-name">Omega-3 Fish Oil</div>
        <div class="supp-dose">2–3g EPA/DHA / day · Softgel</div>
        <div class="supp-use">Reduces systemic inflammation, supports joint health and cardiovascular efficiency. Among the highest-evidence supplements for athletes.</div>
        <div class="supp-timing">⏱ With meals</div>
      </div>

      <div class="supp-card" data-cat="recovery">
        <div class="supp-header"><span class="supp-cat">Recovery / Joint</span><div class="supp-badges"><span class="badge ncaa">NCAA ✓</span></div></div>
        <div class="supp-name">Collagen + Vitamin C</div>
        <div class="supp-dose">15g collagen + 50mg Vit C · Powder</div>
        <div class="supp-use">Taken pre-activity, supports connective tissue synthesis. Especially valuable for joint-heavy sports: basketball, volleyball, wrestling, gymnastics.</div>
        <div class="supp-timing">⏱ 45 min pre-training</div>
      </div>

      <div class="supp-card" data-cat="endurance">
        <div class="supp-header"><span class="supp-cat">Endurance</span><div class="supp-badges"><span class="badge ncaa">NCAA ✓</span><span class="badge nsf">NSF</span></div></div>
        <div class="supp-name">Electrolyte Complex</div>
        <div class="supp-dose">Sodium / Potassium / Magnesium blend</div>
        <div class="supp-use">Replaces sweat losses to maintain performance and prevent cramping. Essential during preseason camps, doubles, and multi-event days.</div>
        <div class="supp-timing">⏱ During training</div>
      </div>

      <div class="supp-card" data-cat="endurance">
        <div class="supp-header"><span class="supp-cat">Endurance / Vascular</span><div class="supp-badges"><span class="badge ncaa">NCAA ✓</span></div></div>
        <div class="supp-name">Beetroot / Nitrate</div>
        <div class="supp-dose">400mg nitrate · Powder or Capsule</div>
        <div class="supp-use">Boosts nitric oxide to improve blood flow and O2 efficiency. Strong evidence for endurance sports — swimming, rowing, soccer, cross country.</div>
        <div class="supp-timing">⏱ 2–3 hrs pre-training</div>
      </div>

      <div class="supp-card" data-cat="sleep">
        <div class="supp-header"><span class="supp-cat">Sleep</span><div class="supp-badges"><span class="badge ncaa">NCAA ✓</span><span class="badge nsf">NSF</span></div></div>
        <div class="supp-name">Melatonin (Low-Dose)</div>
        <div class="supp-dose">0.5–1mg · Sublingual or Capsule</div>
        <div class="supp-use">Low-dose melatonin supports sleep onset without grogginess. Especially useful for athletes traveling across time zones or with late-night practices.</div>
        <div class="supp-timing">⏱ 30 min before bed</div>
      </div>

      <div class="supp-card" data-cat="sleep">
        <div class="supp-header"><span class="supp-cat">Sleep / Recovery</span><div class="supp-badges"><span class="badge ncaa">NCAA ✓</span></div></div>
        <div class="supp-name">Ashwagandha (KSM-66)</div>
        <div class="supp-dose">300–600mg / day · Capsule</div>
        <div class="supp-use">Reduces cortisol and perceived stress. Improves sleep quality and recovery. Growing evidence for strength and VO2 max in high-stress training phases.</div>
        <div class="supp-timing">⏱ PM or with dinner</div>
      </div>

      <div class="supp-card" data-cat="foundation">
        <div class="supp-header"><span class="supp-cat">Foundation</span><div class="supp-badges"><span class="badge ncaa">NCAA ✓</span><span class="badge nsf">NSF</span></div></div>
        <div class="supp-name">Vitamin D3 + K2</div>
        <div class="supp-dose">3000–5000 IU D3 + 100mcg K2 · Softgel</div>
        <div class="supp-use">Bone health, immune function, testosterone production, mood. A huge proportion of D1 athletes are deficient — especially indoor sport athletes.</div>
        <div class="supp-timing">⏱ AM with fat</div>
      </div>

      <div class="supp-card" data-cat="foundation">
        <div class="supp-header"><span class="supp-cat">Foundation</span><div class="supp-badges"><span class="badge ncaa">NCAA ✓</span><span class="badge nsf">NSF</span></div></div>
        <div class="supp-name">Zinc + Copper</div>
        <div class="supp-dose">25mg Zinc + 2mg Copper · Capsule</div>
        <div class="supp-use">Supports immune function, testosterone, and recovery. Heavy training depletes zinc rapidly through sweat. Copper balances absorption.</div>
        <div class="supp-timing">⏱ PM with food</div>
      </div>

      <div class="supp-card" data-cat="foundation">
        <div class="supp-header"><span class="supp-cat">Foundation</span><div class="supp-badges"><span class="badge ncaa">NCAA ✓</span><span class="badge nsf">NSF</span></div></div>
        <div class="supp-name">Sport Multivitamin</div>
        <div class="supp-dose">Daily · Capsule</div>
        <div class="supp-use">Fills micronutrient gaps when training volume is high and dietary variety is limited. The insurance policy for your entire system. NSF-certified only.</div>
        <div class="supp-timing">⏱ AM with breakfast</div>
      </div>

      <div class="supp-card" data-cat="gut">
        <div class="supp-header"><span class="supp-cat">Gut Health</span><div class="supp-badges"><span class="badge ncaa">NCAA ✓</span></div></div>
        <div class="supp-name">Probiotic (Multi-Strain)</div>
        <div class="supp-dose">10–50 billion CFU · Capsule</div>
        <div class="supp-use">Supports gut integrity, immune function, and nutrient absorption. High training stress suppresses gut health — foundational for high-volume athletes.</div>
        <div class="supp-timing">⏱ AM on empty stomach</div>
      </div>

      <div class="supp-card" data-cat="gut">
        <div class="supp-header"><span class="supp-cat">Joint Health</span><div class="supp-badges"><span class="badge ncaa">NCAA ✓</span></div></div>
        <div class="supp-name">Curcumin + Piperine</div>
        <div class="supp-dose">500–1000mg / day · Capsule</div>
        <div class="supp-use">Potent anti-inflammatory for joint pain and soreness. Piperine boosts bioavailability 20x. Best for athletes with chronic joint stress or heavy contact.</div>
        <div class="supp-timing">⏱ With meals</div>
      </div>

    </div>
  </div>
</section>

<!-- AI SECTION -->
<section id="ai" class="section">
  <div class="section-inner">
    <div class="ai-grid">
      <div>
        <div class="section-label reveal">The Intelligence</div>
        <h2 class="ai-tagline reveal">NOT GENERIC.<br>NOT A GUESS.<br><em>YOURS.</em></h2>
        <p class="ai-body reveal">Most brands sell everyone the same box. IBEX's AI reads your full profile — sport, position, training volume, sleep, stress, soreness, sensitivities — and builds a stack that actually fits your life.</p>
        <ul class="feature-list reveal">
          <li>Sport + position-specific selection from a curated NCAA-safe catalog</li>
          <li>GI and caffeine sensitivity flags auto-remove incompatible items</li>
          <li>In-season vs off-season protocol adjustments built in</li>
          <li>Exact AM / PM / Training-window schedule per item</li>
          <li>Ask IBEX Chat answers questions grounded in your audit + evidence links</li>
          <li>No invented citations — if no study is linked, IBEX says so</li>
          <li>Audit ID saved for Stripe checkout order matching</li>
        </ul>
        <a class="btn-gold reveal" href="{AUDIT_URL}" target="_top">Run Your Free Audit →</a>
      </div>
      <div class="reveal">
        <div class="ai-demo">
          <div class="ai-demo-label">⚡ Preview — Sample Stack Generator</div>
          <div class="demo-inputs">
            <select class="demo-select" id="dSport">
              <option value="">Select your sport…</option>
              <option value="football">Football</option>
              <option value="basketball">Basketball</option>
              <option value="soccer">Soccer</option>
              <option value="wrestling">Wrestling</option>
              <option value="swimming">Swimming</option>
              <option value="lacrosse">Lacrosse</option>
              <option value="track">Track & Field</option>
              <option value="baseball">Baseball</option>
            </select>
            <select class="demo-select" id="dGoal">
              <option value="">Primary goal…</option>
              <option value="strength">Strength & Power</option>
              <option value="endurance">Endurance</option>
              <option value="recovery">Recovery</option>
              <option value="sleep">Sleep & Stress</option>
            </select>
            <select class="demo-select" id="dSeason">
              <option value="">Season status…</option>
              <option value="inseason">In-Season</option>
              <option value="offseason">Off-Season</option>
              <option value="preseason">Pre-Season</option>
            </select>
          </div>
          <button class="demo-run-btn" onclick="runDemo()">Generate Sample Stack →</button>
          <div class="demo-output" id="demoOut">
            <div class="demo-placeholder">Select sport, goal, and season to preview a sample stack.</div>
          </div>
          <div style="margin-top:.9rem;font-size:.7rem;color:var(--muted);line-height:1.6;">
            Preview only. Your real audit uses 15+ inputs for a fully personalized protocol.
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- SCHEDULE -->
<section id="schedule" class="section">
  <div class="section-inner">
    <div class="section-label reveal">Daily Protocol</div>
    <h2 class="section-title reveal">YOUR STACK<br>HAS A SCHEDULE</h2>
    <p class="section-sub reveal">IBEX doesn't just tell you what to take — it tells you exactly when. Every recommended item comes with a precise timing window.</p>
    <div class="sched-grid">
      <div class="sched-block reveal">
        <div class="sched-time">MORNING</div>
        <div class="sched-sub">Foundation window</div>
        <div class="sched-item"><span>Vitamin D3 + K2</span><span class="sched-dose">3000 IU</span></div>
        <div class="sched-item"><span>Sport Multivitamin</span><span class="sched-dose">1 cap</span></div>
        <div class="sched-item"><span>Omega-3 Fish Oil</span><span class="sched-dose">2g EPA/DHA</span></div>
        <div class="sched-item"><span>Probiotic</span><span class="sched-dose">30B CFU</span></div>
      </div>
      <div class="sched-block reveal">
        <div class="sched-time">TRAINING</div>
        <div class="sched-sub">Performance window</div>
        <div class="sched-item"><span>Creatine (post)</span><span class="sched-dose">5g</span></div>
        <div class="sched-item"><span>Beta-Alanine (pre)</span><span class="sched-dose">3.2g</span></div>
        <div class="sched-item"><span>Electrolytes (during)</span><span class="sched-dose">1 serving</span></div>
        <div class="sched-item"><span>Whey Protein (post)</span><span class="sched-dose">30g</span></div>
      </div>
      <div class="sched-block reveal">
        <div class="sched-time">EVENING</div>
        <div class="sched-sub">Recovery window</div>
        <div class="sched-item"><span>Magnesium Glycinate</span><span class="sched-dose">400mg</span></div>
        <div class="sched-item"><span>Tart Cherry Extract</span><span class="sched-dose">480mg</span></div>
        <div class="sched-item"><span>Ashwagandha KSM-66</span><span class="sched-dose">300mg</span></div>
        <div class="sched-item"><span>Zinc + Copper</span><span class="sched-dose">25mg / 2mg</span></div>
      </div>
    </div>
  </div>
</section>

<!-- PLANS -->
<section id="plans" class="section">
  <div class="section-inner">
    <div class="section-label reveal">Pricing</div>
    <h2 class="section-title reveal">PICK YOUR PLAN</h2>
    <div class="plan-grid">
      <div class="plan-card reveal">
        <div class="plan-tier">Basic Stack</div>
        <div class="plan-headline">"Foundations, done right."</div>
        <div class="plan-price">$100</div>
        <div class="plan-period">/ month · free shipping</div>
        <ul class="plan-features">
          <li>AI-personalized core stack (Creatine, Omega-3, Vit D, Magnesium, Electrolytes + more)</li>
          <li>Sport + position-specific selection</li>
          <li>AM / PM / Training schedule included</li>
          <li>NCAA compliant · NSF Certified preferred</li>
          <li>Ask IBEX chat grounded in your audit</li>
          <li>Cancel anytime</li>
        </ul>
        <a class="btn-plan" href="{AUDIT_URL}" target="_top">Start with Basic</a>
      </div>
      <div class="plan-card featured reveal">
        <div class="featured-tag">Most Popular</div>
        <div class="plan-tier">Performance Stack</div>
        <div class="plan-headline">"Optimization mode."</div>
        <div class="plan-price">$130</div>
        <div class="plan-period">/ month · free priority shipping</div>
        <ul class="plan-features">
          <li>Everything in Basic</li>
          <li>Expanded catalog: advanced recovery, sleep, gut, joint support</li>
          <li>In-season vs off-season auto-adjustments</li>
          <li>More conditional logic: training load, stress, soreness, travel</li>
          <li>Monthly performance check-in</li>
          <li>Priority shipping · Cancel anytime</li>
        </ul>
        <a class="btn-plan" href="{AUDIT_URL}" target="_top">Go Performance</a>
      </div>
    </div>
    <p style="font-size:.76rem;color:var(--muted);margin-top:1.4rem;text-align:center;">
      Run the free audit first — no card required. Your IBEX Audit ID matches your order at checkout.
    </p>
  </div>
</section>

<!-- PROOF -->
<section id="proof" class="section">
  <div class="section-inner">
    <div class="section-label reveal">Athletes</div>
    <h2 class="section-title reveal">BUILT FOR<br>PEOPLE LIKE YOU</h2>
    <div class="proof-grid">
      <div class="proof-card reveal">
        <div class="proof-stars">★★★★★</div>
        <p class="proof-quote">"Finally a system that understood that a D1 lineman and a D1 swimmer need completely different stacks. The AI got it on the first try."</p>
        <div class="proof-author">— Early Athlete</div>
        <div class="proof-sport">D1 Football · Offensive Line</div>
      </div>
      <div class="proof-card reveal">
        <div class="proof-stars">★★★★★</div>
        <p class="proof-quote">"The Ask IBEX chat is actually useful. I asked about away game travel and it gave me a real protocol based on my actual stack."</p>
        <div class="proof-author">— Early Athlete</div>
        <div class="proof-sport">D1 Swimming</div>
      </div>
      <div class="proof-card reveal">
        <div class="proof-stars">★★★★★</div>
        <p class="proof-quote">"Knowing every item is NCAA-compliant and third-party tested is huge. Last thing I need is something getting me flagged before conference."</p>
        <div class="proof-author">— Early Athlete</div>
        <div class="proof-sport">D1 Track & Field</div>
      </div>
    </div>
  </div>
</section>

<!-- FAQ -->
<section id="faq" class="section">
  <div class="section-inner">
    <div class="section-label reveal">Questions</div>
    <h2 class="section-title reveal">STRAIGHT<br>ANSWERS</h2>
    <div class="faq-wrap">

      <div class="faq-item reveal">
        <div class="faq-q" onclick="this.parentElement.classList.toggle('open')">Are these supplements NCAA compliant?<span class="faq-icon">+</span></div>
        <div class="faq-a">Every ingredient in the IBEX catalog is cross-referenced against the NCAA banned substance list. We prioritize NSF Certified for Sport products. That said, no supplement can be 100% guaranteed — rules change and contamination risk exists. Always confirm with your athletic department.</div>
      </div>

      <div class="faq-item reveal">
        <div class="faq-q" onclick="this.parentElement.classList.toggle('open')">How does the free audit work?<span class="faq-icon">+</span></div>
        <div class="faq-a">Click "Run My Free Audit" and answer ~15 questions about your sport, position, training load, sleep, stress, recovery, sensitivities, and goals. The AI builds your personalized stack in seconds — completely free, no credit card required. Your Audit ID is saved and used to match your order at checkout.</div>
      </div>

      <div class="faq-item reveal">
        <div class="faq-q" onclick="this.parentElement.classList.toggle('open')">What's the difference between Basic and Performance?<span class="faq-icon">+</span></div>
        <div class="faq-a">Basic covers the core foundational stack — creatine, vitamin D, omega-3, magnesium, electrolytes and similar essentials. Performance unlocks the expanded catalog with advanced recovery, sleep, gut health, and joint support items, plus automatic in-season vs off-season protocol adjustments and deeper conditional logic.</div>
      </div>

      <div class="faq-item reveal">
        <div class="faq-q" onclick="this.parentElement.classList.toggle('open')">How does shipping work?<span class="faq-icon">+</span></div>
        <div class="faq-a">Your 30-day supply ships in one package each month — no daily shipments. Basic ships standard (free). Performance ships priority (free). Orders process within 3–5 business days of subscribing.</div>
      </div>

      <div class="faq-item reveal">
        <div class="faq-q" onclick="this.parentElement.classList.toggle('open')">Can I adjust my stack between seasons?<span class="faq-icon">+</span></div>
        <div class="faq-a">Yes. Performance plan subscribers get automatic seasonal protocol adjustments. Basic subscribers can re-run the free audit anytime to refresh their stack.</div>
      </div>

      <div class="faq-item reveal">
        <div class="faq-q" onclick="this.parentElement.classList.toggle('open')">Is IBEX medical advice?<span class="faq-icon">+</span></div>
        <div class="faq-a">No. IBEX is not a medical provider and does not diagnose or treat conditions. If you have a medical condition, take medications, or experience symptoms, consult a qualified professional before starting any supplement protocol.</div>
      </div>

      <div class="faq-item reveal">
        <div class="faq-q" onclick="this.parentElement.classList.toggle('open')">Can I cancel anytime?<span class="faq-icon">+</span></div>
        <div class="faq-a">Yes — no contracts, no lock-ins. Cancel before your next billing date and you won't be charged again.</div>
      </div>

    </div>
  </div>
</section>

<!-- CTA -->
<section id="cta">
  <h2 class="cta-title reveal">STOP GUESSING.<br>START <em>PERFORMING.</em></h2>
  <p class="cta-sub reveal">Run your free AI audit. 3 minutes. No card required.</p>
  <a class="btn-gold reveal" href="{AUDIT_URL}" target="_top">Build My Free Stack →</a>
</section>

<!-- FOOTER -->
<footer>
  <span class="footer-wordmark">IBEX</span>
  <span class="footer-copy">© 2025 IBEX Supplements. Built for D1 athletes.</span>
  <div class="footer-links">
    <a href="#how">How It Works</a>
    <a href="#catalog">The Stack</a>
    <a href="#plans">Plans</a>
    <a href="#faq">FAQ</a>
  </div>
  <p class="footer-disclaimer">
    IBEX is not a medical provider. Supplement recommendations are not medical advice and do not diagnose or treat any condition. All ingredients are cross-referenced against the NCAA banned substance list, but no supplement can be guaranteed compliant for every league or test. Always confirm with your athletic department. Individual results may vary.
  </p>
</footer>

<script>
// ── SCROLL REVEAL ──
const reveals = document.querySelectorAll('.reveal');
const io = new IntersectionObserver((entries) => {{
  entries.forEach((e,i) => {{
    if(e.isIntersecting){{
      setTimeout(()=>e.target.classList.add('visible'), i*70);
      io.unobserve(e.target);
    }}
  }});
}},{{threshold:0.08}});
reveals.forEach(el=>io.observe(el));

// ── CATALOG FILTER ──
document.querySelectorAll('.cat-btn').forEach(btn=>{{
  btn.addEventListener('click',()=>{{
    document.querySelectorAll('.cat-btn').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    const cat=btn.dataset.cat;
    document.querySelectorAll('.supp-card').forEach(card=>{{
      card.classList.toggle('hidden', cat!=='all' && card.dataset.cat!==cat);
    }});
  }});
}});

// ── DEMO STACKS ──
const STACKS = {{
  football:   {{ strength:   [['Creatine Monohydrate','5g post-training'],['Whey Protein','35g post'],['Vitamin D3 + K2','5000 IU AM'],['Beta-Alanine','3.2g pre'],['Electrolytes','During 2-a-days']],
                 recovery:   [['Tart Cherry Extract','480mg AM+PM'],['Magnesium Glycinate','400mg PM'],['Omega-3 Fish Oil','3g w/ meals'],['Collagen + Vit C','15g 45min pre'],['Vitamin D3 + K2','5000 IU AM']],
                 sleep:      [['Magnesium Glycinate','400mg PM'],['Ashwagandha KSM-66','600mg PM'],['Melatonin','0.5mg before bed'],['Tart Cherry Extract','480mg PM'],['Vitamin D3 + K2','3000 IU AM']],
                 endurance:  [['Electrolytes','During training'],['Creatine','5g post'],['Beta-Alanine','3.2g pre'],['Omega-3','2g w/ meals'],['Vitamin D3','5000 IU AM']] }},
  swimming:   {{ endurance:  [['Beetroot Nitrate','400mg 2hrs pre'],['Beta-Alanine','3.2g pre'],['Electrolytes','During training'],['Omega-3','2g w/ meals'],['Vitamin D3 + K2','3000 IU AM']],
                 recovery:   [['Creatine','5g post'],['Tart Cherry','480mg AM+PM'],['Magnesium Glycinate','400mg PM'],['Omega-3','3g w/ meals'],['Vitamin D3','3000 IU AM']],
                 strength:   [['Creatine','5g post'],['Whey Protein','25g post'],['Beta-Alanine','3.2g pre'],['Omega-3','2g w/ meals'],['Vitamin D3','3000 IU AM']],
                 sleep:      [['Magnesium Glycinate','400mg PM'],['Ashwagandha','300mg PM'],['Melatonin','0.5mg before bed'],['Omega-3','2g w/ meals'],['Vitamin D3','3000 IU AM']] }},
  wrestling:  {{ strength:   [['Creatine','5g post'],['Beta-Alanine','3.2g pre'],['Whey Protein','30g post'],['Magnesium Glycinate','400mg PM'],['Vitamin D3','3000 IU AM']],
                 recovery:   [['Tart Cherry','480mg AM+PM'],['Magnesium Glycinate','400mg PM'],['Curcumin + Piperine','500mg w/ meals'],['Omega-3','2g w/ meals'],['Collagen + Vit C','15g 45min pre']],
                 sleep:      [['Magnesium Glycinate','400mg PM'],['Ashwagandha','600mg PM'],['Melatonin','0.5mg before bed'],['Tart Cherry','480mg PM'],['Vitamin D3','3000 IU AM']],
                 endurance:  [['Beta-Alanine','3.2g pre'],['Electrolytes','During training'],['Creatine','5g post'],['Omega-3','2g w/ meals'],['Vitamin D3','3000 IU AM']] }},
}};
const GENERIC = {{
  strength:  [['Creatine Monohydrate','5g post'],['Whey Protein','30g post'],['Beta-Alanine','3.2g pre'],['Omega-3','2g w/ meals'],['Vitamin D3 + K2','3000 IU AM']],
  recovery:  [['Tart Cherry Extract','480mg AM+PM'],['Magnesium Glycinate','400mg PM'],['Omega-3','3g w/ meals'],['Collagen + Vit C','15g 45min pre'],['Vitamin D3','3000 IU AM']],
  endurance: [['Beetroot Nitrate','400mg 2hrs pre'],['Beta-Alanine','3.2g pre'],['Electrolytes','During training'],['Omega-3','2g w/ meals'],['Creatine','5g post']],
  sleep:     [['Magnesium Glycinate','400mg PM'],['Ashwagandha KSM-66','300mg PM'],['Melatonin','0.5mg before bed'],['Tart Cherry','480mg PM'],['Vitamin D3','3000 IU AM']],
}};

function runDemo(){{
  const sport=document.getElementById('dSport').value;
  const goal=document.getElementById('dGoal').value;
  const season=document.getElementById('dSeason').value;
  if(!sport||!goal||!season){{
    document.getElementById('demoOut').innerHTML='<div class="demo-placeholder">Please select all three options.</div>';
    return;
  }}
  const items=(STACKS[sport]&&STACKS[sport][goal])||GENERIC[goal]||GENERIC.recovery;
  const seasonLabel={{inseason:'In-Season · recovery + maintenance',preseason:'Pre-Season · build + prime',offseason:'Off-Season · strength + foundation'}}[season]||season;
  const sportLabel=sport.charAt(0).toUpperCase()+sport.slice(1);
  const goalLabel=goal.charAt(0).toUpperCase()+goal.slice(1);
  let html=`<div class="demo-output-label">${{sportLabel}} · ${{goalLabel}} · ${{seasonLabel}}</div>`;
  items.forEach(([name,dose])=>html+=`<div class="demo-pill">${{name}}<span>${{dose}}</span></div>`);
  html+=`<div style="margin-top:.8rem;font-size:.7rem;color:var(--muted);">⚡ NCAA-compliant · Third-party tested options prioritized</div>`;
  document.getElementById('demoOut').innerHTML=html;
}}
</script>
</body>
</html>
"""

components.html(PAGE_HTML, height=9000, scrolling=False)
