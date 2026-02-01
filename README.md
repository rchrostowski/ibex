# IBEX Streamlit MVP

Streamlit web app that:
- collects an on-site questionnaire
- runs AI to select a personalized supplement system from a 100-product universe
- displays recommendations + AM/PM/training schedule
- links to Stripe Checkout (Payment Links)

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="YOUR_KEY"
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub
2. Deploy on Streamlit Community Cloud
3. Add Secrets:

```toml
OPENAI_API_KEY="..."
OPENAI_MODEL="gpt-4.1-mini"
STRIPE_BASIC_LINK="https://buy.stripe.com/..."
STRIPE_PERF_LINK="https://buy.stripe.com/..."
```

## Data
- `data/products.csv` (100 products, exact required columns)
- `data/exclusions.csv`
