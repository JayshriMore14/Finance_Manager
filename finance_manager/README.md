# 💰 AI Finance Manager

Upload a bank statement (CSV or PDF) and get:
- **Auto-categorized expenses** (rule-based keyword matching across 12 categories)
- **Monthly spending dashboard** (Plotly charts)
- **Unusual transaction detection** (statistical outlier detection per category)
- **Budget recommendations** (50/30/20 rule-based analysis)
- **Savings prediction** (linear trend projection for next 3 months)

100% local — no API keys, no external services. Includes a "Use sample data" toggle so you can try it instantly.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push to GitHub.
2. Connect the repo at https://share.streamlit.io
3. Set `app.py` as entry point and deploy.

## CSV format tips
The app auto-detects common column names (`date`, `description`/`narration`, `amount` or
separate `debit`/`credit` columns). If your bank's export uses different headers, rename the
columns before uploading, or edit `_guess_columns()` in `utils.py`.

## Tech Stack
Python · Streamlit · pandas · Plotly · pdfplumber · scikit-learn / NumPy (statistics)
