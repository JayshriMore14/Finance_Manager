"""Core logic for AI Finance Manager. Fully local — rule-based categorization + statistics."""
import io
import re

import numpy as np
import pandas as pd
import pdfplumber

CATEGORY_KEYWORDS = {
    "Groceries": ["grocery", "supermarket", "walmart", "kroger", "trader joe", "whole foods",
                  "aldi", "safeway", "bigbasket", "dmart", "reliance fresh"],
    "Rent/Housing": ["rent", "landlord", "mortgage", "housing society", "maintenance charge"],
    "Utilities": ["electricity", "water bill", "gas bill", "utility", "broadband", "internet bill",
                  "wifi", "power supply", "cellphone", "mobile bill", "recharge"],
    "Dining": ["restaurant", "cafe", "coffee", "starbucks", "mcdonald", "zomato", "swiggy",
               "doordash", "ubereats", "pizza", "dining", "food delivery"],
    "Transport": ["uber", "ola", "lyft", "taxi", "fuel", "petrol", "diesel", "metro", "bus fare",
                  "parking", "toll", "railway", "irctc", "flight", "airlines"],
    "Shopping": ["amazon", "flipkart", "myntra", "mall", "shopping", "ikea", "target", "best buy",
                 "clothing", "electronics store"],
    "Entertainment": ["netflix", "spotify", "prime video", "hotstar", "movie", "cinema", "concert",
                       "gaming", "steam", "playstation", "xbox"],
    "Healthcare": ["pharmacy", "hospital", "clinic", "doctor", "medical", "medicine", "health insurance",
                   "dental", "apollo", "cvs pharmacy"],
    "Subscriptions": ["subscription", "membership", "prime membership", "icloud", "google one",
                       "adobe", "microsoft 365", "gym membership"],
    "Education": ["tuition", "school fee", "college", "course", "udemy", "coursera", "book store"],
    "Income": ["salary", "payroll", "deposit", "interest credit", "refund", "cashback", "dividend",
               "bonus", "credited"],
    "Transfers": ["transfer to", "transfer from", "upi", "neft", "imps", "wire transfer", "venmo", "zelle"],
}


def categorize(description):
    desc = str(description).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in desc:
                return category
    return "Other"


def _guess_columns(df):
    """Try to map arbitrary bank CSV headers to date / description / amount."""
    cols = {c.lower().strip(): c for c in df.columns}

    def find(*candidates):
        for cand in candidates:
            for lower, orig in cols.items():
                if cand in lower:
                    return orig
        return None

    date_col = find("date", "posted", "transaction date")
    desc_col = find("description", "narration", "details", "particulars", "memo", "payee")
    amount_col = find("amount", "value")
    debit_col = find("debit", "withdrawal")
    credit_col = find("credit", "deposit")

    return date_col, desc_col, amount_col, debit_col, credit_col


def load_statement(uploaded_file):
    """Load CSV or PDF bank statement into a normalized DataFrame with
    columns: date, description, amount (signed: negative=expense, positive=income), category."""
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".pdf"):
        rows = []
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if table:
                    header, *body = table
                    rows.extend(body)
                    header_cache = header
        if not rows:
            raise ValueError("Could not detect a table in the PDF. Try a CSV export instead.")
        df = pd.DataFrame(rows, columns=header_cache)
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or PDF.")

    date_col, desc_col, amount_col, debit_col, credit_col = _guess_columns(df)

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    out["description"] = df[desc_col].astype(str) if desc_col else "Unknown"

    if amount_col:
        out["amount"] = pd.to_numeric(
            df[amount_col].astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce"
        )
    elif debit_col or credit_col:
        debit = pd.to_numeric(
            df[debit_col].astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce"
        ).fillna(0) if debit_col else 0
        credit = pd.to_numeric(
            df[credit_col].astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce"
        ).fillna(0) if credit_col else 0
        out["amount"] = credit - debit
    else:
        raise ValueError("Could not detect an amount column in the statement.")

    out = out.dropna(subset=["amount"])
    out["category"] = out["description"].apply(categorize)
    out = out.sort_values("date", na_position="last").reset_index(drop=True)
    return out


def monthly_summary(df):
    d = df.dropna(subset=["date"]).copy()
    d["month"] = d["date"].dt.to_period("M").astype(str)
    summary = d.groupby("month")["amount"].sum().reset_index()
    summary.columns = ["month", "net"]
    return summary


def category_breakdown(df):
    expenses = df[df["amount"] < 0].copy()
    expenses["amount"] = expenses["amount"].abs()
    return expenses.groupby("category")["amount"].sum().sort_values(ascending=False).reset_index()


def detect_unusual_transactions(df, z_thresh=2.5):
    """Flag transactions unusually large relative to their category's typical spend."""
    expenses = df[df["amount"] < 0].copy()
    expenses["amount_abs"] = expenses["amount"].abs()
    flagged = []
    for category, group in expenses.groupby("category"):
        if len(group) < 3:
            if len(group) > 0 and group["amount_abs"].max() > 3 * group["amount_abs"].mean():
                flagged.append(group.loc[[group["amount_abs"].idxmax()]])
            continue
        mean, std = group["amount_abs"].mean(), group["amount_abs"].std()
        if std == 0 or np.isnan(std):
            continue
        z = (group["amount_abs"] - mean) / std
        unusual = group[z > z_thresh]
        if not unusual.empty:
            flagged.append(unusual)
    if flagged:
        return pd.concat(flagged).sort_values("amount_abs", ascending=False)
    return pd.DataFrame(columns=list(expenses.columns))


def budget_recommendations(df):
    """Simple 50/30/20 style rule-based recommendations."""
    income = df[df["amount"] > 0]["amount"].sum()
    breakdown = category_breakdown(df)
    total_expense = breakdown["amount"].sum()

    needs_categories = {"Rent/Housing", "Utilities", "Groceries", "Healthcare", "Transport"}
    wants_categories = {"Dining", "Entertainment", "Shopping", "Subscriptions"}

    needs = breakdown[breakdown["category"].isin(needs_categories)]["amount"].sum()
    wants = breakdown[breakdown["category"].isin(wants_categories)]["amount"].sum()
    other = total_expense - needs - wants

    recs = []
    if income > 0:
        needs_pct = needs / income * 100
        wants_pct = wants / income * 100
        savings_pct = (income - total_expense) / income * 100

        recs.append(f"Needs spending is {needs_pct:.1f}% of income (target: ~50%).")
        if needs_pct > 55:
            recs.append("⚠️ Essential spending (rent, groceries, utilities) is above the recommended 50% — "
                         "look for ways to reduce fixed costs.")
        recs.append(f"Wants spending is {wants_pct:.1f}% of income (target: ~30%).")
        if wants_pct > 35:
            recs.append("⚠️ Discretionary spending (dining, shopping, entertainment) is high — "
                         "consider a monthly cap on these categories.")
        recs.append(f"Current savings rate is {savings_pct:.1f}% of income (target: ~20%).")
        if savings_pct < 15:
            recs.append("⚠️ Savings rate is below the recommended 20% — consider automating a fixed "
                         "transfer to savings right after income arrives.")
        else:
            recs.append("✅ Savings rate looks healthy!")
    else:
        recs.append("No income transactions detected — upload a statement covering income to get personalized recommendations.")

    top_category = breakdown.iloc[0] if not breakdown.empty else None
    if top_category is not None:
        recs.append(f"Your largest expense category is **{top_category['category']}** "
                     f"(₹{top_category['amount']:.0f} / ${top_category['amount']:.0f}). "
                     "Review recent transactions there for potential savings.")

    return recs


def predict_savings(monthly_df, months_ahead=3):
    """Simple linear regression on net monthly cashflow to project future months."""
    if len(monthly_df) < 2:
        return None
    x = np.arange(len(monthly_df))
    y = monthly_df["net"].values
    coeffs = np.polyfit(x, y, 1)
    future_x = np.arange(len(monthly_df), len(monthly_df) + months_ahead)
    predictions = np.polyval(coeffs, future_x)
    return predictions
