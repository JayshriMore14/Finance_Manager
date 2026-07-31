"""AI Finance Manager
Upload a bank statement (CSV/PDF) -> auto-categorize -> dashboard -> anomaly detection ->
budget recommendations -> savings prediction. Fully local, no API key required.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    load_statement, monthly_summary, category_breakdown,
    detect_unusual_transactions, budget_recommendations, predict_savings,
)

st.set_page_config(page_title="AI Finance Manager", page_icon="💰", layout="wide")

st.title("💰 AI Finance Manager")
st.caption("Upload your bank statement to get automatic categorization, a spending dashboard, "
           "anomaly detection, budget recommendations, and a savings forecast. 100% local processing.")

uploaded_file = st.file_uploader("Upload bank statement (CSV or PDF)", type=["csv", "pdf"])

st.info("💡 No statement handy? Try the sample data below.", icon="💡")
use_sample = st.checkbox("Use sample data instead")

df = None
if use_sample:
    import numpy as np
    rng = pd.date_range("2025-01-01", periods=90, freq="D")
    sample_rows = []
    descriptions = [
        ("Salary Deposit", 3500, 1), ("Whole Foods Grocery", -85, 12), ("Rent Payment", -1200, 3),
        ("Netflix Subscription", -15.99, 3), ("Uber Ride", -22, 10), ("Starbucks Coffee", -6.5, 20),
        ("Electricity Bill", -75, 3), ("Amazon Shopping", -60, 8), ("Gym Membership", -40, 3),
        ("Doctor Visit", -150, 2), ("Unusual Big Purchase", -900, 1),
    ]
    np.random.seed(42)
    for desc, amt, count in descriptions:
        dates = np.random.choice(rng, size=count, replace=False)
        for d in dates:
            noise = np.random.uniform(0.85, 1.15)
            sample_rows.append({"date": d, "description": desc, "amount": round(amt * noise, 2)})
    raw = pd.DataFrame(sample_rows)
    from utils import categorize
    raw["category"] = raw["description"].apply(categorize)
    df = raw.sort_values("date").reset_index(drop=True)
elif uploaded_file is not None:
    try:
        df = load_statement(uploaded_file)
    except Exception as e:
        st.error(f"Could not parse statement: {e}")
        st.stop()

if df is not None and not df.empty:
    st.divider()

    income = df[df["amount"] > 0]["amount"].sum()
    expenses = df[df["amount"] < 0]["amount"].sum()
    net = income + expenses

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Income", f"${income:,.2f}")
    m2.metric("Total Expenses", f"${abs(expenses):,.2f}")
    m3.metric("Net Cashflow", f"${net:,.2f}", delta=f"{net:,.2f}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Dashboard", "🚨 Unusual Transactions", "💡 Budget Recommendations",
         "🔮 Savings Prediction", "📋 Raw Data"]
    )

    with tab1:
        breakdown = category_breakdown(df)
        c1, c2 = st.columns(2)
        with c1:
            if not breakdown.empty:
                fig = px.pie(breakdown, names="category", values="amount", title="Spending by Category")
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            monthly = monthly_summary(df)
            if not monthly.empty:
                fig2 = px.bar(monthly, x="month", y="net", title="Monthly Net Cashflow",
                              color="net", color_continuous_scale=["red", "green"])
                st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Category Totals")
        st.dataframe(breakdown, use_container_width=True)

    with tab2:
        unusual = detect_unusual_transactions(df)
        if unusual.empty:
            st.success("No unusual transactions detected.")
        else:
            st.warning(f"{len(unusual)} unusual transaction(s) detected (significantly larger than typical for their category):")
            st.dataframe(unusual[["date", "description", "category", "amount"]], use_container_width=True)

    with tab3:
        st.subheader("Personalized Budget Recommendations")
        for rec in budget_recommendations(df):
            st.write(f"- {rec}")

    with tab4:
        monthly = monthly_summary(df)
        if len(monthly) >= 2:
            preds = predict_savings(monthly, months_ahead=3)
            future_months = [f"Month +{i+1}" for i in range(len(preds))]
            pred_df = pd.DataFrame({"month": future_months, "predicted_net": preds})
            st.subheader("Projected Net Cashflow (next 3 months)")
            st.dataframe(pred_df, use_container_width=True)
            fig3 = px.line(
                pd.concat([
                    monthly.rename(columns={"net": "value"}).assign(type="Actual"),
                    pred_df.rename(columns={"predicted_net": "value"}).assign(type="Predicted"),
                ]),
                x="month", y="value", color="type", markers=True,
                title="Net Cashflow: Actual vs Predicted",
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Need at least 2 months of data to generate a savings prediction.")

    with tab5:
        st.dataframe(df, use_container_width=True)
        st.download_button("⬇️ Download categorized data (CSV)", df.to_csv(index=False),
                            file_name="categorized_transactions.csv")
else:
    st.info("Upload a bank statement or check 'Use sample data' above to get started.")
