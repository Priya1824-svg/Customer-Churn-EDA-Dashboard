import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# ── Load data ────────────────────────────────────────────────
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
    df = pd.read_csv(url)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df.drop('customerID', axis=1, inplace=True)
    df['TenureBucket'] = pd.cut(df['tenure'],
                                 bins=[0, 12, 24, 48, 72],
                                 labels=['0-12 mo', '13-24 mo', '25-48 mo', '49-72 mo'])
    return df

df = load_data()

# ── Sidebar filters ───────────────────────────────────────────
st.sidebar.title("Filters")
contract_filter = st.sidebar.multiselect(
    "Contract type",
    options=df['Contract'].unique(),
    default=list(df['Contract'].unique())
)
internet_filter = st.sidebar.multiselect(
    "Internet service",
    options=df['InternetService'].unique(),
    default=list(df['InternetService'].unique())
)
senior_filter = st.sidebar.radio(
    "Customer segment",
    options=["All", "Senior Citizens", "Non-Senior"]
)

filtered = df[
    df['Contract'].isin(contract_filter) &
    df['InternetService'].isin(internet_filter)
]
if senior_filter == "Senior Citizens":
    filtered = filtered[filtered['SeniorCitizen'] == 1]
elif senior_filter == "Non-Senior":
    filtered = filtered[filtered['SeniorCitizen'] == 0]

# ── Header ────────────────────────────────────────────────────
st.title("📊 Customer Churn Analysis")
st.caption("Telco Customer Dataset · Exploratory Data Analysis")

# ── KPI cards ─────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Customers", f"{len(filtered):,}")
k2.metric("Churned Customers", f"{filtered['Churn'].sum():,}")
k3.metric("Churn Rate", f"{filtered['Churn'].mean()*100:.1f}%")
k4.metric("Avg Monthly Charges", f"${filtered['MonthlyCharges'].mean():.2f}")

st.divider()

# ── Row 1: Contract + Tenure ──────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn rate by contract type")
    contract_churn = filtered.groupby('Contract')['Churn'].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(contract_churn.index, contract_churn.values,
                  color=['#D85A30', '#EF9F27', '#1D9E75'], width=0.5)
    ax.set_ylabel('Churn rate (%)')
    ax.set_ylim(0, 55)
    for bar, val in zip(bars, contract_churn.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    st.pyplot(fig)
    st.caption("Month-to-month customers churn the most — offering annual plans reduces churn significantly.")

with col2:
    st.subheader("Churn rate by tenure group")
    tenure_churn = filtered.groupby('TenureBucket', observed=True)['Churn'].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(tenure_churn.index, tenure_churn.values,
                  color=['#D85A30', '#EF9F27', '#5DCAA5', '#1D9E75'], width=0.5)
    ax.set_ylabel('Churn rate (%)')
    ax.set_ylim(0, 65)
    for bar, val in zip(bars, tenure_churn.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    st.pyplot(fig)
    st.caption("New customers (first 12 months) are at highest risk — onboarding programs can help.")

st.divider()

# ── Row 2: Charges + Internet ──────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("Monthly charges distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    filtered[filtered['Churn']==0]['MonthlyCharges'].plot(
        kind='kde', ax=ax, label='Stayed', color='#1D9E75', linewidth=2)
    filtered[filtered['Churn']==1]['MonthlyCharges'].plot(
        kind='kde', ax=ax, label='Churned', color='#D85A30', linewidth=2)
    ax.set_xlabel('Monthly charges ($)')
    ax.set_ylabel('Density')
    ax.legend()
    st.pyplot(fig)
    st.caption("Churned customers tend to pay higher monthly charges — price sensitivity is a key factor.")

with col4:
    st.subheader("Churn by internet service")
    internet_churn = filtered.groupby('InternetService')['Churn'].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(internet_churn.index, internet_churn.values,
                  color=['#D85A30', '#EF9F27', '#1D9E75'], width=0.5)
    ax.set_ylabel('Churn rate (%)')
    ax.set_ylim(0, 55)
    for bar, val in zip(bars, internet_churn.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    st.pyplot(fig)
    st.caption("Fiber optic users churn more despite paying the most — service quality may be an issue.")

st.divider()

# ── Row 3: Payment method + Tech support ───────────────────────
col5, col6 = st.columns(2)

with col5:
    st.subheader("Churn by payment method")
    pay_churn = filtered.groupby('PaymentMethod')['Churn'].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(pay_churn.index, pay_churn.values,
            color=['#D85A30', '#EF9F27', '#5DCAA5', '#1D9E75'])
    ax.set_xlabel('Churn rate (%)')
    for i, val in enumerate(pay_churn.values):
        ax.text(val + 0.3, i, f'{val:.1f}%', va='center', fontsize=10)
    st.pyplot(fig)
    st.caption("Electronic check users churn most — auto-pay adoption reduces churn.")

with col6:
    st.subheader("Churn by tech support")
    tech_churn = filtered.groupby('TechSupport')['Churn'].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(tech_churn.index, tech_churn.values,
                  color=['#D85A30', '#1D9E75', '#888780'], width=0.5)
    ax.set_ylabel('Churn rate (%)')
    ax.set_ylim(0, 55)
    for bar, val in zip(bars, tech_churn.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    st.pyplot(fig)
    st.caption("Customers without tech support churn at 2.5x the rate of supported customers.")

st.divider()

# ── Raw data explorer ─────────────────────────────────────────
with st.expander("View raw data"):
    st.dataframe(filtered.head(100), use_container_width=True)
    st.caption(f"Showing 100 of {len(filtered):,} rows")

st.caption("Built with Python · Pandas · Seaborn · Streamlit")
