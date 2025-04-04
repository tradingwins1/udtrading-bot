import streamlit as st
from learn import get_stats

st.set_page_config(layout="wide")

st.title("📊 AI Trading Bot Dashboard")

stats = get_stats()
if stats is None:
    st.warning("No trades found in the database.")
else:
    st.subheader("Setup Performance Summary")
    st.dataframe(stats)

    st.subheader("Strategy Suggestions")
    for _, row in stats.iterrows():
        if row['win_rate_%'] < 50:
            st.markdown(f"- ❌ Consider avoiding **{row['setup_type']}** setups — win rate is {row['win_rate_%']:.2f}%")
        elif row['win_rate_%'] > 65:
            st.markdown(f"- ✅ Favor **{row['setup_type']}** setups — high win rate of {row['win_rate_%']:.2f}%")
