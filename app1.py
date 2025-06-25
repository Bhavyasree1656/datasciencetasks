import streamlit as st
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary
import matplotlib.pyplot as plt

# --- Sample Influencer Data ---
influencers = ['Alice', 'Bob', 'Cara', 'Dave', 'Eva']
cpm = [10, 20, 15, 25, 12]  # Cost per 1000 impressions
reach_k = [50, 100, 60, 90, 55]  # in 1000s
engagement_rate = [0.08, 0.05, 0.07, 0.04, 0.09]
overlap_index = [0.1, 0.2, 0.15, 0.25, 0.05]

# --- Streamlit UI ---
st.title("🎯 Dynamic Influencer Engagement Optimizer")

budget = st.slider("Marketing Budget ($)", min_value=500, max_value=3000, value=1500, step=100)

# --- LP Model ---
model = LpProblem("Influencer_Selection", LpMaximize)
x = LpVariable.dicts("Select", influencers, 0, 1, LpBinary)

model += lpSum([
    reach_k[i] * 1000 * engagement_rate[i] * (1 - overlap_index[i]) * x[influencers[i]]
    for i in range(len(influencers))
])

model += lpSum([
    cpm[i] * reach_k[i] * x[influencers[i]]
    for i in range(len(influencers))
]) <= budget

model.solve()

# --- Collect Results ---
selected = []
costs = []
engagements = []

for i in range(len(influencers)):
    if x[influencers[i]].value() == 1:
        selected.append(influencers[i])
        cost = cpm[i] * reach_k[i]
        engagement = reach_k[i] * 1000 * engagement_rate[i] * (1 - overlap_index[i])
        costs.append(cost)
        engagements.append(engagement)

# --- Display Results ---
st.subheader("✅ Selected Influencers")
st.write(selected if selected else "No influencers selected within budget.")

st.metric("💰 Total Cost", f"${sum(costs):,.2f}")
st.metric("🔥 Total Engagement Score", f"{sum(engagements):,.0f}")

# --- Plot ---
if selected:
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.bar(selected, costs, color='skyblue', label='Cost ($)', width=0.4)
    ax1.set_ylabel('Cost ($)', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')

    ax2 = ax1.twinx()
    ax2.plot(selected, engagements, color='green', marker='o', label='Engagement', linewidth=2)
    ax2.set_ylabel('Engagement Score', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    plt.title("Cost vs Engagement for Selected Influencers")
    fig.tight_layout()
    st.pyplot(fig)
