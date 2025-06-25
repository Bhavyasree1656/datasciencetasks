from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary

# Sample Data
influencers = ['Alice', 'Bob', 'Cara', 'Dave', 'Eva']
cpm = [10, 20, 15, 25, 12]  # Cost per 1000 impressions
reach_k = [50, 100, 60, 90, 55]  # in 1000s (so total cost = CPM * reach_k)
engagement_rate = [0.08, 0.05, 0.07, 0.04, 0.09]  # 8%, 5%, etc.
overlap_index = [0.1, 0.2, 0.15, 0.25, 0.05]  # Redundancy in niche

# Marketing budget in dollars
budget = 1500

# Setup the LP problem
model = LpProblem("Dynamic_Influencer_Engagement", LpMaximize)

# Decision variables: 0 or 1 (select or not)
x = LpVariable.dicts("Select", influencers, 0, 1, LpBinary)

# Objective: Maximize adjusted effective engagement
model += lpSum([
    reach_k[i] * 1000 * engagement_rate[i] * (1 - overlap_index[i]) * x[influencers[i]]
    for i in range(len(influencers))
]), "Total_Effective_Engagement"

# Budget constraint
model += lpSum([
    cpm[i] * reach_k[i] * x[influencers[i]]
    for i in range(len(influencers))
]) <= budget, "Total_Budget"

# Solve the model
model.solve()

# Output
print("Selected Influencers:")
for i in influencers:
    if x[i].value() == 1:
        print(f" - {i}")
print("\nTotal Cost:", sum(cpm[i] * reach_k[i] * x[influencers[i]].value() for i in range(len(influencers))))
print("Total Engagement Score:", model.objective.value())
