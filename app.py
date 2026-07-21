import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("NFL Combine Analytics")
st.write("Analyzing which combine metrics predict career success for CB, S, and WR")

career_totals = pd.read_csv('merged_combine_stats.csv')
career_totals = career_totals.groupby(['player_name_x', 'pos', 'forty', 'vertical', 'broad_jump', 'cone', 'shuttle']).agg(
    career_receiving_yards=('receiving_yards', 'sum'),
    career_receptions=('receptions', 'sum'),
    career_receiving_tds=('receiving_tds', 'sum'),
    career_def_interceptions=('def_interceptions', 'sum'),
    career_def_pass_defended=('def_pass_defended', 'sum'),
    career_def_tackles_solo=('def_tackles_solo', 'sum'),
    seasons_played=('season', 'count')
).reset_index()

position = st.selectbox("Select Position", ['WR', 'CB', 'S'])
pos_data = career_totals[career_totals['pos'] == position]

st.write(f"Players analyzed: {len(pos_data)}")

metrics = ['forty', 'vertical', 'broad_jump', 'cone', 'shuttle']

if position == 'WR':
    outcomes = ['career_receiving_yards', 'career_receptions', 'career_receiving_tds']
else:
    outcomes = ['career_def_interceptions', 'career_def_pass_defended', 'career_def_tackles_solo']

corr_data = pd.DataFrame(
    {m: [pos_data[m].corr(pos_data[o]) for o in outcomes] for m in metrics},
    index=outcomes
)

st.subheader(f"{position} Combine Metric Correlations with Career Outcomes")
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, fmt='.3f', ax=ax)
st.pyplot(fig)
plt.close()
st.subheader(f"Explore Individual Metrics")

metric = st.selectbox("Select Combine Metric", metrics)

if position == 'WR':
    outcome = 'career_receiving_yards'
    outcome_label = 'Career Receiving Yards'
else:
    outcome = 'career_def_pass_defended'
    outcome_label = 'Career Pass Deflections'

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.scatter(pos_data[metric], pos_data[outcome], alpha=0.5)
ax2.set_xlabel(metric)
ax2.set_ylabel(outcome_label)
ax2.set_title(f"{position}: {metric} vs {outcome_label}")
corr = pos_data[metric].corr(pos_data[outcome])
ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax2.transAxes, fontsize=12, verticalalignment='top')
st.pyplot(fig2)
plt.close()
st.subheader("Key Findings")

if position == 'WR':
    st.write("""
    - **Broad jump** is the strongest predictor of receiving production (r = 0.157 for TDs)
    - **40-yard dash** shows near-zero correlation with career receiving yards (r = -0.062)
    - All five combine metrics together explain only **3.3% of career receiving yards** (R² = 0.033)
    - Conclusion: Raw speed is a poor indicator of WR success — explosive power matters more
    """)
elif position == 'CB':
    st.write("""
    - **Vertical jump** is the strongest predictor of pass deflections (r = 0.120)
    - **40-yard dash** shows a negative correlation with career production
    - Combine metrics together produce a **negative R² (-0.085)** — worse than guessing the average
    - Conclusion: Combine testing does not reliably identify successful cornerbacks
    """)
elif position == 'S':
    st.write("""
    - **Vertical jump** is the strongest predictor of pass deflections (r = 0.134)
    - **Shuttle run** shows the strongest negative correlation (-0.271 for tackles)
    - Combine metrics together produce a **negative R² (-0.220)** — significantly worse than guessing
    - Conclusion: For safeties, combine metrics may actively mislead evaluators
    """)

st.subheader("Overall Conclusion")
st.write("""
Across all three positions, NFL combine metrics explain very little about career success.
The 40-yard dash — the most watched event at the combine — is a weak or negative predictor 
for cornerbacks, safeties, and wide receivers alike. Jumping metrics (vertical and broad jump) 
show the most consistent positive signal, suggesting explosive power matters more than raw speed.
""")