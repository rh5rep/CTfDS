# %%
import pandas as pd
import plotly.express as px
# import os
import pickle

# # Load election data
# cwd = os.getcwd()
# df = pd.read_csv(cwd + '/data/PRESIDENT_precinct_general.csv')
# df = pd.read_csv('PRESIDENT_precinct_general.csv')

with open('/Users/rami/Documents/DTU/Semester 1/Computational Tools for Data Science/CTfDS/data/filtered_president_precinct_general.pkl', 'rb') as file:
    df = pickle.load(file)
    # print(df)
# Filter the DataFrame
# Group the filtered DataFrame by state and candidate, and sum the votes
state_votes = df.groupby(['state_po', 'candidate'])['votes'].sum().unstack(fill_value=0)

# Calculate the total votes per state
state_votes['total_votes'] = state_votes.sum(axis=1)

# print(state_votes['candidate'])

# Calculate the percentage of votes for Biden
state_votes['biden_pct'] = state_votes['JOSEPH R BIDEN'] / state_votes['total_votes'] * 100
state_votes['democratic_proportion'] = state_votes['JOSEPH R BIDEN'] / state_votes['total_votes']

# Calculate lead size
state_votes['lead_size'] = (state_votes['JOSEPH R BIDEN'] - state_votes['DONALD J TRUMP']) / state_votes['total_votes'] * 100
# Reset the index to make state_po a column
state_votes = state_votes.reset_index()

x = (state_votes['DONALD J TRUMP'].sum())
y = (state_votes['JOSEPH R BIDEN'].sum())

print(abs(x-y)/(x+y)*100)
#%%
# print(state_votes.head())
color_scale = [
    [0, 'rgb(165,0,38)'],    # Deep Red (Trump >10%)
    [0.2, 'rgb(215,48,39)'],  # Red (Trump 5-10%)
    [0.4, 'rgb(244,109,67)'], # Light Red (Trump 0-5%)
    [0.5, 'rgb(255,255,255)'], # White (Tie)
    [0.6, 'rgb(116,169,207)'], # Light Blue (Biden 0-5%)
    [0.8, 'rgb(54,144,192)'],  # Blue (Biden 5-10%)
    [1, 'rgb(5,112,176)']     # Deep Blue (Biden >10%)
]


baseline_fig = px.choropleth(
    state_votes,
    locations='state_po',
    locationmode='USA-states',
    color='lead_size',
    hover_name='state_po',
    color_continuous_scale=color_scale,
    range_color=[-15, 15]
)

baseline_fig.update_layout(
    title_text="Lead by State via Harvard Polling Data",
    geo=dict(scope='usa'),
    coloraxis_colorbar=dict(
        title="Lead Size",
        ticktext=['Trump >10%', 'Trump 5-10%', 'Trump 0-5%', 'Tie', 'Biden 0-5%', 'Biden 5-10%', 'Biden >10%'],
        tickvals=[-12, -7.5, -2.5, 0, 2.5, 7.5, 12],
        tickmode='array'
    )
)

# Show the plot
baseline_fig.show()

# %%
