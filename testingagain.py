#%%
import plotly.graph_objects as go
from baselinedata   import *
from plotnewnew import *

# Combine the two datasets into a single DataFrame
combined_df = state_votes.merge(tweet_counts_per_state[['state_code', 'democratic_proportion', 'republican_proportion', 'lead_size']], left_on='state_po', right_on='state_code', how='left')

# Calculate the difference in lead size between the two datasets
combined_df['lead_size_diff'] = combined_df['lead_size_x'] - combined_df['lead_size_y']

# Create the third choropleth map
fig = go.Figure(data=go.Choropleth(
    locations=combined_df['state_po'],
    locationmode='USA-states',
    z=combined_df['lead_size_diff'],
    colorscale=color_scale,
    colorbar=dict(
        title="Lead Size Difference",
        ticktext=['Trump >10%', 'Trump 5-10%', 'Trump 0-5%', 'Tie', 'Biden 0-5%', 'Biden 5-10%', 'Biden >10%'],
        tickvals=[-12, -7.5, -2.5, 0, 2.5, 7.5, 12],
        tickmode='array'
    )
))

fig.update_layout(
    title_text="Difference in Lead by State",
    geo=dict(scope='usa')
)

fig.show()
