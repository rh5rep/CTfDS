#%%
import plotly.graph_objects as go
from baselinedata   import *
from plot_sentiment import *

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


# %%
import plotly.graph_objects as go
import numpy as np
from baselinedata import *
from plot_sentiment import *

# Merge datasets
combined_df = state_votes.merge(
    tweet_counts_per_state[['state_code', 'democratic_proportion', 'republican_proportion', 'lead_size']], 
    left_on='state_po', 
    right_on='state_code', 
    how='left'
)

# Calculate differences
combined_df['lead_size_diff'] = combined_df['lead_size_x'] - combined_df['lead_size_y']

# Define state centers
state_centers = {
    'AL': [32.806671, -86.791130],
    'AK': [61.370716, -152.404419],
    'AZ': [33.4484, -112.0740],
    'AR': [34.969704, -92.373123],
    'CA': [36.116203, -119.681564],
    'CO': [39.059811, -105.311104],
    'CT': [41.597782, -72.755371],
    'DE': [39.318523, -75.507141],
    'FL': [27.766279, -81.686783],
    'GA': [33.040619, -83.643074],
    'HI': [21.094318, -157.498337],
    'ID': [44.240459, -114.478828],
    'IL': [40.349457, -88.986137],
    'IN': [39.849426, -86.258278],
    'IA': [42.011539, -93.210526],
    'KS': [38.526600, -96.726486],
    'KY': [37.668140, -84.670067],
    'LA': [31.169546, -91.867805],
    'ME': [44.693947, -69.381927],
    'MD': [39.063946, -76.802101],
    'MA': [42.230171, -71.530106],
    'MI': [43.326618, -84.536095],
    'MN': [45.694454, -93.900192],
    'MS': [32.741646, -89.678696],
    'MO': [38.456085, -92.288368],
    'MT': [46.921925, -110.454353],
    'NE': [41.125370, -98.268082],
    'NV': [38.313515, -117.055374],
    'NH': [43.452492, -71.563896],
    'NJ': [40.298904, -74.521011],
    'NM': [34.840515, -106.248482],
    'NY': [42.165726, -74.948051],
    'NC': [35.630066, -79.806419],
    'ND': [47.528912, -99.784012],
    'OH': [40.388783, -82.764915],
    'OK': [35.565342, -96.928917],
    'OR': [44.572021, -122.070938],
    'PA': [40.590752, -77.209755],
    'RI': [41.680893, -71.511780],
    'SC': [33.856892, -80.945007],
    'SD': [44.299782, -99.438828],
    'TN': [35.747845, -86.692345],
    'TX': [31.054487, -97.563461],
    'UT': [40.150032, -111.862434],
    'VT': [44.045876, -72.710686],
    'VA': [37.769337, -78.169968],
    'WA': [47.400902, -121.490494],
    'WV': [38.491226, -80.954456],
    'WI': [44.268543, -89.616508],
    'WY': [42.755966, -107.302490]
}

# Create base map
fig = go.Figure()

# Add choropleth base layer
fig.add_trace(go.Choropleth(
    locations=combined_df['state_po'],
    locationmode='USA-states',
    colorscale='RdBu',
    colorbar=dict(
        title="Lead Size Difference",
        ticktext=['More Republican', 'No Change', 'More Democratic'],
        tickvals=[-10, 0, 10],
        tickmode='array'
    )
))

# Add scattergeo layer for arrows
for idx, row in combined_df.iterrows():
    state = row['state_po']
    if state in state_centers:
        diff = row['lead_size_diff']
        if diff < 1 and diff >= 0: # Make sure arrow is visible  
            diff = 1
        elif diff > -1 and diff <= 0:
            diff = -1
        if abs(diff) > 0:
            # Calculate arrow properties
            arrow_color = 'blue' if diff > 0 else 'red'
            arrow_size = min(abs(diff) * 0.5, 10)  # Scale arrow size
            
            # Add arrow as scatter point
            fig.add_trace(go.Scattergeo(
                lon=[state_centers[state][1]],
                lat=[state_centers[state][0]],
                mode='markers+text',
                text='▲' if diff > 0 else '▼', 
                textposition='middle center',
                hovertext=f"{state}: {diff:.1f}%",
                textfont=dict(
                    size=arrow_size * 2,
                    color=arrow_color
                ),
                marker=dict(
                    size=0,  # Hide the marker, show only text
                    color=arrow_color
                ),
                showlegend=False
            ))

fig.update_layout(
    title_text="Prediction vs. Actual Results Difference<br>(▲=More Democratic than predicted, ▼=More Republican)",
    geo=dict(
        scope='usa',
        projection_type='albers usa',  # Use Albers USA projection
        showland=True,
        landcolor='rgb(240, 240, 240)',
        showlakes=True,
        lakecolor='rgb(255, 255, 255)',
        subunitcolor='rgb(217, 217, 217)',
        countrycolor='rgb(217, 217, 217)',
    ),
    height=400,
    width=600
)

fig.show()
# %%
