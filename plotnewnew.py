#%%
import pandas as pd
import plotly.express as px
from Preprocessing import filtered_tweets_biden, filtered_tweets_trump, filtered_tweets
import pickle 
import random
# Load the filtered tweets from a pickle file
with open('/Users/rami/Documents/DTU/Semester 1/Computational Tools for Data Science/CTfDS/filtered_tweets_nlp.pkl', 'rb') as file:
    filtered_tweets = pickle.load(file)

# Define a mapping for US states to Plotly's choropleth map
state_abbreviations = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
    'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
    'District of Columbia': 'DC'
}


# Calculate tweet counts and sentiment proportions per state
tweet_counts_per_state = filtered_tweets['state'].value_counts().reset_index()
tweet_counts_per_state.columns = ['state', 'tweet_count']
tweet_counts_per_state['state_code'] = tweet_counts_per_state['state'].map(state_abbreviations)

sentiment_counts = filtered_tweets.groupby(['state', 'likely_party']).size().unstack(fill_value=0)
sentiment_counts['total'] = sentiment_counts.sum(axis=1)
print(sentiment_counts)
# Calculate initial proportions
sentiment_counts['democratic_proportion'] = sentiment_counts.get(1, 0) / (sentiment_counts[1] + sentiment_counts[2])
sentiment_counts['republican_proportion'] = sentiment_counts.get(2, 0) / (sentiment_counts[1] + sentiment_counts[2])

total_democratic_tweets = sentiment_counts[1].sum()
total_republican_tweets = sentiment_counts[2].sum()

# Adjust republican proportions by the ratio of total tweets
tweet_ratio = total_democratic_tweets / total_republican_tweets
sentiment_counts['republican_proportion'] = sentiment_counts['republican_proportion'] * tweet_ratio

# Normalize to ensure proportions sum to 1
total_proportion = sentiment_counts['democratic_proportion'] + sentiment_counts['republican_proportion']
sentiment_counts['democratic_proportion'] = sentiment_counts['democratic_proportion'] / total_proportion
sentiment_counts['republican_proportion'] = sentiment_counts['republican_proportion'] / total_proportion

sentiment_counts['democratic_proportion'] = sentiment_counts['democratic_proportion'] * 100
sentiment_counts['republican_proportion'] = sentiment_counts['republican_proportion'] * 100
# Merge sentiment proportions with tweet counts
tweet_counts_per_state = tweet_counts_per_state.merge(sentiment_counts, left_on='state', right_index=True)



min_dem_prop = tweet_counts_per_state['democratic_proportion'].min()
max_dem_prop = tweet_counts_per_state['democratic_proportion'].max()
mid_point = (min_dem_prop + max_dem_prop) / 2
# Calculate lead size (difference between democratic and republican proportion)
tweet_counts_per_state['lead_size'] = tweet_counts_per_state['democratic_proportion'] - 50

# Create custom color scale for discrete ranges
color_scale = [
    [0, 'rgb(165,0,38)'],    # Deep Red (Trump >10%)
    [0.2, 'rgb(215,48,39)'],  # Red (Trump 5-10%)
    [0.4, 'rgb(244,109,67)'], # Light Red (Trump 0-5%)
    [0.5, 'rgb(255,255,255)'], # White (Tie)
    [0.6, 'rgb(116,169,207)'], # Light Blue (Biden 0-5%)
    [0.8, 'rgb(54,144,192)'],  # Blue (Biden 5-10%)
    [1, 'rgb(5,112,176)']     # Deep Blue (Biden >10%)
]

sentiment_fig = px.choropleth(
    tweet_counts_per_state,
    locations='state_code',
    locationmode='USA-states',
    color='lead_size',
    hover_name='state',
    color_continuous_scale=color_scale,
    range_color=[-15, 15]
)

sentiment_fig.update_layout(
    title_text="Tweet Sentiment Lead by State",
    geo=dict(scope='usa'),
    coloraxis_colorbar=dict(
        title="Lead Size",
        ticktext=['Trump >10%', 'Trump 5-10%', 'Trump 0-5%', 'Tie', 'Biden 0-5%', 'Biden 5-10%', 'Biden >10%'],
        tickvals=[-12, -7.5, -2.5, 0, 2.5, 7.5, 12],
        tickmode='array'
    )
)

sentiment_fig.show()
# %%
