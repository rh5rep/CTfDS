# %%
import pandas as pd
import plotly.express as px
from Preprocessing import filtered_tweets_biden, filtered_tweets_trump, filtered_tweets
import pickle 
import random
# Load the filtered tweets from a pickle file
with open('/Users/rami/Documents/DTU/Semester 1/Computational Tools for Data Science/CTfDS/filtered_tweets_nlp.pkl', 'rb') as file:
    filtered_tweets = pickle.load(file)
# Add a column 'party' with random assignment of 'Democratic' or 'Republican'
# filtered_tweets['party'] = [random.choice(['Democratic', 'Republican']) for _ in range(len(filtered_tweets))]


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
# Create the choropleth map with discrete color steps
fig = px.choropleth(
    tweet_counts_per_state,         
    locations='state_code',         
    locationmode='USA-states',      
    color='democratic_proportion',  
    hover_name='state',             
    hover_data={
        'tweet_count': True,
        'democratic_proportion': ':.1f',
        'republican_proportion': ':.1f'
    },
    color_continuous_scale=px.colors.diverging.RdBu[::2],  
    labels={'democratic_proportion': 'Democratic Proportion'},
    # Define discrete color steps at 20% intervals
    range_color=[min_dem_prop, max_dem_prop],
    color_continuous_midpoint=mid_point
)

# Update layout with discrete color steps
fig.update_layout(
    title_text="Tweet Sentiment by State",
    title_x=0.5,
    geo=dict(
        scope='usa',
        projection_type='albers usa',
        showcoastlines=True,
        coastlinecolor="Black",
        showland=True,
        landcolor="white",
        lakecolor="white"
    ),
    coloraxis_colorbar=dict(
        tickmode='array',
        tickvals=[min_dem_prop + i*(max_dem_prop-min_dem_prop)/5 for i in range(6)],
        ticktext=[f"{i*20}%" for i in range(6)],
        title="Democratic Proportion"
    )
)

# Show the plot
fig.show()
print(file)
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

fig = px.choropleth(
    tweet_counts_per_state,
    locations='state_code',
    locationmode='USA-states',
    color='lead_size',
    hover_name='state',
    color_continuous_scale=color_scale,
    range_color=[-15, 15]
)

fig.update_layout(
    title_text="Tweet Sentiment Lead by State",
    geo=dict(scope='usa'),
    coloraxis_colorbar=dict(
        title="Lead Size",
        ticktext=['Trump >10%', 'Trump 5-10%', 'Trump 0-5%', 'Tie', 'Biden 0-5%', 'Biden 5-10%', 'Biden >10%'],
        tickvals=[-12, -7.5, -2.5, 0, 2.5, 7.5, 12],
        tickmode='array'
    )
)

# %%
# print(filtered_tweets)
# Get the amount of tweets per state
tweet_counts_per_state = filtered_tweets['state'].value_counts().reset_index()
tweet_counts_per_state.columns = ['state', 'tweet_count']

# Create a dictionary with state, sentiment, and tweet count
data_dict = {
    'state': tweet_counts_per_state['state'],
    'sentiment': filtered_tweets.groupby('state')['likely_party'].first().reindex(tweet_counts_per_state['state']).values,
    'tweet_count': tweet_counts_per_state['tweet_count']
}

print(data_dict)
# %%

# %%
import pandas as pd
import plotly.express as px
from Preprocessing import filtered_tweets_biden, filtered_tweets_trump, filtered_tweets
import pickle 
import random
# Load the filtered tweets from a pickle file
with open('/Users/rami/Documents/DTU/Semester 1/Computational Tools for Data Science/CTfDS/filtered_tweets_nlp.pkl', 'rb') as file:
    filtered_tweets = pickle.load(file)
# Add a column 'party' with random assignment of 'Democratic' or 'Republican'
# filtered_tweets['party'] = [random.choice(['Democratic', 'Republican']) for _ in range(len(filtered_tweets))]


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
sentiment_counts['democratic_proportion'] = sentiment_counts.get(1, 0) / sentiment_counts['total']
sentiment_counts['republican_proportion'] = sentiment_counts.get(2, 0) / sentiment_counts['total']

sentiment_counts['democratic_proportion'] = sentiment_counts.get(1, 0) / (sentiment_counts[1] + sentiment_counts[2])
sentiment_counts['republican_proportion'] = sentiment_counts.get(2, 0) / (sentiment_counts[1] + sentiment_counts[2])

print(sentiment_counts.columns)

total_democratic_tweets = sentiment_counts[1].sum()
total_republican_tweets = sentiment_counts[2].sum()

# Adjust republican proportions by the ratio of total tweets
tweet_ratio = total_democratic_tweets / total_republican_tweets
sentiment_counts['republican_proportion'] = sentiment_counts['republican_proportion'] * tweet_ratio

# Merge sentiment proportions with tweet counts
tweet_counts_per_state = tweet_counts_per_state.merge(sentiment_counts, left_on='state', right_index=True)

# Calculate the sum of republican and democrat tweets
total_democratic_tweets = sentiment_counts[1].sum()
total_republican_tweets = sentiment_counts[2].sum()

# Create a DataFrame for the comparison
comparison_df = pd.DataFrame({
    'Party': ['Democratic', 'Republican'],
    'Tweet Count': [total_democratic_tweets, total_republican_tweets]
})

# Create the bar plot
fig = px.bar(
    comparison_df,
    x='Party',
    y='Tweet Count',
    color='Party',
    title='Comparison of Total Tweets by Party',
    labels={'Tweet Count': 'Total Tweet Count'}
)

# Show the plot
fig.show()

# Create a DataFrame for the comparison per state
state_comparison_df = tweet_counts_per_state[['state', 1, 2]].copy()
state_comparison_df.columns = ['state', 'Democratic', 'Republican']

# Melt the DataFrame for easier plotting
state_comparison_melted = state_comparison_df.melt(id_vars='state', var_name='Party', value_name='Tweet Count')

# Create the bar plot per state
fig = px.bar(
    state_comparison_melted,
    x='state',
    y='Tweet Count',
    color='Party',
    title='Comparison of Tweets by Party per State',
    labels={'Tweet Count': 'Total Tweet Count'},
    barmode='group'
)

# Show the plot
fig.show()

# %%


# Calculate the sum of compound sentiment for Democrats and Republicans
democratic_sentiment_sum = abs(filtered_tweets[filtered_tweets['likely_party'] == 1]['compound']).sum()
republican_sentiment_sum = abs(filtered_tweets[filtered_tweets['likely_party'] == 2]['compound']).sum()

# Calculate the difference and normalize it
sentiment_difference = democratic_sentiment_sum - republican_sentiment_sum
normalized_sentiment_difference = sentiment_difference / (abs(democratic_sentiment_sum) + abs(republican_sentiment_sum))

print(f"Normalized Sentiment Difference: {normalized_sentiment_difference}")
# %%
print(filtered_tweets[['likely_party', 'state']].value_counts())
filtered_tweets['likely_party'].value_counts(normalize=True)

# %%
