# %%
import pandas as pd
import plotly.express as px
from Preprocessing import filtered_tweets_biden, filtered_tweets_trump, filtered_tweets
import pickle 
import random
#%%
# Load the filtered tweets from a pickle file
with open('filtered_tweets_nlp.pkl', 'rb') as file:
    filtered_tweets = pickle.load(file)
# Add a column 'party' with random assignment of 'Democratic' or 'Republican'
# filtered_tweets['party'] = [random.choice(['Democratic', 'Republican']) for _ in range(len(filtered_tweets))]
#%%

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



#%%
##open new csv as pandas
df_pres = pd.read_csv('data/1976-2020-president.csv')
# filter for only democrate and republican
df_pres = df_pres[df_pres['party_detailed'].isin(['DEMOCRAT', 'REPUBLICAN'])]
# filter for only president
df_pres = df_pres[df_pres['office'] == 'US PRESIDENT']
# print first 5 rows
# print(df_pres.head())
## remove 2020 year
df_pres = df_pres[df_pres['year'] != 2020]

## add ratio comments to the data
df_pres['ratio'] =0
for row in df_pres.iterrows():
    year = row[1]['year']
    total_votes = df_pres[(df_pres['year'] == year) & (df_pres['state'] == row[1]["state"])]['candidatevotes'].sum()
    df_pres.loc[row[0], 'ratio'] = row[1]['candidatevotes'] / total_votes
# df_pres['ratio'] = df_pres['candidatevotes'] / df_pres['totalvotes']

# average per state per party
df_pres = df_pres.groupby([ "state",'state_po', 'party_detailed'])['ratio'].mean().reset_index()
# print the texas stats
print(df_pres[df_pres['state'] == 'TEXAS'])
# print first 5 rows   
#%%
for row in sentiment_counts.iterrows():
    state = (row[0]).upper()
    dem_ratio = df_pres[(df_pres['state'] == state) & 
                        (df_pres['party_detailed'] == 'DEMOCRAT')]['ratio'].values[0]  * row[1]['democratic_proportion']
    rep_ratio = df_pres[(df_pres['state'] == state) & 
                        (df_pres['party_detailed'] == 'REPUBLICAN')]['ratio'].values[0] * row[1]['republican_proportion']
    
    if state=="TEXAS":
        print("==========")
        print(sentiment_counts.loc['Texas'])

        print(sentiment_counts.loc[row[0], 'democratic_proportion'], "dem")
        print(sentiment_counts.loc[row[0], 'republican_proportion'], "rep")

        print(dem_ratio, "dem ratio from election")
        print(rep_ratio, "rep ratio from election")

        print("post dem" , sentiment_counts.loc[row[0], 'democratic_proportion'] * dem_ratio)
        print("post rep", sentiment_counts.loc[row[0], 'republican_proportion']* rep_ratio)
        print("==========")
    sentiment_counts.loc[row[0], 'democratic_proportion'] = dem_ratio /(dem_ratio+rep_ratio) 
    sentiment_counts.loc[row[0], 'republican_proportion'] = rep_ratio /(dem_ratio+rep_ratio) 
print(sentiment_counts.loc['Texas'])
print("==========")

# %%
# Merge sentiment proportions with tweet counts
tweet_counts_per_state = tweet_counts_per_state.merge(sentiment_counts, left_on='state', right_index=True)
custom_color_scale = [
    (0.0, "red"),
    (0.30, "lightcoral"),
    (0.45, "lightpink"),
    (0.5, "white"),
    (0.55, "lightblue"),
    (0.80, "lightskyblue"),
    (1.0, "blue")
]
# Create the choropleth map
fig = px.choropleth(
    tweet_counts_per_state,         # DataFrame with tweet data
    locations='state_code',         # State codes
    locationmode='USA-states',      # Use US states mode
    color='democratic_proportion',  # Color by democratic proportion
    hover_name='state',             # Hover over the state to show name
    hover_data={
        'tweet_count': True,
        'democratic_proportion': ':.3f',
        'republican_proportion': ':.3f'
        # 1: 'Democratic',
        # 2: 'Republican'
    },
    color_continuous_scale=px.colors.diverging.RdBu[::1],  # Red to blue gradient
    # color_continuous_scale=custom_color_scale,  # Red to blue gradient
    color_continuous_midpoint = 0.5,
    labels={'democratic_proportion': 'Democratic Proportion'}
)

# Update layout for better visualization
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
        title="Democratic Proportion",
        tickvals=[0, 0.5, 1],
        ticktext=["Republican", "Neutral", "Democratic"]
    )
)

# Show the plot
fig.show()


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

