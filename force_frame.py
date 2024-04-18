'''
Force-Frame Data Analysis

'''


import streamlit as st 
import pandas as pd
import numpy as np
import plotly.graph_objects as go


st.set_page_config(layout="wide")


col_pic, col_title = st.columns([0.2, 0.8])
with col_pic:
	st.image('rowing_canada.png', width = 100)
with col_title:
	st.title('Vald Force Frame Analysis')

uploaded_data = st.file_uploader('Upload Force Frame Excel Doc')

if uploaded_data is not None:
  df = pd.read_excel(uploaded_data, engine  = 'openpyxl')
  
else: 
  st.header('Please Upload Data')
  st.stop()


head_shots = pd.read_excel('headshots.xlsx')


athlete = st.sidebar.selectbox('Select Athlete', sorted(df['Name'].unique()), placeholder="Select Athlete for Analysis")

athlete_df = df[df['Name']==athlete]
group = athlete_df.iloc[-1][-1]
grouped_df = df[df['program'] == group]


latest_radar_list = []
last_radar_list = []
averages = []
for col in df.columns[:-1]:
	latest_radar_list.append(df[df['Name']==athlete][col].iloc[-1])
	
	try:
		last_radar_list.append(df[df['Name']==athlete][col].iloc[-2])
	except:
		last_radar_list = []



for col in df.columns[2:16]:
	df[col] = pd.to_numeric(df[col], errors='coerce') 
	averages.append(df[col].mean())


left_average = averages[0::2]
right_average = averages[1::2]

average_df = pd.DataFrame()
average_df['left'] = left_average
average_df['right'] = right_average

total_average = average_df.mean(axis=1)



fig = go.Figure()

fig.add_trace(go.Scatterpolar(
  r=latest_radar_list[2::2],
  theta=['Shoulder IR','Shoulder ER','Grip Strength', 'Hip Abduction', 'Hip Adduction', 'Knee Flexion', 'Knee Extension'],
  fill='toself', 
  name = 'Right', 
  line=dict(color='red')
))

fig.add_trace(go.Scatterpolar(
  r=latest_radar_list[3::2],
  theta=['Shoulder IR','Shoulder ER','Grip Strength', 'Hip Abduction', 'Hip Adduction', 'Knee Flexion', 'Knee Extension'],
  fill='toself', 
  name = 'Left', 
  line=dict(color='blue')
))

fig.add_trace(go.Scatterpolar(
  r=total_average,
  theta=['Shoulder IR','Shoulder ER','Grip Strength', 'Hip Abduction', 'Hip Adduction', 'Knee Flexion', 'Knee Extension'],
  fill='toself', 
  name = 'Team Average', 
  line=dict(color='grey')
))


fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 1000]
    )),
  showlegend=True
)


#Dashboarding

col1, col2,col3 = st.columns([0.4,0.4, 0.2])
with col1: 
	try:
		st.metric('Mass (kg)', latest_radar_list[-1], delta = round(latest_radar_list[-1]-last_radar_list[-1], 2))
	except:
		st.metric('Mass (kg)', latest_radar_list[-1])
	try:
		st.metric('Assessment Date', latest_radar_list[0].strftime('%Y-%m-%d'))
	except:
		st.metric('Assessment Date', latest_radar_list[0])
	
with col2: 
  st.header(athlete)
  st.subheader(group)
  st.metric('Star/Port/Scull', latest_radar_list[-2])


with col3: 
	st.image(list(head_shots[head_shots['Name'] == athlete]['Link']))
	
st.header('Graphical Analysis')
dcol1, dcol2 = st.columns([.6,.4])
with dcol1: 
	st.plotly_chart(fig, use_container_width=True)


comparisons = pd.DataFrame()
comparisons['Metric'] = ['Shoulder IR','Shoulder ER','Grip Strength', 'Hip Abduction', 'Hip Adduction', 'Knee Flexion', 'Knee Extension']
comparisons['Right'] = pd.to_numeric(latest_radar_list[3::2][:-1], errors='coerce')
comparisons['Left'] = pd.to_numeric(latest_radar_list[2::2][:-1], errors='coerce')
comparisons['R-L'] = comparisons['Right'] - comparisons['Left'] 
comparisons['Asym'] = comparisons['R-L']/((comparisons['Right'] + comparisons['Left'])) *100


asym_fig = go.Figure()

# Define colors based on the sign of 'Asym' values
colors = ['rgba(255, 0, 0, 0.6)' if x >= 0 else 'rgba(0, 0, 255, 0.6)' for x in comparisons['Asym']]
line_colors = ['rgba(255, 0, 0, 1.0)' if x >= 0 else 'rgba(0, 0, 255, 1.0)' for x in comparisons['Asym']]

asym_fig.add_trace(go.Bar(
    y=comparisons['Metric'],
    x=comparisons['Asym'],
    name='Asymmetries',
    orientation='h',
    marker=dict(
        color=colors,  # Apply the color conditionally
        line=dict(color=line_colors, width=3) 
    )
))

for i, (metric, value) in enumerate(zip(comparisons['Metric'], comparisons['Asym'])):
    asym_fig.add_annotation(
        x=value,  # Position of the annotation
        y=metric,  # Position on the y-axis
        text=str(round(value,2)),  # Text of the annotation
        showarrow=False,
        xanchor='left' if value < 0 else 'right',
        font=dict(
            family="Arial",
            size=12,
            color="black"
        ),
        xshift=-5 if value < 0 else 5  # Shift text to the left or right depending on value
    )
# Set the x-axis range
asym_fig.update_layout(
    xaxis=dict(
        range=[-15, 15],  # Setting symmetric limits for the x-axis
        showgrid=True,  # Enable vertical grid lines
        gridcolor='gray',  # Color of the grid lines
        gridwidth=1  # Width of the grid lines
    ),
    yaxis=dict(
        showgrid=True,  # Enable horizontal grid lines
        gridcolor='gray',  # Color of the grid lines
        gridwidth=1  # Width of the grid lines
    )
)

with dcol2: 
	st.plotly_chart(asym_fig, use_container_width = True)


st.header('Progression Table')

data = {
    'Latest': pd.to_numeric(latest_radar_list[2:16], errors = 'coerce'),
    'Previous': pd.to_numeric(last_radar_list[2:16], errors = 'coerce')
}
change_df = pd.DataFrame(data)


# Calculate percent change
change_df['Percent Change'] = ((change_df['Latest'] - change_df['Previous']) / change_df['Previous']) * 100

# Function to apply a color gradient based on the value
def gradient_color(value):
    """Apply a color gradient based on value: more intense colors for larger absolute percentages."""
    if value > 0:
        # Green, scaled by magnitude of the percent change (capped at 100 for practicality)
        hue = 120  # Green in HSL
        saturation = 100
        lightness = 100 - min(abs(value), 100)  # More intense for larger changes
    elif value < 0:
        # Red, scaled similarly
        hue = 0  # Red in HSL
        saturation = 100
        lightness = 100 - min(abs(value), 100)
    else:
        # Neutral color for zero
        hue = 0
        saturation = 0
        lightness = 100

    return f'background-color: hsl({hue}, {saturation}%, {lightness}%)'

change_df['Measurement']= df.columns[2:16]

# Apply the gradient color styling to the 'Percent Change' column

df_L = change_df[change_df['Measurement'].str.endswith('L')].copy()
df_R = change_df[change_df['Measurement'].str.endswith('R')].copy()

# Remove 'L' and 'R' from the Measurement names for easier comparison and merging
df_L['Measurement'] = df_L['Measurement'].str[:-2]
df_R['Measurement'] = df_R['Measurement'].str[:-2]

# Merge the DataFrames
df_merged = pd.merge(df_R, df_L, on='Measurement', suffixes=(' Right', ' Left'))
df_merged.set_index('Measurement', inplace=True)


styled_df = df_merged.style.applymap(gradient_color, subset=['Percent Change Right', 'Percent Change Left'])

# Display the styled DataFrame in Streamlit
st.dataframe(styled_df)

grouped_df['Date'] = pd.to_datetime(grouped_df['Date'])  # Convert the 'Date' column to datetime type

# Filter by the most recent date
most_recent_date = grouped_df['Date'].max()
grouped_df = grouped_df[grouped_df['Date']==most_recent_date]

# Convert columns to numeric if possible and create ranking columns
ranked_df = pd.DataFrame()
ranked_df['Name'] = grouped_df['Name']
for col in grouped_df.columns[2:]:
    grouped_df[col] = pd.to_numeric(grouped_df[col], errors='coerce')
    ranked_df[f'{col}'] = grouped_df[col].rank(method='average')


max_rank = len(ranked_df)



# Function to determine color based on rank
def get_color(rank, max_rank):
    # Normalize rank between 0 and 1
    norm_rank = rank / max_rank
    # Create color (red to blue scale)
    return f'rgb({255 * (1 - norm_rank)}, 0, {255 * norm_rank})'


rank_fig = go.Figure()
for col in ranked_df.columns[1:-3]:
  color = get_color(int(ranked_df[ranked_df['Name']==athlete][col]), max_rank) 

  rank_fig.add_trace(go.Bar(
      x=[col],
      y=ranked_df[ranked_df['Name']==athlete][col],
      text=[str(int(ranked_df[ranked_df['Name']==athlete][col]))],
      textposition='outside',
      marker_color=color
  ))
rank_fig.update_layout(title = 'Ranking Figure')
st.plotly_chart(rank_fig)














