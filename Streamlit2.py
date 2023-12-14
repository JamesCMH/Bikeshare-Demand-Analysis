import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import joblib
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_squared_error
import json
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")


@st.cache_data
def load_data():
    data = pd.read_csv('bike-sharing-hourly.csv')
    return data
data = load_data()

def home_page():
    st.title(':bike: Capital Bikeshare Demand Analysis and Forecasting')
    st.markdown('##')
    st.markdown("""
<div style='text-align: center; font-size: 40px;'> 
Welcome to our Bike-Sharing Analysis & Prediction Tool!

<div style='text-align: center; font-size: 20px;'> 
<div style='margin-top: 20px;'>
This application is designed to assist the Washington D.C. administration in comprehensively analyzing and understanding the usage patterns of the city's bike-sharing service. Our goal is to provide valuable insights that will aid in optimizing the provisioning of bikes, enhancing public transportation services, and reducing operational costs.
</div>
""", unsafe_allow_html=True)
   
    st.markdown("""
    <div style='margin-top: 30px;'>
    <div style='text-align: center; font-size: 20px;'> 
    Our interactive platform is designed for the head of transportation services and other stakeholders in the local government. It offers user-friendly navigation and comprehensive data visualization to support informed decision-making and strategic planning for the city's bike-sharing service.
    <div style='margin-top: 20px;'>
    Embark on a journey of data-driven insights and proactive transportation management with our Bike-Sharing Analysis & Prediction Tool!
    <div style='margin-top: 40px;'>
    <div style='text-align: center; font-size: 30px;'>
    Key App Features:
    <div style='margin-top: 20px;'>
    <div style='text-align: center; font-size: 20px;'> 
    In-Depth Usage Analysis <br>
    Business Recommendations <br>
    Predictive Modeling<br>
    <div style='margin-top: 20px;'>
    <div style='margin-top: 20px;'>
    
<div style='margin-top: 20px;'>
<div style='text-align: center; font-size: 20px;'>

</div>
""", unsafe_allow_html=True)
    
    multi = """<div style='margin-top: 20px;'>
    <div style='text-align: center; font-size: 20px;'>
    Team Members:<br>
    Sarah Awad<br>
    Alejandro Danus<br>
    Lucas Gonzalez Gago<br>
    Mary Ann Mousa<br>
    Sebastian Perez<br>
    Yu Shi
    """
    def display_details():
        st.markdown(multi, unsafe_allow_html=True)
    with st.expander("Group 2 Team Members"):
        display_details()   


def eda_page(data):
    
    # Main panel
    st.title('📊Exploratory Data Analysis')
    st.markdown('##')
    
    st.header('Bike Sharing Data')

    col1, col2 = st.columns(2)

    with col1:  
        st.dataframe(data.head(19), height=700)        
    with col2:  
        st.write("""- `instant`: record index
- `dteday` : date
- `season` : season 1: winter, season 2: spring, season 3: summer, season 4: fall
- `yr` : year (0: 2011, 1:2012)
- `mnth` : month ( 1 to 12)
- `hr` : hour (0 to 23)
- `holiday` : weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
- `weekday` : day of the week
- `workingday` : if day is neither weekend nor holiday is 1, otherwise is 0.
+ `weathersit` : 
	- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
	- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
	- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
	- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
- `temp` : Normalized temperature in Celsius. The values are divided to 41 (max)
- `atemp`: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
- `hum`: Normalized humidity. The values are divided to 100 (max)
- `windspeed`: Normalized wind speed. The values are divided to 67 (max)
- `casual`: count of casual users
- `registered`: count of registered users
- `cnt`: count of total rental bikes including both casual and registered""", use_container_width=True)
    
    def streamlit_stat_summary_app(data):
   
        st.write('Statistical Summary:')
        st.write(data.describe())

        if 'show_analysis' not in st.session_state:
            st.session_state['show_analysis'] = False

        def show_analysis_callback():
            st.session_state['show_analysis'] = True

        def hide_analysis_callback():
            st.session_state['show_analysis'] = False

        
        show_button = st.button('Show Analysis', on_click=show_analysis_callback)
        hide_button = st.button('Hide Analysis', on_click=hide_analysis_callback)

        if st.session_state['show_analysis']:
            st.write("""
Initial analysis:

The average season is approximately 2.5, suggesting the data might be uniformly distributed across the four seasons.
The yr has a mean of 0.5026, which suggests that the data is approximately evenly split between two years (likely coded as 0 and 1).
The hr column has a mean of around 11.55, indicating that the data is spread across all hours of the day.
Standard Deviation (std): This tells us about the variability of each variable.

Variables like holiday, workingday, and weathersit have lower standard deviations, indicating less variability, which makes sense for categorical variables that have a limited range of values.

For temp, the 25th percentile is 0.34, the median (50th percentile) is 0.5, and the 75th percentile is 0.66, which suggests a relatively symmetric distribution of temperature values around the median.
Possible Categorical Variables: The variables season, yr, holiday, weekday, workingday, and weathersit appear to be categorical, given their lower ranges and standard deviations.

Continuous Variables: The temp variable appears to be continuous, with a range from 0.02 to 1, which could be a normalized representation of temperature.""")

    streamlit_stat_summary_app(data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st. markdown("""<div style='text-align: left; font-size: 20px;'> Number of Days </div>
""", unsafe_allow_html=True)
        st.subheader('731')
        st.markdown("""<div style='margin-top: 30px;'>""", unsafe_allow_html=True)
        
    with col2:
        st. markdown("""<div style='text-align: left; font-size: 20px;'> Total Rentals </div>
""", unsafe_allow_html=True)
        st.subheader("3,292,679")
        st.markdown("""<div style='margin-top: 30px;'>""", unsafe_allow_html=True)

    with col3:
        st. markdown("""<div style='text-align: left; font-size: 20px;'> Total Casula Users </div>
""", unsafe_allow_html=True)
        st.subheader("620,017")
        st.markdown("""<div style='margin-top: 30px;'>""", unsafe_allow_html=True)
    

    col1, col2, col3 = st.columns(3)
    with col1:
        st. markdown("""<div style='text-align: left; font-size: 20px;'> Total Registered Users </div>
""", unsafe_allow_html=True)
        st.subheader("2,672,662")
        st.markdown("""<div style='margin-top: 30px;'>""", unsafe_allow_html=True)
    with col2:
        st. markdown("""<div style='text-align: left; font-size: 20px;'> Count of Missing Records </div>
""", unsafe_allow_html=True)
        st.subheader("76")
        st.markdown("""<div style='margin-top: 30px;'>""", unsafe_allow_html=True)
    with col3:
        st. markdown("""<div style='text-align: left; font-size: 20px;'> Count with < 10 hours </div>
""", unsafe_allow_html=True)
        st.subheader("2")
        st.markdown("""<div style='margin-top: 30px;'>""", unsafe_allow_html=True)
        
    col1, col2, col3, = st.columns(3)
    with col1:
        st. markdown("""<div style='text-align: left; font-size: 20px;'> Number of Null Values </div>
""", unsafe_allow_html=True)
        st.subheader('0')
    with col2:
        st. markdown("""<div style='text-align: left; font-size: 20px;'> Number of Rows </div>
""", unsafe_allow_html=True)
        st.subheader("17,379")
    with col3:
        st.subheader("")
        
    
    st.title("Feature Engineering")
    st.header("Created Variables:")
    def display_details():
        st.subheader("Hours of Daylight")
        st.write("Based on the hour of the day (`hr`). It is a binary feature where a value of 1 indicates daylight hours (6 AM to 7 PM), and 0 indicates nighttime. This feature can help the model discern patterns that differ between daytime and nighttime")
            
        st.subheader("High and Low Humidity")
        st.write("The high_humidity feature is a binary variable indicating whether the humidity is high (≥ 75%). This transforms a continuous variable (humidity) into a categorical one, which can sometimes make patterns more evident for certain models.")
            
        st.subheader("Temperature Squared")
        st.write("By squaring the temperature, the model can potentially capture non-linear relationships between temperature and the target variable. ")
            
        st.subheader("Rolling Average")
        st.write("Computes a moving average of the cnt variable over a window of three time periods. This feature smooths out short-term fluctuations and can highlight longer-term trends, which might be more predictive for the task at hand.")
        st.markdown("""
        - Note: we did not include the rolling average column in our model because that would cause data leakage with our data split.""")
        
        st.subheader("Week Work")
        st.write("This feature is a combination of two conditions: the day of the week and whether it is during daylight hours. It essentially flags work hours on weekdays, under the assumption that patterns during these times might be different from those during weekends or nights.")
        
        st.subheader("Cyclical Encoding")
        st.write("Cyclical encoding is used to transform cyclical features. The reason for using cyclical encoding is to maintain the continuity of the cycle, ensuring that the model recognizes the proximity of the start and end points in the cycle. The encoding is done using sine and cosine transformations, which map the cyclical feature onto a circle.")
    with st.expander("Click here to expand/collapse for more details"):

        display_details()
        

    def plot_boxplots(df, columns):
        num_rows = len(columns) // 3 + (len(columns) % 3 > 0)
        fig = make_subplots(rows=num_rows, cols=3, subplot_titles=[f'Box Plot of {col}' for col in columns])

        for i, col in enumerate(columns):
            row = i // 3 + 1
            col_pos = i % 3 + 1
            fig.add_trace(
                go.Box(y=df[col], name=col),
                row=row, col=col_pos
            )
    
        fig.update_layout(showlegend=False, title_text="Box Plots of Numerical Columns")
        fig.update_layout(height=300*num_rows, width=900)
    
        return fig

# Funcion for plotting histograms
    def plot_histogram_with_kde(df, column, bins=30):
    # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Histogram
        fig.add_trace(
            go.Histogram(
                x=df[column], 
                nbinsx=bins,
                name=f'{column} Histogram',
                histnorm='probability density',
            ),
            secondary_y=False,
        )

    # KDE
        kde_data = df[column]
        kde = np.histogram(kde_data, bins=bins, density=True)
        kde_x = (kde[1][:-1] + kde[1][1:]) / 2 
        kde_y = kde[0]

        fig.add_trace(
            go.Scatter(
                x=kde_x, 
                y=kde_y, 
                mode='lines',
                name=f'{column} KDE',
                line=dict(shape='spline') 
            ),
            secondary_y=True,
        )

    # Add figure title
        fig.update_layout(
            title_text=f"Histogram with KDE of {column}"
        )

    # Set x-axis title
        fig.update_xaxes(title_text=column)

    # Set y-axes titles
        fig.update_yaxes(title_text="<b>Count</b> Density", secondary_y=False)
        fig.update_yaxes(title_text="<b>KDE</b> Density", secondary_y=True)

        return fig

    text_analysis = {
    'temp': "The temperature data is bimodal, with two peaks suggesting two different modes or most common values in the data set. The distribution is relatively symmetrical around the modes. The KDE line follows the shape of the histogram closely, indicating a good fit to the data points.",
    'atemp': "The feels like temperature also exhibits a bimodal distribution, but with sharper peaks compared to the actual temperature. This could suggest that people's perception of temperature clusters around specific 'feels like' conditions more than the actual temperature readings",
    'hum': "The humidity variable displays a right-skewed distribution with a gradual increase in frequency towards higher values, and then a sharp decline after peaking. The multiple peaks in the KDE suggest that there may be several common humidity levels where the frequency of occurrences is higher.",
    'windspeed': "Wind speed shows a highly right-skewed distribution with a peak close to zero, indicating that lower wind speeds are far more common than higher ones. The KDE line shows a long tail towards the higher values, which means that while higher wind speeds are less frequent, there is a wide range of values that occur.",
    'casual': "The distribution of casual counts is highly right-skewed, with most of the data clustered close to the lower end and a long tail extending towards higher values. This suggests that there are generally fewer casual users or events, but occasional spikes in numbers.",
    'registered': "Similar to the casual counts, the registered counts also show a right-skewed distribution. However, the peak is not as close to zero as the casual one, indicating that higher counts of registered users or events are more common than casual ones. The tail extends less far than the casual counts, suggesting less extreme values.",
        'cnt': "We can see that the cnt variable is also skewed and this aligns with the fact that cnt is a combination of the casual and registered users."
}




    numerical_columns = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']


    # Histogram with KDE Section
    st.header('Normality Distribution KDE Plot')
    variable = st.selectbox('Select Variable', options=numerical_columns)
    fig = plot_histogram_with_kde(data, variable)
    st.plotly_chart(fig, use_container_width=True)

    if variable in text_analysis:
        st.write(text_analysis[variable])
    else:
        st.write("No text analysis available for the selected variable.")
        
        

    # Box Plot Section
    st.header('Outlier Detection Box Plot')
    selected_columns = st.multiselect('Select columns to plot', numerical_columns, default = numerical_columns)
    
    if selected_columns:
        fig = plot_boxplots(data, selected_columns)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("")
        

    st.title(':chart_with_upwards_trend: Time Series Chart')

# Radio button to select the data to plot
    option = st.radio(
        'Which data do you want to plot?',
        ('Casual', 'Registered', 'Both')
    )
    

    import plotly.express as px

    def create_time_series_chart(df, option):
    # Group by 'dteday' and sum the counts for each day
        daily_totals = df.groupby('dteday').sum().reset_index()

        if option == 'Registered':
            y_data = ['registered']
            title = 'Time Series of Registered Bike Counts'
        elif option == 'Casual':
            y_data = ['casual']
            title = 'Time Series of Casual Bike Counts'
        else:  # 'Both' option
            y_data = ['registered', 'casual']
            title = 'Time Series of Casual and Registered Bike Counts'

    # Create the time series plot
        fig = px.line(daily_totals, x='dteday', y=y_data, title=title)
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Count')
        fig.update_layout(legend_title='User Type')

        return fig

    def display_analysis(data, option):
        fig = create_time_series_chart(data, option)
        st.plotly_chart(fig, use_container_width=True)
    
        if 'show_analysis' not in st.session_state:
            st.session_state['show_analysis'] = False

        col1, col2 = st.columns(2)
        with col1:
            if st.button('Show Analysis',key='show_button'):
                st.session_state['show_analysis'] = True
        with col2:
            if st.button('Hide Analysis' , key='hide_button'):
                st.session_state['show_analysis'] = False

        if st.session_state['show_analysis']:
            st.write("""
**Seasonality:** Both casual and registered user counts show a clear seasonal pattern, with peaks that likely correspond to warmer months and troughs that correspond to colder months. This suggests that weather conditions have a strong influence on bike usage.

**Trend:** There appears to be an upward trend in the counts of registered users over the period shown, indicating growing popularity or increased adoption of the bike service among registered users.

**User Type Disparity:** The count of registered users is consistently higher than that of casual users throughout the entire time span. This could imply a strong base of dedicated users who use the service regularly.

**Volatility:** The casual user count appears to be more volatile than the registered user count, with sharper peaks and troughs. This might imply that casual use is more spontaneous and influenced by immediate conditions such as weather, events, or tourism trends.

**Data Resolution:** The data is recorded with a high frequency, possibly daily, given the number of data points and the timeframe covered.
""")
    display_analysis(data, option)

    weather_labels = {
        1: 'Clear',
        2: 'Mist',
        3: 'Light Precipitation',
        4: 'Heavy Precipitation'
        }
    data['weather_situation'] = data['weathersit'].map(weather_labels)

# Function to create bar chart
    def create_weather_chart(df, selected_variables):
        average_counts = df.groupby('weather_situation')[selected_variables].mean().reset_index()
        fig = px.bar(average_counts, x='weather_situation', y=selected_variables,
                 title='Average Bike Rentals by Weather Situation',
                 labels={var: f'Average {var.capitalize()} Counts' for var in selected_variables},
                 barmode='group')
        fig.update_xaxes(title_text='Weather Situation')
        fig.update_yaxes(title_text='Average Count of Users')
        return fig

    def create_season_chart(df, selected_variables):
        season_labels = {
        1: 'Spring',
        2: 'Summer',
        3: 'Fall',
        4: 'Winter'
        }
        df['season_name'] = df['season'].map(season_labels)
        average_counts = df.groupby('season_name')[selected_variables].mean().reset_index()

        fig = px.bar(average_counts, x='season_name', y=selected_variables,
                 title='Average Bike Rentals by Season',
                 labels={var: f'Average {var.capitalize()} Counts' for var in selected_variables},
                 barmode='group')
        fig.update_xaxes(title_text='Season')
        fig.update_yaxes(title_text='Average Count of Users')
        return fig
# Multiselect for user to choose the variables
    selected_variables = st.multiselect('Select user types', options=['casual', 'registered'], default=['casual', 'registered'])

    col1, col2 = st.columns(2)
    with col1:
        st.header('Bike Rentals by Weather Situation')

# Display the bar chart for weather
        if selected_variables:
            fig_weather = create_weather_chart(data, selected_variables)
            st.plotly_chart(fig_weather, use_container_width=True)
        else:
            st.write("Please select at least one user type to display the weather chart.")

    with col2:
        st.header('Bike Rentals by Season')
# Display the bar chart for season
        if selected_variables:
            fig_season = create_season_chart(data, selected_variables)
            st.plotly_chart(fig_season, use_container_width=True)
        else:
            st.write("Please select at least one user type to display the season chart.")
        
    st.write("""
Both casual and registered users rent bikes most frequently in clear weather.
Registered users consistently rent more bikes than casual users across all weather conditions.
Bike rentals decrease for both user types as weather conditions worsen, with the sharpest decline during heavy precipitation.
Registered users' rental patterns suggest they may use bikes for regular commutes, as their numbers decrease less than those of casual users when the weather gets worse.
""")
    
       
    def create_heatmap(df, x, y, z):
        fig = go.Figure(data=go.Heatmap(
            x=df[x],
            y=df[y],
            z=df[z],
            colorscale='Viridis',
            colorbar_title=f'Average {z.capitalize()} Counts',
        ))
        fig.update_layout(
            title=f'Average {z.capitalize()} Bike Counts per {x.title()} and {y.title()}',
            xaxis=dict(title=x.title()),
            yaxis=dict(title=y.title()),
        )
        return fig

    st.title('Bike Rental Heatmaps 🔥')

# Sidebar for user inputs
    x_axis = st.selectbox('Select X-axis', options=['hr', 'mnth'], index=0)
    y_axis = 'weekday'  # Fixed for this example, but you can make it selectable as well
    count_type = st.selectbox('Select Count Type', options=['casual', 'registered'], index=0)

    average_counts = data.groupby([x_axis, y_axis])[['casual', 'registered']].mean().reset_index()

# Display the heatmap
    fig = create_heatmap(average_counts, x_axis, y_axis, count_type)
    st.plotly_chart(fig, use_container_width=True)

    def load_data():
        data['dteday'] = pd.to_datetime(data['dteday'])  # Ensure the date column is in datetime format
        data['month'] = data['dteday'].dt.month
        data['hour'] = data['hr']  # Assuming there is an 'hr' column for hours
        data['weekday'] = data['dteday'].dt.dayofweek  # 0 is Monday, 6 is Sunday
        return data

    data = load_data()

    st.title('Bike Rental Trends Analysis')

    user_type = st.radio(
        "Select the user type for analysis:",
        ('Casual', 'Registered', 'Both')
    )
    
    col1, col2 = st.columns(2)
    
    with col1:


# Monthly Trends
        st.subheader('Monthly Bike Rental Trends')
        if user_type == 'Both':
            monthly_data = data.groupby('month')[['casual', 'registered']].sum().reset_index()
            fig_monthly = px.bar(monthly_data, x='month', y=['casual', 'registered'], title='Total Bike Rentals per Month', barmode='group')
        else:
            monthly_data = data.groupby('month')[user_type.lower()].sum().reset_index()
            fig_monthly = px.bar(monthly_data, x='month', y=user_type.lower(), title=f'Total {user_type} Bike Rentals per Month')

        st.plotly_chart(fig_monthly, use_container_width=True)
  
    
    with col2:
        st.subheader('Weekly Bike Rental Trends')
        data['week'] = data['dteday'].dt.isocalendar().week

    # Weekly trends
        if user_type == 'Both':
        # Aggregating data for both casual and registered
            weekly_data = data.groupby('week').agg({'casual':'mean', 'registered':'mean'}).reset_index()
            fig_weekly = px.bar(weekly_data, x='week', y=['casual', 'registered'], 
                             title='Average Bike Rentals per Week', 
                             labels={'value':'Average Rentals', 'variable':'User Type'}, barmode='group')
        else:
        # Aggregating data for either casual or registered
            weekly_data = data.groupby('week')[user_type.lower()].mean().reset_index()
            fig_weekly = px.bar(weekly_data, x='week', y=user_type.lower(), 
                             title=f'Average {user_type} Bike Rentals per Week',
                             labels={user_type.lower():'Average Rentals'})

        st.plotly_chart(fig_weekly, use_container_width=True)
    
    
        text_analysis_hourly = {
    'both': """Peak Hours: Bike rentals peak at two key times - midday and early evening - for both casual and registered users, possibly aligning with lunch hours and the end of the standard workday. 
    Off-Peak Hours: Rentals are lowest in the early hours of the morning for both user types, which is expected as this is a common low-activity period.
    Evening Activity: There is a notable drop in rentals for casual users in the evening, while registered users maintain a higher level of rentals, reinforcing the idea that registered users may be using the service for commuting.""",
    'casual': """Casual bike rentals start low in the early hours, gradually increase, and peak in the mid to late afternoon.
The highest average rentals occur around the 15th hour (3 PM), suggesting a preference for afternoon rentals.
There is a steady decline in rentals after the peak, with significantly fewer rentals in the evening hours.""",
    'registered': """Registered bike rentals show two significant peaks, likely corresponding to typical commuting hours in the morning and evening.
The largest peak occurs in the early evening, suggesting a high use of bikes for the return commute.
Rentals are lowest in the early morning, increase for the morning commute, and then taper off until the afternoon.
There is a more pronounced evening peak compared to the morning, indicating a possible preference for biking home from work or after daytime activities.
"""}

# Hourly Trends
    st.subheader('Average Hourly Bike Rental Trends')
    if user_type == 'Both':
        hourly_data = data.groupby('hour')[['casual', 'registered']].mean().reset_index()
        fig_hourly = px.bar(hourly_data, x='hour', y=['casual', 'registered'], title='Average Bike Rentals per Hour', barmode='group')
    else:
        hourly_data = data.groupby('hour')[user_type.lower()].mean().reset_index()
        fig_hourly = px.bar(hourly_data, x='hour', y=user_type.lower(), title=f'Average {user_type} Bike Rentals per Hour')

    st.plotly_chart(fig_hourly, use_container_width=True)

    user_type_key = user_type.lower() if user_type != 'Both' else 'both'
    if user_type_key in text_analysis_hourly:
        st.write(text_analysis_hourly[user_type_key])
    else:
        st.write("No text analysis available for the selected user type.")

        
    col1, col2 = st.columns(2)

    with col1: 
          
# Day of the Week Trends
        st.subheader('Bike Rental Trends by Day of the Week')
        if user_type == 'Both':
            weekday_data = data.groupby('weekday')[['casual', 'registered']].mean().reset_index()
            fig_weekday = px.bar(weekday_data, x='weekday', y=['casual', 'registered'], title='Average Bike Rentals by Day of the Week', barmode='group')
        else:
            weekday_data = data.groupby('weekday')[user_type.lower()].mean().reset_index()
            fig_weekday = px.bar(weekday_data, x='weekday', y=user_type.lower(), title=f'Average {user_type} Bike Rentals by Day of the Week')

        st.plotly_chart(fig_weekday, use_container_width=True)

    with col2:
        
        st.subheader('Bike Rental Trends by Workday and Weekend')

# Function to map 'workingday' to 'Workday' and 'Weekend'
        def map_workingday(data_frame):
            data_frame['workingday'] = data_frame['workingday'].map({0: 'Weekend', 1: 'Workday'})
            return data_frame

        if user_type == 'Both':
            average_counts = data.groupby('workingday')[['casual', 'registered']].mean().reset_index()
            average_counts = map_workingday(average_counts)
            fig_workweek = px.bar(average_counts, x='workingday', y=['casual', 'registered'], title='Average Bike Rentals by Day Type', 
                                  barmode='group')
        else:
            average_counts = data.groupby('workingday')[user_type.lower()].mean().reset_index()
            average_counts = map_workingday(average_counts)
            fig_workweek = px.bar(average_counts, x='workingday', y=user_type.lower(), title=f'Average {user_type} Bike Rentals by Day Type')

        st.plotly_chart(fig_workweek, use_container_width=True)
    
    
    st.title('Histogram')
    col1, col2 = st.columns(2)

 
    with col1:
        filtered_data = data
        month_mapping = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
        7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }

        season_mapping = {
        1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'
        }

        day_mapping = {
        0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday',
        4: 'Thursday', 5: 'Friday', 6: 'Saturday'
        }

        weather_sit = {
        1: 'Clear',
        2: 'Misty',
        3: 'Light Precipitation',
        4: 'Heavy Precipitation'
        }

        filtered_data['mnth'] = filtered_data['mnth'].map(month_mapping)
        filtered_data['season'] = filtered_data['season'].map(season_mapping)
        filtered_data['weekday'] = filtered_data['weekday'].map(day_mapping)
        filtered_data['weathersit'] = filtered_data['weathersit'].map(weather_sit)

    
        hour_options = ['All'] + list(range(24))
        hour = st.selectbox('Select Hour', options=hour_options, index=0)

        month_options = ['All'] + list(data['mnth'].unique())
        month = st.selectbox('Select Month', options=month_options, index=0)

        season_options = ['All'] + list(data['season'].unique())
        season = st.selectbox('Select Season', options=season_options, index=0)

        weekday_options = ['All'] + list(data['weekday'].unique())
        weekday = st.selectbox('Select Weekday', options=weekday_options, index=0)

        col21, col22 = st.columns(2)

        with col22:
            FilterButton = st.button('Apply filters')   
            if FilterButton:
                if hour != 'All':
                    filtered_data = filtered_data[filtered_data['hr'] == int(hour)]
                if month != 'All':
                    filtered_data = filtered_data[filtered_data['mnth'] == month]
                if season != 'All':
                    filtered_data = filtered_data[filtered_data['season'] == season]
                if weekday != 'All':
                    filtered_data = filtered_data[filtered_data['weekday'] == weekday]
    
        with col21:
            ResotreButton = st.button('Restore charts to original data')
            if ResotreButton:
                filtered_data = data

    st.write('Resulting table after applying the selected filter:')
    st.dataframe(data=filtered_data) 

    with col2:

        daily_aggregated_data = filtered_data.groupby('dteday')['cnt'].sum().reset_index()

    # Line chart of bike rentals over time
        line_fig = px.line(daily_aggregated_data, x='dteday', y='cnt', labels={'cnt': 'Number of Rentals', 'dteday': 'Date'}, title='Bike Rentals Over Time')
        st.plotly_chart(line_fig)
        

    # Box Plot of Bike Rentals per Weather Situation
    fig_weather_box = px.box(filtered_data, x='weathersit', y='cnt', 
                             labels={'cnt': 'Number of Rentals', 'weathersit': 'Weather Situation'},
                             title='Bike Rentals Distribution per Weather Situation')

    fig_weather_box.update_layout(
            xaxis_title='Weather Situation',
            yaxis_title='Number of Bike Rentals'
        )

    st.plotly_chart(fig_weather_box)


    # Calculate average rentals per weather situation
  
    def create_scatter_plot(df, x_var, y_var):
        fig = px.scatter(df, x=x_var, y=y_var, title=f'{y_var} vs. {x_var}')
        return fig

    st.title('Interactive Scatter Plot')

    x_var = st.selectbox('Select the variable for the x-axis:', numerical_columns)

    y_var = st.selectbox('Select the variable for the y-axis:', numerical_columns, index=numerical_columns.index('cnt') if 'cnt' in numerical_columns else 0)

# Display the scatter plot
    fig = create_scatter_plot(data, x_var, y_var)
    st.plotly_chart(fig, use_container_width=True)

    def plot_correlation_heatmap(df, columns):
        plt.figure(figsize=(15, 15))
        sns.heatmap(data=df[columns].corr(), annot=True, fmt=".2f", cmap='coolwarm')
        st.pyplot(plt)

    st.header('Correlation Matrix')
    columns = ['season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'hum', 'daylight_hours', 'hr_sin', 'hr_cos']
    plot_correlation_heatmap(data, columns)


        
def pred_page():
    st.header("Bike Rental Prediction Model")
    model = joblib.load('best_model_xgb_2.joblib')
    with open('weekday_encoding.json', 'r') as json_file:
        weekday_encoding = json.load(json_file)

    # Define weather labels
    weather_labels = {
        1: 'Clear',
        2: 'Mist',
        3: 'Light Precipitation',
        4: 'Heavy Precipitation'
    }
    
    def encode_season(season):
        season_encoding = {'spring': 1, 'summer': 2, 'fall': 3, 'winter': 4}
        return season_encoding[season]
    
    st.title('Bike Rental Prediction Dashboard')
    
    season_months = {
        'spring': [3, 4, 5],
        'summer': [6, 7, 8],
        'fall': [9, 10, 11],
        'winter': [12, 1, 2]
    }
    
    def get_high_humidity(humidity_value):
        return 1 if humidity_value >= 0.75 else 0
    

    st.subheader('Input Features')
    
    col1, col2 = st.columns(2)
    with col1:
        season = st.selectbox('Season', list(season_months.keys()))
        mnth = st.selectbox('Month', season_months[season])
        weathersit = st.selectbox('Weather Situation', list(weather_labels.values()))
        weekday = st.selectbox('Day of the Week', list(weekday_encoding.keys()))
        daylight_hours = st.number_input('Daylight Hours', min_value=0, max_value=24, step=1, value=12)
    
    with col2:
        hr = st.slider('Hour', 0, 23, 1)
        temp = st.slider('Temperature (normalized)', min_value=0.0, max_value=1.0, step=0.01, value=0.5)
        hum = st.slider('Humidity (normalized)', min_value=0.0, max_value=1.0, step=0.01, value=0.5)
        workingday = st.radio('Working Day', [0, 1])
        holiday = st.radio('Holiday', [0, 1])
        

    
    max_hour = 23
    hr_sin = np.sin(2 * np.pi * hr / max_hour)
    hr_cos = np.cos(2 * np.pi * hr / max_hour)
    

    # Predict button
    if st.button('Predict'):
        high_humidity_placeholder = get_high_humidity(hum)
        windspeed_placeholder = 0.5
        season_encoded = encode_season(season)
        weekday_encoded = weekday_encoding[weekday]
        weathersit_encoded = {v: k for k, v in weather_labels.items()}[weathersit]

        user_input = pd.DataFrame({
        'season': [season_encoded], 
        'mnth': [mnth], 
        'holiday': [holiday], 
        'weekday': [weekday_encoded],
        'workingday': [workingday], 
        'weathersit': [weathersit_encoded], 
        'temp': [temp], 
        'hum': [hum],
        'daylight_hours': [daylight_hours],
        'hr_sin': [hr_sin],  # Include hr_sin
        'hr_cos': [hr_cos]   # Include hr_cos
    }, index=[0])

        # Make predictions
        prediction = model.predict(user_input)

        # Display the prediction
        st.subheader(f'The predicted bike rental count is: {prediction[0]:.4f}')

    
def model_page():
    st.header("Summary of the Models")
    
    st.subheader("Data Split")
    st.markdown("""
    Feature Selection:

Selected input features for the model: 'season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'hum', 'daylight_hours', 'hr_sin', 'hr_cos'.

Data Preparation:

- X: Input features dataset extracted based on selected features.
- y: Target variable, representing the value to be predicted by the model.<br>
Data Splitting:

Training and Testing Sets: Data split into 70% for training and 30% for testing.
- X_train and y_train: Training datasets for model fitting.
- X_test and y_test: Testing datasets for model evaluation.


Future Data Preparation:
- X_future and y_future: Last 168 data points from the testing set, reserved for future predictions or validation.""", unsafe_allow_html=True)
    
    st.image('ModelsMAPE.jpeg', caption='Results', width=400)
    
    st.subheader("XGBoost Regressive Model")
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold !important;
    }
    </style>
    <p class="big-font">MAPE for XGBoost: 0.844</p>
    """, unsafe_allow_html=True)
    
    st.subheader("""XGBoost Advantages:""")
    st.markdown(""" Use of MAPE (Mean Absolute Percentage Error) as Metric:

- MAPE is a common metric used to evaluate the performance of regression models, including XGBoost, especially in applications like bike-sharing demand prediction.

- Interpretability: MAPE expresses error as a percentage, making it easy to interpret and communicate to stakeholders. For instance, a MAPE of 5% means that the model's predictions are, on average, within 5% of the actual values.

- Scale-Independence: It’s a scale-independent measure, which is useful in comparing the accuracy of models across different scales of data.

- Applicability to Demand Prediction: In bike-sharing, where demand prediction is crucial, MAPE helps in understanding the accuracy of predictions in terms of actual usage numbers.""")

    col1, col2 = st.columns(2)
    
    with col1:
        st.image('XGBScreenshot1.png', caption='Results', use_column_width='always')
        
    with col2:
        st.image('XGBScreenshot2.png' , caption='Randomized Search CV', use_column_width='always')

        
    features = ['season','yr', 'mnth', 'hr', 'holiday', 'weekday', 'weathersit', 'temp', 'hum', 'windspeed']
    X = data[features]
    y = data['cnt']

# Splitting the data
    train_size = int(0.7 * len(data))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
# Saving the last 7 days data as "future data"
    X_future = X_test[-168:]
    y_future = y_test[-168:]
    
    def load_json(filename):
        with open(filename, 'r') as file:
            return json.load(file)
    
    
    
    
    end_time = pd.Timestamp('2012-12-31 23:00:00')
    start_time = end_time - pd.Timedelta(hours=167)
    date_range = pd.date_range(start=start_time, end=end_time, freq='H')
    
    def plot_line_plotly(y_pred, y_future):
        data = pd.DataFrame({'y_pred': y_pred, 'y_future': y_future})
        fig = px.line(data, title='Line Plot of y_pred and y_future')
        fig.add_scatter(x=data.index, y=data['y_future'], mode='lines', name='y_future')
        return fig

    st.subheader("Line Plot of XGBoost Predictions")

# Load y_pred data
    y_pred = load_json('y_pred.json')  
    y_future = [float(value) for value in y_future if value]
    

    try:

        if y_pred and y_future and len(y_pred) == len(y_future):
            
        # Plotting
            fig = plot_line_plotly(y_pred, y_future)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("The lengths of y_pred and y_future do not match or one of them is empty.")

    except ValueError:
        st.error("Please enter valid numeric values for y_future.")
        
        
    st.subheader("Random Forest Regressive Model")
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold !important;
    }
    </style>
    <p class="big-font">MAPE for Random Forest: 1.049</p>
    """, unsafe_allow_html=True)
    
    st.markdown("""Randomized Search Cross-Validation:

Employed RandomizedSearchCV with a TimeSeriesSplit cross-validator to tune the hyperparameters of a RandomForestRegressor.
10 time-series cross-validation folds were used, and 30 different combinations of parameters were evaluated, resulting in a total of 300 fits.
Parameter Space:
- Explored hyperparameters include max_depth (tree depth), max_features (number of features to consider when looking for the best split), min_samples_leaf (minimum samples required to be at a leaf node), min_samples_split (minimum number of samples required to split an internal node), and n_estimators (number of trees in the forest).
The parameters were chosen randomly from the specified distributions during the search process.
Performance Evaluation:

- The mean_absolute_percentage_error was used as the scoring metric to evaluate model performance, with a lower score indicating a better model.
The model achieved a MAPE of 1.049, which indicates that the average error of the predictions as a percentage of the actual values is approximately 1.049%.""")
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.image('RFScreenshot1.png', caption='Results', use_column_width='always')
        
    with col2:
        st.image('RFScreenshot2.png', caption='Randomized Search CV', use_column_width='always')
        
    st.subheader("Line Plot of Random Forest Predictions")
    y_pred_rf = load_json('y_pred_rf.json')
    fig = plot_line_plotly(y_pred_rf, y_future)
    st.plotly_chart(fig, use_container_width=True)
        
    st.subheader("Linear Regression Model")
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold !important;
    }
    </style>
    <p class="big-font">MAPE for Linear Regression: 5.698</p>
    """, unsafe_allow_html=True)
    
    st.markdown("""Model Fitting:

The model_lr.fit method is called with X_train and y_train, which represent the training data and target values, respectively, to train the model.
Cross-Validation:

- A TimeSeriesSplit with 5 splits is initialized for cross-validation. This is important for time series data to maintain the temporal order of observations.
cross_val_score is used to evaluate model_lr across the different folds of the time series split. It is set up to use the mean absolute percentage error (MAPE) as the scoring metric.
The MAPE is negated (greater_is_better=False) because by convention, cross-validation functions expect a "greater is better" scenario, but since MAPE is an error metric where lower is better, it is negated to fit this convention.
The cross-validation scores (cv_scores) are printed, which show the MAPE for each fold. The mean of these scores is also printed to give an average measure of model performance across all folds.
<br>
Prediction:

- The trained model_lr is used to predict future values y_pred_lr using X_future, which is assumed to be a set of features for future observations.
Performance Evaluation:

The MAPE is calculated between the predicted values y_pred_lr and the actual future values y_future, providing a measure of the model's predictive performance.""", unsafe_allow_html=True)

        
    st.subheader("Line Plot of Linear Regression Predictions")
    y_pred_lr = load_json('y_pred_lr.json')
    fig = plot_line_plotly(y_pred_lr, y_future)
    st.plotly_chart(fig, use_container_width=True)

    def create_mape_comparison_chart():
    # Create the figure with Scatter plot
        fig = go.Figure(data=[
            go.Scatter(
                x=['XGBoost', 'Random Forest', 'Linear Regression'],
                y=[0.844, 1.049, 5.698],
                mode='markers',
                marker=dict(size=12)
        )
    ])

        fig.update_layout(
            title='Comparison of MAPEs',
            xaxis_title='Models',
            yaxis_title='MAPE (%)',
            yaxis=dict(range=[0, max(0, 1, 6) + 0.5])
    )

        return fig
    fig = create_mape_comparison_chart()

    st.subheader("""Comparison of Model MAPEs""")
    st.plotly_chart(fig, use_container_width=True)
    
    
        
    
def recs_page():

    st.title("Business Recommendations")
    
    st.subheader("Seasonal Supply Management")
    
    def display_details():
        st.subheader("""Scaling Up/Down:""")
        st.write("""Increase the number of bikes available during peak seasons and reduce during off-peak seasons to optimize costs and meet demand efficiently.
        Based on the data analysis conducted, we can see that the peak season for bike sharing runs from May to September. This is consistent year over year. """)
   
        
        col1, col2 = st.columns(2)
        
        with col1:
            daily_data = data.groupby('dteday')[['cnt']].sum().reset_index()
            fig = px.line(daily_data, x='dteday', y=['cnt'], title='Time Series of Casual and Registered Bike Counts')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            average_cnt_month = data.groupby('mnth')['cnt'].mean().reset_index()
            fig2 = px.bar(average_cnt_month, x='mnth', y='cnt', title='Average Bike Sharing by Month')
            fig2.update_xaxes(title_text='Month')
            fig2.update_yaxes(title_text='Average Count of Total Rental Bikes')
            st.plotly_chart(fig2, use_container_width=True)
    with st.expander("Click here to expand/collapse for more details"):

        display_details()
        
   
        
    st.subheader("""Targeted User Recommendations""")
    def display_details():
        st.subheader("""User Type Management:""")
        st.write("""Increase bike availability during commuting hours on weekdays to cater to registered users while ensuring ample bike availability during weekends.""")
        st.markdown("""
            - Peak Hours: The usage by casual riders peaks during the midday hours, specifically around 12 PM to 3 PM, with the brightest spots appearing in these intervals. This suggests that casual users may be using the bikes for leisure activities or errands during lunch hours or early afternoon breaks.
            - Weekday Variation: There's a noticeable pattern across the weekdays, but without the labels for the days of the week, it's hard to pinpoint which days are busier. However, there is a visible variation, with some weekdays showing higher usage than others.
            - Low Usage: Similar to the pattern observed with casual users on a monthly basis, the early morning hours (before 5 AM) and later in the evening (after 7 PM) show significantly lower usage.
""")
       
        st.subheader("""Pricing Strategy:""")
        st.write("""Implement higher rates during commuting hours on weekdays when demand from registered users is high and offer discounted rates or special promotions for casual users during weekends. This will encourage casual users, who engage in bike sharing less frequently, to start sharing bikes.""")
        st. markdown("""
            - Commuter Patterns: For registered users, there are two distinct peaks corresponding to typical rush hours, around 7-9 AM and 4-6 PM. This suggests a strong pattern of use for commuting to and from work.
- Weekday Consistency: The usage is fairly consistent across weekdays, with a drop-off in the late-night hours.""")
        st. subheader("""Time Based Rates:""")
        st.markdown("""
        - Dynamic Pricing System: Use a dynamic pricing system that automatically adjusts rates in real-time based on demand (i.e., $3 per 30 minutes).
- Off-Peak Promotions: Offer discounts or special flat rates to encourage usage during low-demand hours (i.e., $1.50 per 30 minutes).
- Communication to Users: Inform users through the mobile app or website about the different rate schedules so they can plan their trips accordingly.""")
        st.subheader("""Type-Based Rates:""")
        st.markdown("""- Subscription Plans: Offer subscription plans with preferential rates for registered users who use the service regularly.
- Day or Week Passes: For casual users, offer day or week passes that allow unlimited travel during those periods at a fixed cost.""")
        
        
        average_counts = data.groupby(['hr', 'weekday', 'mnth'])[['casual', 'registered']].mean().reset_index()

        col1, col2 = st.columns(2)
        with col1:
        # Create a heatmap for 'hr' vs. 'weekday' with 'casual' counts
            fig1 = go.Figure(data=go.Heatmap(
                x=average_counts['hr'],
                y=average_counts['weekday'],
                z=average_counts['casual'],
                colorscale='Viridis',
                colorbar_title='Average Casual Counts',
            ))
            fig1.update_layout(
                title='Average Casual Bike Counts per Hour and Weekday',
                xaxis=dict(title='Hour of the Day'),
                yaxis=dict(title='Weekday'),
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
        # Create a heatmap for 'hr' vs. 'weekday' with 'registered' counts
            fig2 = go.Figure(data=go.Heatmap(
                x=average_counts['hr'],
                y=average_counts['weekday'],
                z=average_counts['registered'],
                colorscale='Viridis',
                colorbar_title='Average Registered Counts',
            ))
            fig2.update_layout(
                title='Average Registered Bike Counts per Hour and Weekday',
                xaxis=dict(title='Hour of the Day'),
                yaxis=dict(title='Weekday'),
            )
            st.plotly_chart(fig2, use_container_width=True)
            
        data['workingday'] = data['workingday'].map({0: 'Weekend', 1: 'Workday'})
        average_counts = data.groupby('workingday')[['casual', 'registered']].mean().reset_index()
        fig_workweek = px.bar(average_counts, x='workingday', y=['casual', 'registered'], title='Average Bike Rentals by Day Type', 
                                  barmode='group')
        fig_workweek.update_yaxes(title_text='Total Rental Bikes')
        st.plotly_chart(fig_workweek, use_container_width=True)
     
    with st.expander("Click here to expand/collapse for more details"):
        display_details()
        
        
    st.subheader("""Weather Based Promotions""")
        
    def display_details():
        st.subheader("""Dynamic Pricing:""")
        st.write("""Implement weather-based pricing strategies, such as discounts on rainy or extremely cold days to encourage usage. Create special promotions for days with favorable weather forecasts to attract more riders.""")
        
        st.subheader("""Seasonal Memberships:""")
        st.write("""Introduce seasonal membership plans that cater to tourists or occasional users who may prefer short-term commitments. Reward regular users with loyalty programs that offer seasonal benefits or discounts""")
        
        
        col1, col2 = st.columns(2)
        with col1:
        
            weather_labels = {
                1: 'Clear',
                2: 'Mist',
                3: 'Light Precipitation',
                4: 'Heavy Precipitation'
            }

            data['weather_situation'] = data['weathersit'].map(weather_labels)

            average_counts = data.groupby('weather_situation')[['cnt']].mean().reset_index()

            fig = px.bar(average_counts, x='weather_situation', y=['cnt'],
                         title='Average Bike Rentals by Weather Situation'
                        )
            fig.update_xaxes(title_text='Weather Situation')
            fig.update_yaxes(title_text='Average Count of Rental Bikes')

            st.plotly_chart(fig, use_container_width=True) 
        
        with col2:
        
            season_labels = {
            1: 'Spring',
            2: 'Summer',
            3: 'Fall',
            4: 'Winter'
            }
            data['season_name'] = data['season'].map(season_labels)
            average_counts = data.groupby('season_name')['cnt'].mean().reset_index()
            fig1 = px.bar(average_counts, x='season_name', y='cnt',
                     title='Average Bike Rentals by Season')
            fig1.update_xaxes(title_text='Season')
            fig1.update_yaxes(title_text='Average Count of Rental Bikes')
            st.plotly_chart(fig1, use_container_width=True) 
# Multiselect for user to choose the variables

    
    
    with st.expander("Click here to expand/collapse for more details"):
        display_details()
        
    st.subheader("""Operational Cost Optimization""")
    def display_details():
        st.subheader("""Maintenance Scheduling:""")
        st.write("""Schedule maintenance during off-peak hours that align with lower user demands. This timing is pivotal in ensuring that bikes are readily available during peak usage times, a crucial factor in maintaining high levels of customer satisfaction and service reliability.""")
        st.markdown("""
        - Higher quality maintenance work.
        - Extends the overall life of the bikes
        - Ensures customers are riding bikes in top condition, fosters higher rates of user retention""")
        
        st. subheader("""Seasonal Staffing:""")
        st.write("""Adjust staffing levels based on seasonal demand, with more staff during peak seasons for maintenance and customer service.""")
        st.markdown("""
        - More efficient use of resources.
        - Reduces overtime costs.""")
        
        sum_cnt_month = data.groupby('mnth')['cnt'].sum().reset_index()

        # Create a bar plot for 'mnth' vs. average 'cnt'
        fig2 = px.bar(sum_cnt_month, x='mnth', y='cnt', title='Total Bike Sharing by Month')
        fig2.update_xaxes(title_text='Month')
        fig2.update_yaxes(title_text='Total Rental Bikes')
        
        st.plotly_chart(fig2, use_container_width=True)
        
    with st.expander("Click here to expand/collapse for more details"):
        display_details()

      
    
    
def main():
    # Load data
    data = load_data()
    pages = {
        "Home Page": home_page,
        "Exploratory Data Analysis": eda_page,
        "Business Recommendations": recs_page,
        "Prediction": pred_page,  
        "Models Summary" : model_page
    }

    st.sidebar.title('Navigation')
    page = st.sidebar.selectbox("Select a page:", list(pages.keys()))
    if page in ["Prediction", "Models Summary", "Business Recommendations", "Home Page"]:
        pages[page]()  
    else:
        pages[page](data)

if __name__ == "__main__":
    main()

