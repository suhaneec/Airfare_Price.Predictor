import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
import warnings
from datetime import datetime

# Ignore warnings
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")

# Google Drive link for dataset
DATASET_URL = "https://drive.google.com/uc?id=1jBlIQUBYTYJFsfcHN8zaWqxhcOy20b_Q"
DATASET_PATH = "updated_flight_dataset.csv"

# Download dataset if not found
if not os.path.exists(DATASET_PATH):
    print("📥 Downloading dataset from Google Drive...")
    gdown.download(DATASET_URL, DATASET_PATH, quiet=False)

# Ensure the file is not empty
if os.path.getsize(DATASET_PATH) == 0:
    raise ValueError("❌ Error: The dataset file is empty!")

# Read dataset
df = pd.read_csv(DATASET_PATH)
print("✅ Dataset loaded successfully!")

# Display first few rows to verify
print(df.head())

# Ensure 'flight_date' is in datetime format
df['Flight Date'] = pd.to_datetime(df['Flight Date'], errors='coerce')

# Create 'departure_hour' for analysis
df['departure_hour'] = pd.to_datetime(df['Departure Time'], format='%H:%M', errors='coerce').dt.hour

# Streamlit app configuration
st.set_page_config(page_title="Flight price_predictor", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        /* Apply a darker blue gradient background to the sidebar */
        [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #050A13, #0B132B, #1B263B, #415A77) !important;
color: white;
        }

        /* Ensure sidebar text remains visible */
        [data-testid="stSidebar"] * {
            color: white !important;
            font-weight: bold;
        }

        /* Style dropdowns, sliders, and input boxes */
        select, input, .stSlider, .stMultiSelect {
            background-color: rgba(255, 255, 255, 0.15) !important;
            color: white !important;
            border-radius: 8px;
            padding: 8px;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }

        /* Style buttons */
        button {
            background-color: #00509E !important;
            color: white !important;
            border-radius: 8px;
            font-weight: bold;
        }

    </style>
""", unsafe_allow_html=True)


# Sidebar configuration
with st.sidebar:
    # Load and display the logo in the sidebar
    logo = Image.open(r'APP new logo.jpg')
    st.image(logo, width=1000)
    
    st.header("FILTER OPTIONS")

    # Airline filter
    airlines_options = ['Select All'] + list(df['Airline'].unique())
    airlines = st.multiselect("🏢 SELECT AIRLINE:", options=airlines_options, default=['Select All'])
    if 'Select All' in airlines:
        airlines = df['Airline'].unique()
        
    # Filter for Origin
    origin_options = ['Select All'] + sorted(df['Origin'].unique().tolist())
    selected_origin = st.multiselect("✈️SELECT ORIGIN:", options=origin_options, default=['Select All'])
    if 'Select All' in selected_origin:
        selected_origin = df['Origin'].unique().tolist()

    # Filter for Destination
    destination_options = ['Select All'] + sorted(df['Destination'].unique().tolist())
    selected_destination = st.multiselect("🌍 SELECT DESTINATION:", options=destination_options, default=['Select All'])
    if 'Select All' in selected_destination:
        selected_destination = df['Destination'].unique().tolist()
    # Number of stops filter
    stops_options = ['Select All'] + list(df['Number of Stops'].unique())
    stops = st.multiselect("🛑 SELECT NUMBER OF STOPS:", options=stops_options, default=['Select All'])
    if 'Select All' in stops:
        stops = df['Number of Stops'].unique()

    # Date range filter
    min_date = df['Flight Date'].min().date()
    max_date = df['Flight Date'].max().date()
    st.markdown(f"**Available Date Range: {min_date} to {max_date}**")
    date_range = st.slider(
        "SELECT DATE RANGE:", 
        min_value=min_date, 
        max_value=max_date, 
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    # Price range slider
    min_price = df['Price (₹)'].min()
    max_price = df['Price (₹)'].max()
    price_slider = st.slider("SELECT PRICE RANGE (\u20B9):", min_value=int(min_price), max_value=int(max_price), value=(int(min_price), int(max_price)))


# Apply filters
filtered_df = df[
    (df['Origin'].isin(selected_origin)) &
    (df['Destination'].isin(selected_destination))
]

# Apply filters
filtered_df = df[
    (df['Airline'].isin(airlines)) & 
    (df['Number of Stops'].isin(stops)) & 
    (df['Flight Date'].dt.date >= date_range[0]) & 
    (df['Flight Date'].dt.date <= date_range[1]) & 
    (df['Price (₹)'] >= price_slider[0]) & 
    (df['Price (₹)'] <= price_slider[1])
]

st.title("\U0001F4C8 AIRFARE PRICE PREDICTOR \U0001F680")
st.markdown("**Analyze flight pricing trends, understand key factors influencing airfare, and gain insights for better travel planning.**")
st.markdown("---")

# Convert 'Flight Date' to datetime
df['Flight Date'] = pd.to_datetime(df['Flight Date'], errors='coerce')

# Compute KPIs
num_airlines = df['Airline'].nunique()
total_flights = df.shape[0]
earliest_flight = df['Flight Date'].min().date()
latest_flight = df['Flight Date'].max().date()
avg_ticket_price = round(df['Price (₹)'].mean(), 2)
most_frequent_departure = df['Origin'].mode()[0]
most_frequent_destination = df['Destination'].mode()[0]

# --- STYLIZED KPI SECTION ---
st.markdown("## ✈️ DATA OVERVIEW")
st.markdown("**Get a quick glance at key metrics for flights in this dataset.**")

# Define CSS for KPI box styling
st.markdown("""
    <style>
        .kpi-box {
            background: linear-gradient(135deg, #002147, #0096FF);
            padding: 18px;
            border-radius: 12px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            color: white;
            box-shadow: 3px 3px 10px rgba(0,0,0,0.3);
            margin: 10px;
        }
        .kpi-title {
            font-size: 16px;
            font-weight: normal;
            opacity: 0.9;
        }
        .kpi-value {
            font-size: 28px;
            font-weight: bold;
            margin-top: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Function to create a KPI box
def kpi_box(title, value):
    return f"""
        <div class="kpi-box">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
        </div>
    """

# First row of KPIs
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(kpi_box("🏢 TOTAL AIRLINES", num_airlines), unsafe_allow_html=True)
with col2:
    st.markdown(kpi_box("🛫 TOTAL FLIGHTS", total_flights), unsafe_allow_html=True)
with col3:
    st.markdown(kpi_box("📅 EARLIEST FLIGHT DATE", earliest_flight), unsafe_allow_html=True)

# Second row of KPIs
col4, col5, col6 = st.columns(3)
with col4:
    st.markdown(kpi_box("📅 LATEST FLIGHT DATE", latest_flight), unsafe_allow_html=True)
with col5:
    st.markdown(kpi_box("💰 AVG. TICKET PRICE", f"₹{avg_ticket_price}"), unsafe_allow_html=True)
with col6:
    st.markdown(kpi_box("🏙 MOST FREQUENT ROUTE", f"{most_frequent_departure} → {most_frequent_destination}"), unsafe_allow_html=True)

st.markdown("---")
st.subheader("🌲 **AIRLINES, ROUTES & TICKET PRICES**")

# Use the full dataset (df) instead of filtered_df to prevent fluctuation
fig = px.treemap(
    df,  # Ensures Treemap remains unchanged regardless of filters
    path=["Airline", "Origin", "Destination"],
    values="Price (₹)",
    color="Price (₹)",
    color_continuous_scale="Blues",
    title="WHICH AIRLINES HAVE THE MOST EXPENSIVE ROUTES? (UNAFFECTED BY FILTERS)"
)

fig.update_layout(
    font=dict(color="white", size=14),
    plot_bgcolor="#121212",
    paper_bgcolor="#121212"
)

st.plotly_chart(fig)
st.markdown("""
- Some airlines **charge significantly higher prices on specific routes**.
- Budget airlines **dominate shorter domestic flights**.
- International flights **generally have more price variation** across different airlines.
- This treemap **remains unchanged**, allowing users to compare airline pricing patterns independently of filters.
""")
st.markdown("---")

# VISUAL 1
# ---- CREATE A SINGLE ROW WITH TWO VISUALS ----
st.subheader("📊 **IMPACT OF AIRLINE & NUMBER OF STOPS ON THE TICKET PRICES**")

# Create two columns for side-by-side visualization
col1, col2 = st.columns(2)

# 1️⃣ Impact of Airline on Ticket Price (Box Plot)
with col1:
    st.markdown("✈️ **AIRLINE Vs. TICKET PRICES**")
    fig1 = px.box(
        filtered_df, x='Airline', y='Price (₹)', color='Airline', 
        color_discrete_sequence=px.colors.sequential.Blues
    )
    fig1.update_layout(
        showlegend=False,
        xaxis_title="Airline",
        yaxis_title="Ticket Price (₹)",
        font=dict(color="lightblue", size=14, family="Arial", weight="bold"),
        plot_bgcolor="#121212",
        paper_bgcolor="#121212"
    )
    st.plotly_chart(fig1)
    st.markdown("**\U0001F4DD INSIGHT:** Some airlines charge much higher prices than others, but cheaper airlines are also available.")
# Impact of Number of Stops on Ticket Price (Box Plot)
with col2:
    st.markdown("🛑 **NUMBER OF STOPS Vs. TICKET PRICES**")
    fig2 = px.box(
        filtered_df, x='Number of Stops', y='Price (₹)', color='Number of Stops', 
        color_discrete_sequence=px.colors.sequential.Blues
    )
    fig2.update_layout(
        showlegend=False,
        xaxis_title="Number of Stops",
        yaxis_title="Ticket Price (₹)",
        font=dict(color="lightblue", size=14, family="Arial", weight="bold"),
        plot_bgcolor="#121212",
        paper_bgcolor="#121212"
    )
    st.plotly_chart(fig2)
    st.markdown("**\U0001F4DD INSIGHT:** Non-stop flights generally cost more, while flights with layovers tend to be cheaper.")
#-----
st.markdown("---")
# 4. FLIGHT DURATION VS TICKET PRICE
st.subheader("\U0000231B FLIGHT DURATION VS TICKET PRICE")

# Price filter specific to this visual
price_range_visual = st.slider("FILTER PRICE RANGE FOR THIS VISUAL (\u20B9):", min_value=int(min_price), max_value=int(max_price), value=(int(min_price), int(max_price)))
filtered_visual_df = filtered_df[(filtered_df['Price (₹)'] >= price_range_visual[0]) & (filtered_df['Price (₹)'] <= price_range_visual[1])]

fig = px.scatter(filtered_visual_df, x='Duration (Minutes)', y='Price (₹)', color_discrete_sequence=['#00BFFF'])
fig.update_layout(showlegend=False, xaxis_title='DURATION (MINUTES)', yaxis_title='TICKET PRICE (\u20B9)', font=dict(color='lightblue', size=14, family='Arial', weight='bold'))
st.plotly_chart(fig)
st.markdown("**\U0001F4DD INSIGHT: Longer flights can be more expensive, but some short flights are costly too.**")
st.markdown("---")

# SIDE-BY-SIDE VISUALIZATIONS
st.subheader("\U0001F4CA FLIGHT FREQUENCY AND DURATION DISTRIBUTION")
col1, col2 = st.columns(2)

with col1:
    airline_counts = filtered_df['Airline'].value_counts().reset_index()
    airline_counts.columns = ['Airline', 'count']
    fig = px.bar(airline_counts, x='Airline', y='count', color='Airline', color_discrete_sequence=px.colors.sequential.Blues)
    fig.update_layout(showlegend=False, xaxis_title='AIRLINE', yaxis_title='NUMBER OF FLIGHTS', font=dict(color='lightblue', size=14, family='Arial', weight='bold'))
    st.plotly_chart(fig)
    st.markdown("**\U0001F4DD INSIGHT: Some airlines operate more flights than others, influencing ticket availability and pricing.**")

with col2:
    fig = px.histogram(filtered_df, x='Duration (Minutes)', nbins=50, color_discrete_sequence=['#1E90FF'])
    fig.update_layout(showlegend=False, xaxis_title='DURATION (MINUTES)', yaxis_title='FREQUENCY', font=dict(color='lightblue', size=14, family='Arial', weight='bold'))
    st.plotly_chart(fig)
    st.markdown("**\U0001F4DD INSIGHT: Most flights have moderate durations, but there are some long-haul flights with longer durations.**")
st.markdown("---")
# Ensure 'Flight Date' exists before processing
if 'Flight Date' in df.columns:
    df['Flight Date'] = pd.to_datetime(df['Flight Date'], errors='coerce')

    # Prepare Data for Graphs
    daily_avg_price = df.groupby('Flight Date')['Price (₹)'].mean().reset_index()
    df['flight_day_of_week'] = df['Flight Date'].dt.day_name()
    avg_price_by_day = df.groupby('flight_day_of_week')['Price (₹)'].mean().reset_index()

    # Maintain correct day order
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    avg_price_by_day['flight_day_of_week'] = pd.Categorical(avg_price_by_day['flight_day_of_week'], categories=day_order, ordered=True)
    avg_price_by_day = avg_price_by_day.sort_values('flight_day_of_week')

    # LINE GRAPHS SIDE BY SIDE
    st.subheader("\U0001F4C9 AVERAGE PRICE TRENDS")
    col1, col2 = st.columns(2)

    with col1:
        # 1. AVERAGE TICKET PRICE OVER TIME
        fig1 = px.line(daily_avg_price, x='Flight Date', y='Price (₹)', color_discrete_sequence=['#1E90FF'],
                       title="AVERAGE TICKET PRICE OVER TIME ")
        fig1.update_layout(showlegend=False, xaxis_title='DATE', yaxis_title='AVERAGE TICKET PRICE  (₹)', 
                           font=dict(color='lightblue', size=14, family='Arial', weight='bold'))
        st.plotly_chart(fig1)
        st.markdown("**\U0001F4DD INSIGHT: Ticket prices change over time, going up and down depending on demand or season.**")

    with col2:
        # 2. AVERAGE PRICE BY DAY OF THE WEEK
        if not avg_price_by_day.empty:
            fig2 = px.line(avg_price_by_day, x='flight_day_of_week', y='Price (₹)', color_discrete_sequence=['#1E90FF'],
                           title="AVERAGE TICKET PRICE BY DAY OF THE WEEK")
            fig2.update_layout(showlegend=False, xaxis_title='DAY OF THE WEEK', yaxis_title='AVERAGE TICKET PRICE (₹)', 
                               font=dict(color='lightblue', size=14, family='Arial', weight='bold'))
            st.plotly_chart(fig2)
            st.markdown("**\U0001F4DD INSIGHT: Ticket prices tend to fluctuate throughout the week, being highest on Tuesdays.**")
        else:
            st.warning("No valid flight date data available for day-wise price trends.")

else:
    st.warning("Flight Date column is missing. Unable to generate price trends.")
st.markdown("---")
import plotly.figure_factory as ff

st.subheader("📊 **AIRLINE PRICES ACROSS ROUTES**")
# Handle missing values before sorting
origin_options = ['Select All'] + sorted(df['Origin'].dropna().unique().tolist())
selected_origin = st.multiselect("✈️ **Select Origin:**", options=origin_options, default=['Select All'])
if 'Select All' in selected_origin:
    selected_origin = df['Origin'].dropna().unique().tolist()

destination_options = ['Select All'] + sorted(df['Destination'].dropna().unique().tolist())
selected_destination = st.multiselect("🌍 **Select Destination:**", options=destination_options, default=['Select All'])
if 'Select All' in selected_destination:
    selected_destination = df['Destination'].dropna().unique().tolist()

# Create a pivot table of average ticket prices for each Origin-Destination-Airline combination
pivot_table = filtered_df.pivot_table(index="Origin", columns="Destination", values="Price (₹)", aggfunc="mean")

# Check if the pivot table is empty (no flights available for selected filters)
if pivot_table.empty:
    st.warning("No data available for the selected filters. Please try different filter options.")
else:
    # Create a heatmap visualization
    fig = ff.create_annotated_heatmap(
        z=pivot_table.fillna(0).values,  # Fill NaN with 0 to avoid errors
        x=pivot_table.columns.tolist(),  # Destination cities
        y=pivot_table.index.tolist(),    # Origin cities
        annotation_text=pivot_table.round(0).fillna(0).astype(str).values,  # Show price values
        colorscale="Blues",  # Blue color theme to match dashboard
        showscale=True
    )

    # Customize layout
    fig.update_layout(
        title="AVERAGE TICKET PRICES B/W ORIGING & DESTINATION",
        xaxis_title="DESTINATION",
        yaxis_title="ORIGIN",
        font=dict(color="white"),
        plot_bgcolor="#121212",  # Dark background
        paper_bgcolor="#121212"
    )

    # Display the heatmap
    st.plotly_chart(fig)

    st.markdown("**\U0001F4DD INSIGHT:  This heatmap helps identify price variations across different airlines and routes, making it easier to choose budget-friendly travel options.**")

st.markdown("---")

import plotly.express as px
import streamlit as st

# Create a two-column layout
col1, col2 = st.columns(2)

# 📊 Chart 1: Which Airline Operated the Most Flights?
with col1:
    st.subheader("✈️ **AIRLINE WITH MAXIMUM FLIGHTS**")

    # Count the number of flights per airline
    airline_flight_count = filtered_df["Airline"].value_counts().reset_index()
    airline_flight_count.columns = ["Airline", "Flight Count"]

    # Create a bar chart
    fig1 = px.bar(
        airline_flight_count, x="Airline", y="Flight Count", 
        color="Flight Count", color_continuous_scale="Blues",
       
    )

    # Apply styling for transparency
    fig1.update_layout(
        xaxis_title="AIRLINE",
        yaxis_title="NUMBER OF FLIGHTS",
        font=dict(color="white", size=14, family="Arial", weight="bold"),
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent chart background
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent surrounding background
    )

    # Display chart
    st.plotly_chart(fig1)

# 📊 Chart 2: Which Airline is the Cheapest?
with col2:
    st.subheader("💰 **CHEAPEST AIRLINES BASED ON PRICE**")

    # Calculate the average ticket price per airline
    cheapest_airline = filtered_df.groupby("Airline")["Price (₹)"].mean().reset_index()
    cheapest_airline = cheapest_airline.sort_values(by="Price (₹)", ascending=True)  # Sort in ascending order (cheapest first)

    # Create a bar chart for cheapest airlines
    fig2 = px.bar(
        cheapest_airline, x="Airline", y="Price (₹)", 
        color="Price (₹)", color_continuous_scale="Blues",
       
    )

    # Apply styling for transparency
    fig2.update_layout(
        xaxis_title="AIRLINE",
        yaxis_title="AVERAGE TICKET PRICE (₹)",
        font=dict(color="white", size=14, family="Arial", weight="bold"),
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent chart background
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent surrounding background
    )

    # Display chart
    st.plotly_chart(fig2)

st.markdown("---")
import streamlit as st
import pandas as pd
import joblib
import os
import gdown  # Install using: pip install gdown
import time

# Google Drive Links for Model and Feature Columns
MODEL_URL = "https://drive.google.com/uc?id=1oNac4uYGsMz0beRvRT5GFvNBXwNRhC6r"
FEATURES_URL = "https://drive.google.com/uc?id=1MFqlLzUFgGlJjGLqRPeG0kwQoNViwIml"

MODEL_PATH = "flight_price_rf_model.pkl"
FEATURES_PATH = "feature_columns.pkl"

# Function to download files if they don't exist
def download_file(url, output):
    if not os.path.exists(output):
        st.info(f"📥 Downloading {output}...")
        gdown.download(url, output, quiet=False)
        st.success(f"✅ {output} downloaded successfully!")

# Download the model and feature files if they don't exist
download_file(MODEL_URL, MODEL_PATH)
download_file(FEATURES_URL, FEATURES_PATH)

# Load the model and feature columns
try:
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    st.success("✅ Model and features loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model or feature columns: {e}")

# Streamlit App Title
st.title("✈️ Flight Price Predictor")
st.markdown("### Predict the price of your flight based on various factors.")
st.divider()

# Layout organization
col1, col2 = st.columns(2)

# User Inputs - First Column
with col1:
    airline = st.selectbox("✈️ SELECT AIRLINE", ['AirAsia', 'GO FIRST', 'Indigo', 'SpiceJet', 'StarAir', 'Trujet', 'Vistara'])
    flight_class = st.selectbox("🎫 SELECT CLASS", ['economy'])
    origin = st.selectbox("📍 SELECT ORIGIN", ['Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai'])
    destination = st.selectbox("📍 SELECT DESTINATION", ['Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai'])

# User Inputs - Second Column
with col2:
    stops = st.number_input("⏭ NUMBER OF STOPS", min_value=0, max_value=5, step=1)
    date = st.number_input("📆 SELECT DATE", min_value=1, max_value=31, step=1)
    month = st.number_input("📅 SELECT MONTH", min_value=1, max_value=12, step=1)
    year = st.number_input("📆 SELECT YEAR", min_value=2023, max_value=2030, step=1)
    duration = st.number_input("⏳ SELECT DURATION (Minutes)", min_value=30, max_value=1500, step=1)

# Function to preprocess user input for model prediction
def preprocess_input(airline, flight_class, origin, destination, stops, date, month, year, duration):
    # Create a dictionary for input features
    input_data = {
        "Number of Stops": stops,
        "Date": date,
        "Month": month,
        "Year": year,
        "Duration (Minutes)": duration
    }

    # One-hot encoding for categorical features (ensure alignment with training features)
    for col in feature_columns:
        if col.startswith("Airline_"):
            input_data[col] = 1 if f"Airline_{airline}" == col else 0
        elif col.startswith("Class_"):
            input_data[col] = 1 if f"Class_{flight_class}" == col else 0
        elif col.startswith("Origin_"):
            input_data[col] = 1 if f"Origin_{origin}" == col else 0
        elif col.startswith("Destination_"):
            input_data[col] = 1 if f"Destination_{destination}" == col else 0
        else:
            input_data[col] = 0  # Default value for missing columns

    return pd.DataFrame([input_data])

st.divider()

# Button Click for Prediction
if st.button("💰 Predict Price", use_container_width=True):
    # Preprocess input
    input_df = preprocess_input(airline, flight_class, origin, destination, stops, date, month, year, duration)
    
    # Make Prediction
    predicted_price = model.predict(input_df)[0]

    # Progress Bar Effect
    progress_bar = st.progress(0)
    for percent in range(100):
        time.sleep(0.01)  # Simulating progress
        progress_bar.progress(percent + 1)

    # Show animations after completion
    st.success(f"🎯 Predicted Flight Price: ₹{predicted_price:.2f}")  # Re-confirm price
    st.balloons()
    st.toast("🚀 Prediction Completed!", icon="✅")



