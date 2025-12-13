import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import requests
from io import BytesIO

st.set_page_config(page_title="Car Price Predictor", layout="wide", initial_sidebar_state="expanded")
# --- 1. CONFIGURATION AND INITIAL SETUP ---
MODEL_PATH = 'trained_model.pkl' 
DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id=1p6erIlfcy55XJ7RfYF26XZ0BSkqhoKnZ"
FEATURE_NAMES = ['Car Name', 'Year', 'Distance', 'Owner', 'Fuel', 'Location', 'Drive', 'Type']

# --- Define Lists of Possible Values (CRITICAL: Order must match training data) ---

CAR_NAMES = [
    'Maruti S PRESSO', 'Hyundai Xcent', 'Tata Safari', 'Maruti Vitara Brezza', 
    'Tata Tiago', 'Maruti Swift', 'Hyundai i20', 'Renault Kwid', 'Hyundai Grand i10', 
    'Maruti IGNIS', 'Honda Brio', 'Hyundai Elite i20', 'Honda City', 'Maruti Baleno',
    'Honda WR-V', 'Honda Amaze', 'Maruti Alto 800', 'Maruti Celerio', 'Ford Ecosport', 
    'Maruti Ciaz', 'Datsun Redi Go', 'Tata TIAGO NRG', 'Hyundai Santro Xing', 
    'Ford FREESTYLE', 'Maruti Dzire', 'Maruti Alto', 'Hyundai NEW SANTRO', 
    'Maruti Alto K10', 'Ford Endeavour', 'Maruti Swift Dzire', 'Maruti Wagon R 1.0', 
    'Hyundai GRAND I10 NIOS', 'Maruti Celerio X', 'Toyota URBAN CRUISER', 
    'Mahindra XUV500', 'Hyundai Verna', 'Hyundai VENUE', 'Tata NEXON', 
    'Mahindra KUV 100 NXT', 'Toyota YARIS', 'Mahindra XUV 3OO', 'Renault TRIBER', 
    'Hyundai Tucson New', 'Mahindra TUV300', 'Toyota Glanza', 'Maruti Eeco', 
    'Renault Duster', 'Hyundai i10', 'Nissan MAGNITE', 'KIA SONET', 'Maruti Ertiga', 
    'Honda Jazz', 'KIA SELTOS', 'Volkswagen Ameo', 'Renault Kiger', 'Honda Accord', 
    'Hyundai NEW I20', 'Tata ALTROZ', 'Maruti A Star', 'Maruti Ritz', 'Nissan Micra', 
    'Hyundai Eon', 'Hyundai Creta', 'Mahindra Bolero', 'Toyota Etios Liva', 
    'Maruti New Wagon-R', 'Nissan Micra Active', 'Tata Harrier', 'Tata TIGOR', 
    'Tata PUNCH', 'Volkswagen Polo', 'Toyota Camry', 'Toyota Corolla Altis', 
    'Honda Civic', 'Volkswagen Vento', 'Maruti S Cross', 'Skoda Octavia', 
    'Hyundai i20 Active', 'Hyundai New Elantra', 'Honda BR-V', 'Hyundai AURA', 
    'Mahindra Thar', 'Maruti Zen Estilo', 'Hyundai NEW I20 N LINE', 'Tata Hexa', 
    'Maruti XL6', 'Honda CRV', 'Toyota Innova', 'Skoda Rapid', 'Datsun Go', 
    'Maruti Wagon R Stingray', 'Volkswagen TIGUAN', 'Toyota Etios', 'Tata Zest', 
    'Ford New Figo', 'Mahindra Kuv100', 'Skoda SLAVIA', 'Mahindra Scorpio', 
    'Nissan Terrano', 'Volkswagen TAIGUN', 'Renault Captur', 'Mahindra XUV700', 
    'Hyundai Sonata', 'Mahindra BOLERO NEO', 'Maruti BREZZA', 'Datsun Go Plus', 
    'Hyundai ALCAZAR', 'BMW 3 Series', 'Jeep Compass', 'Toyota Innova Crysta', 
    'KIA CARENS', 'Skoda KUSHAQ', 'Volkswagen Jetta', 'Renault Pulse', 
    'Ford Figo Aspire', 'Maruti Wagon R', 'Mahindra TUV 300 PLUS', 'MG HECTOR PLUS', 
    'Tata Bolt', 'MG HECTOR', 'Volkswagen T-ROC', 'Maruti OMNI E', 
    'Jeep GRAND CHEROKEE', 'Toyota Fortuner', 'Mahindra MARAZZO', 'Nissan Sunny'
]
FUEL_OPTIONS = ['Petrol', 'Diesel', 'LPG', 'CNG']
DRIVE_OPTIONS = ['Manual', 'Automatic']
TYPE_OPTIONS = ['Hatchback', 'Sedan', 'SUV', 'Lux_SUV', 'Lux_Sedan']
LOCATION_OPTIONS = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Kolkata']
OWNER_OPTIONS = [1, 2, 3] 

# --- Create the Encoding Maps ---
def create_mapping(options):
    """Creates a dictionary map from string value to its list index (0-based label)."""
    return {value: index for index, value in enumerate(options)}

CAR_NAME_MAP = create_mapping(CAR_NAMES)
FUEL_MAP = create_mapping(FUEL_OPTIONS)
DRIVE_MAP = create_mapping(DRIVE_OPTIONS)
TYPE_MAP = create_mapping(TYPE_OPTIONS)
LOCATION_MAP = create_mapping(LOCATION_OPTIONS)


# --- REUSABLE PREDICTION FUNCTIONS ---

def get_input_dataframe(car_name, year, distance, owner, fuel, location, drive, car_type):
    """Encodes inputs and creates the prediction DataFrame."""
    try:
        input_data_dict = {
            'Car Name': CAR_NAME_MAP.get(car_name),
            'Year': year,
            'Distance': distance,
            'Owner': owner,
            'Fuel': FUEL_MAP.get(fuel),
            'Location': LOCATION_MAP.get(location),
            'Drive': DRIVE_MAP.get(drive),
            'Type': TYPE_MAP.get(car_type)
        }
        if None in input_data_dict.values():
             raise ValueError("Input value missing in encoding map.")
             
        return pd.DataFrame([input_data_dict], columns=FEATURE_NAMES).astype(float)
    except Exception:
        return None

def predict_price(df, model):
    """Runs the prediction on the DataFrame."""
    if df is None or model is None:
        return -1
    try:
        prediction = model.predict(df)[0]
        return max(0, prediction) # Return 0 for negative predictions
    except Exception:
        return -1

# Function to load the model (Preserving the download logic from app.py)
@st.cache_resource(show_spinner=False)
def load_model(url):
    """Downloads the model from the URL and loads it."""
    st.info(f"⏳ Downloading model file. This may take a moment...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() 

        model = pickle.load(BytesIO(response.content))
        
        st.success(f"✅ Model loaded successfully from cloud storage.")
        return model
    
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error downloading model from URL: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model (pickle failed): {e}")
        return None

# Load the Model
model = load_model(DOWNLOAD_URL)


# --- 2. STREAMLIT UI DESIGN (ADOPTED FROM APP1.PY) ---

# Custom CSS for high contrast and modern look (ADOPTED FROM APP1.PY)
st.markdown("""
<style>
.main-header { 
    font-size: 3.5em; 
    font-weight: 800; 
    color: #007BFF; /* Blue accent for modern look */ 
    text-align: center; 
    padding: 10px 0; 
    margin-bottom: 10px;
}
.stTabs [data-testid="stTab"] {
    font-size: 1.1em;
    font-weight: bold;
}
/* Style for the Predict button */
.stButton>button { 
    background-color: #007BFF; 
    color: white; 
    font-weight: bold; 
    border-radius: 8px; 
    padding: 12px 24px; 
    margin-top: 20px; 
    border: none;
    transition: all 0.2s;
}
.stButton>button:hover {
    background-color: #0056b3;
}
/* Metric styles for high contrast */
div[data-testid="stMetric"] {
    background-color: #e6f2ff; /* Light blue background */
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}
div[data-testid="stMetricValue"] {
    font-size: 2.5em; 
    color: #004085; /* Dark blue for metric value */
    font-weight: 700;
}
div[data-testid="stMetricLabel"] {
    font-size: 1.2em;
    font-weight: 600;
}
.summary-box {
    background-color: #f8f9fa; 
    border-left: 5px solid #007BFF; 
    padding: 15px; 
    border-radius: 5px; 
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🚗 Used Car Price Predictor</p>', unsafe_allow_html=True)
st.markdown("---") 

# --- 3. INPUT GATHERING (ADOPTED FROM APP1.PY) ---

# Initialize session state for all inputs and prediction result
input_defaults = {
    'car_name': CAR_NAMES[0], 'year': 2018, 'owner': 1, 'distance': 50000, 
    'fuel': FUEL_OPTIONS[0], 'car_type': TYPE_OPTIONS[0], 'drive': DRIVE_OPTIONS[0], 
    'location': LOCATION_OPTIONS[0], 'last_prediction': None
}
for key, default in input_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

tab1, tab2, tab3 = st.tabs(["*1. Identity", "2. Specs", "3. Predict*"])

with tab1:
    st.subheader("🚘 Car Identity Details")
    st.session_state.car_name = st.selectbox('Select Car Model', options=CAR_NAMES, key='input_car_name', index=CAR_NAMES.index(st.session_state.car_name), help="Choose the exact model name.")
    col_t1_1, col_t1_2 = st.columns(2)
    with col_t1_1:
        st.session_state.year = st.slider('Year of Registration', min_value=2010, max_value=2025, value=st.session_state.year, step=1, key='input_year', help="The year the car was first registered.")
    with col_t1_2:
        st.session_state.owner = st.selectbox('Number of Previous Owners', options=OWNER_OPTIONS, key='input_owner', index=OWNER_OPTIONS.index(st.session_state.owner), help="Select 1, 2, or 3+ owners.")
    
with tab2:
    st.subheader("⚙️ Usage and Technical Specifications")
    
    # --- Real-time feedback ---
    years_old = 2025 - st.session_state.year
    kms_per_year = st.session_state.distance / years_old if years_old > 0 else 0
    
    col_t2_1, col_t2_2 = st.columns(2)
    with col_t2_1:
        st.session_state.distance = st.number_input('Distance Driven (in kilometers)', min_value=1000, max_value=300000, value=st.session_state.distance, step=1000, key='input_distance', help="Total distance covered by the car.")
        st.session_state.fuel = st.selectbox('Fuel Type', options=FUEL_OPTIONS, key='input_fuel', index=FUEL_OPTIONS.index(st.session_state.fuel), help="Is it Petrol, Diesel, LPG, or CNG?")
        
        if st.session_state.distance > 200000:
            st.warning("⚠️ High mileage detected. This may lower the predicted price.")
        elif kms_per_year < 5000 and years_old > 3:
            st.info("ℹ️ Very low mileage for the car's age. (Positive factor)")

    with col_t2_2:
        st.session_state.car_type = st.selectbox('Car Body Type', options=TYPE_OPTIONS, key='input_type', index=TYPE_OPTIONS.index(st.session_state.car_type), help="e.g., Hatchback, Sedan, SUV.")
        st.session_state.drive = st.selectbox('Transmission Type', options=DRIVE_OPTIONS, key='input_drive', index=DRIVE_OPTIONS.index(st.session_state.drive), help="Manual or Automatic.")
    
    st.markdown("---")
    st.subheader("📍 Location")
    st.session_state.location = st.selectbox('Location (City where the car is being sold)', options=LOCATION_OPTIONS, key='input_location', index=LOCATION_OPTIONS.index(st.session_state.location), help="The city is used as a pricing factor.")


# --- 4. PREDICTION LOGIC (ADOPTED FROM APP1.PY) ---

with tab3:
    st.subheader("✅ Confirm Inputs and Get Prediction")
    
    # Dynamic Summary Box
    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
    st.markdown(f"**Car:** *{st.session_state.car_name}* ({st.session_state.year})")
    st.markdown(f"**Usage:** {st.session_state.distance:,} km | {st.session_state.owner} Owner(s)")
    st.markdown(f"**Specs:** {st.session_state.fuel} | {st.session_state.drive} | {st.session_state.car_type}")
    st.markdown(f"**Location:** {st.session_state.location}")
    st.markdown('</div>', unsafe_allow_html=True)

    
    if st.button('💰 PREDICT CAR PRICE', use_container_width=True):
        input_df = get_input_dataframe(
            st.session_state.car_name, st.session_state.year, st.session_state.distance, 
            st.session_state.owner, st.session_state.fuel, st.session_state.location, 
            st.session_state.drive, st.session_state.car_type
        )
        prediction = predict_price(input_df, model)

        if prediction > 0:
            st.session_state.last_prediction = prediction
            st.balloons() 
            st.markdown("### 🏆 Estimated Selling Price is Ready!")

            prediction_lakhs = prediction / 100000
            
            # Display Prediction (using st.metric and st.info, no graphs)
            st.metric(
                label="Predicted Price (INR)", 
                value=f"₹ {prediction_lakhs:,.2f} Lakhs" 
            )
            
            # Show a simple listing range for completeness
            st.info(f"Recommended Listing Range: **₹{prediction_lakhs * 0.95:,.2f} Lakhs** — **₹{prediction_lakhs * 1.05:,.2f} Lakhs**")

        elif prediction == 0:
            st.warning("Prediction was calculated as less than zero. Please review the inputs or model.")
        else:
            st.error("Prediction failed. Please ensure all inputs are selected and the model is loaded.")


# --- 5. FOOTER ---
st.markdown("---")
st.caption("Disclaimer: AI predictions are based on historical data. Final price may vary.")
