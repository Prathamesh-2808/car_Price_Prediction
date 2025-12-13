import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import requests
from io import BytesIO # Needed if you want to load the model directly from memory

# --- 1. CONFIGURATION AND INITIAL SETUP ---
MODEL_PATH = 'trained_model.pkl' # !!! CHANGE THIS TO YOUR ACTUAL MODEL FILENAME !!!

DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id=1p6erIlfcy55XJ7RfYF26XZ0BSkqhoKnZ"
# --- Define Lists of Possible Values (CRITICAL: Order must match training data) ---
# ... (All your lists are correct and remain the same) ...

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
LOCATION_OPTIONS = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Kolkata'] # Used for encoding

# Features that were NOT Label Encoded (using them as-is)
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

# Function to load the model (same)
@st.cache_resource(show_spinner=False)
def load_model(path, url):
    """Downloads the model from the URL and loads it."""
    st.info(f"⏳ Downloading model file  This may take a moment...")
    
    try:
        # Use requests to download the file content
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)

        # Load the model directly from the downloaded content (in memory)
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
model = load_model(MODEL_PATH, DOWNLOAD_URL)


# --- 2. STREAMLIT UI DESIGN (same) ---
st.set_page_config(page_title="Car Price Predictor", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-header { font-size: 3em; font-weight: bold; color: #FF9900; /* Orange accent */ text-align: center; padding: 10px 0; }
.stButton>button { background-color: #FF9900; color: white; font-weight: bold; border-radius: 8px; padding: 10px 20px; margin-top: 20px; }
.prediction-box { border: 3px solid #FF9900; padding: 20px; border-radius: 10px; text-align: center; background-color: #fff8e1; /* Light yellow background */ }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🚗 Used Car Price Predictor</p>', unsafe_allow_html=True)
st.write("Enter the car features below to get an estimated selling price.")

# --- 3. INPUT GATHERING (same) ---
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Car Details")
    car_name = st.selectbox('Car Name', options=CAR_NAMES)
    year = st.slider('Year of Purchase/Registration', min_value=2010, max_value=2025, value=2018, step=1)
    owner = st.selectbox('Number of Owners', options=OWNER_OPTIONS)

with col2:
    st.header("Usage & Type")
    distance = st.number_input('Distance Driven (in kms)', min_value=1000, value=50000, step=1000)
    fuel = st.selectbox('Fuel Type', options=FUEL_OPTIONS)
    car_type = st.selectbox('Car Body Type', options=TYPE_OPTIONS)

with col3:
    st.header("Location & Specs")
    location = st.selectbox('Location (City)', options=LOCATION_OPTIONS)
    drive = st.selectbox('Transmission Type', options=DRIVE_OPTIONS)


# --- 4. PREDICTION LOGIC (UPDATED) ---

if st.button('Predict Price'):
    if model is None:
        st.warning("Prediction cannot be performed because the model failed to load.")
    else:
        try:
            # 1. Apply the manual Label Encoding for all 5 categorical features
            encoded_car_name = CAR_NAME_MAP.get(car_name) # Changed names to car_name here for consistency
            encoded_fuel = FUEL_MAP.get(fuel)
            encoded_drive = DRIVE_MAP.get(drive)
            encoded_type = TYPE_MAP.get(car_type)
            encoded_location = LOCATION_MAP.get(location)

            # Safety check: Ensure all categorical features were found
            if None in [encoded_car_name, encoded_fuel, encoded_drive, encoded_type, encoded_location]:
                st.error("Error in encoding: One of the selected values could not be mapped to a label. Please check the feature lists.")
                st.stop()
            
            # --- FIX FOR WARNING: Create a DataFrame with Feature Names ---
            
            # CRITICAL: Feature list order MUST match the training data
            FEATURE_NAMES = ['Car Name', 'Year', 'Distance', 'Owner', 'Fuel', 'Location', 'Drive', 'Type']
            
            # Create a dictionary of the input data
            input_data_dict = {
                'Car Name': encoded_car_name,
                'Year': year,
                'Distance': distance,
                'Owner': owner,
                'Fuel': encoded_fuel,
                'Location': encoded_location,
                'Drive': encoded_drive,
                'Type': encoded_type
            }

            # Create the DataFrame with the correct order and data types (float)
            X_new_df = pd.DataFrame([input_data_dict], columns=FEATURE_NAMES).astype(float)
            
            # 2. Make Prediction (using the DataFrame)
            prediction = model.predict(X_new_df)[0]
            
            # 3. Display Result
            if prediction < 0:
                 formatted_price = "Prediction resulted in a negative value (Model Error)."
            else:
                 # Format the price (assuming INR)
                 formatted_price = f"₹ {prediction:,.2f}"
                 
            st.markdown("---")
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown(f"## **Estimated Price**")
            st.markdown(f"### **{formatted_price}**", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction Error! Please ensure the **trained_model.pkl** file is compatible.")
            st.error(f"Details: {e}")
            st.info("💡 **Developer Tip:** The feature list `['Car Name', 'Year', 'Distance', 'Owner', 'Fuel', 'Location', 'Drive', 'Type']` must be in the exact order used during model training.")
