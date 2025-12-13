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

# --- New Mapping for Car Model to Body Type ---
CAR_TYPE_MAPPING = {
    'BMW 3 Series': 'Lux_Sedan', 'Datsun Go': 'Hatchback', 'Datsun Go Plus': 'Hatchback', 
    'Datsun Redi Go': 'Hatchback', 'Ford Ecosport': 'SUV', 'Ford Endeavour': 'Lux_SUV', 
    'Ford FREESTYLE': 'SUV', 'Ford Figo Aspire': 'Sedan', 'Ford New Figo': 'Hatchback', 
    'Honda Accord': 'Sedan', 'Honda Amaze': 'Sedan', 'Honda BR-V': 'SUV', 
    'Honda Brio': 'Hatchback', 'Honda CRV': 'Lux_SUV', 'Honda City': 'Sedan', 
    'Honda Civic': 'Sedan', 'Honda Jazz': 'Hatchback', 'Honda WR-V': 'SUV', 
    'Hyundai ALCAZAR': 'SUV', 'Hyundai AURA': 'Sedan', 'Hyundai Creta': 'SUV', 
    'Hyundai Elite i20': 'Hatchback', 'Hyundai Eon': 'Hatchback', 
    'Hyundai GRAND I10 NIOS': 'Hatchback', 'Hyundai Grand i10': 'Hatchback', 
    'Hyundai NEW I20': 'Hatchback', 'Hyundai NEW I20 N LINE': 'Hatchback', 
    'Hyundai NEW SANTRO': 'Hatchback', 'Hyundai New Elantra': 'Sedan', 
    'Hyundai Santro Xing': 'Hatchback', 'Hyundai Sonata': 'Sedan', 
    'Hyundai Tucson New': 'SUV', 'Hyundai VENUE': 'SUV', 'Hyundai Verna': 'Sedan', 
    'Hyundai Xcent': 'Sedan', 'Hyundai i10': 'Hatchback', 'Hyundai i20': 'Hatchback', 
    'Hyundai i20 Active': 'Hatchback', 'Jeep Compass': 'SUV', 
    'Jeep GRAND CHEROKEE': 'Lux_SUV', 'KIA CARENS': 'SUV', 'KIA SELTOS': 'SUV', 
    'KIA SONET': 'SUV', 'Mahindra BOLERO NEO': 'SUV', 'Mahindra Bolero': 'SUV', 
    'Mahindra KUV 100 NXT': 'Hatchback', 'Mahindra Kuv100': 'Hatchback', 
    'Mahindra MARAZZO': 'SUV', 'Mahindra Scorpio': 'SUV', 'Mahindra Thar': 'SUV', 
    'Mahindra TUV 300 PLUS': 'SUV', 'Mahindra TUV300': 'SUV', 'Mahindra XUV 3OO': 'SUV', 
    'Mahindra XUV500': 'SUV', 'Mahindra XUV700': 'SUV', 'Maruti A Star': 'Hatchback', 
    'Maruti Alto': 'Hatchback', 'Maruti Alto 800': 'Hatchback', 
    'Maruti Alto K10': 'Hatchback', 'Maruti Baleno': 'Hatchback', 'Maruti BREZZA': 'SUV', 
    'Maruti Celerio': 'Hatchback', 'Maruti Celerio X': 'Hatchback', 
    'Maruti Ciaz': 'Sedan', 'Maruti Dzire': 'Sedan', 'Maruti Eeco': 'Hatchback', 
    'Maruti Ertiga': 'SUV', 'Maruti IGNIS': 'Hatchback', 
    'Maruti New Wagon-R': 'Hatchback', 'Maruti OMNI E': 'Hatchback', 
    'Maruti S CROSS': 'SUV', 'Maruti S PRESSO': 'Hatchback', 'Maruti Ritz': 'Hatchback', 
    'Maruti Swift': 'Hatchback', 'Maruti Swift Dzire': 'Sedan', 
    'Maruti Vitara Brezza': 'SUV', 'Maruti Wagon R': 'Hatchback', 
    'Maruti Wagon R 1.0': 'Hatchback', 'Maruti Wagon R Stingray': 'Hatchback', 
    'Maruti XL6': 'SUV', 'Maruti Zen Estilo': 'Hatchback', 'MG HECTOR': 'SUV', 
    'MG HECTOR PLUS': 'SUV', 'Nissan MAGNITE': 'SUV', 'Nissan Micra': 'Hatchback', 
    'Nissan Micra Active': 'Hatchback', 'Nissan Sunny': 'Sedan', 'Nissan Terrano': 'SUV', 
    'Renault Captur': 'SUV', 'Renault Duster': 'SUV', 'Renault Kiger': 'SUV', 
    'Renault Kwid': 'Hatchback', 'Renault Pulse': 'Hatchback', 'Renault TRIBER': 'SUV', 
    'Skoda KUSHAQ': 'SUV', 'Skoda Octavia': 'Sedan', 'Skoda RAPID': 'Sedan', 
    'Skoda SLAVIA': 'Sedan', 'Tata ALTROZ': 'Hatchback', 'Tata Bolt': 'Hatchback', 
    'Tata Harrier': 'SUV', 'Tata Hexa': 'SUV', 'Tata NEXON': 'SUV', 'Tata PUNCH': 'SUV', 
    'Tata Safari': 'SUV', 'Tata TIGOR': 'Sedan', 'Tata TIAGO NRG': 'Hatchback', 
    'Tata Tiago': 'Hatchback', 'Tata Zest': 'Sedan', 'Toyota Camry': 'Lux_Sedan', 
    'Toyota Corolla Altis': 'Sedan', 'Toyota Etios': 'Sedan', 
    'Toyota Etios Liva': 'Hatchback', 'Toyota Fortuner': 'Lux_SUV', 
    'Toyota Glanza': 'Hatchback', 'Toyota Innova': 'SUV', 'Toyota Innova Crysta': 'SUV', 
    'Toyota URBAN CRUISER': 'SUV', 'Toyota YARIS': 'Sedan', 'Volkswagen Ameo': 'Sedan', 
    'Volkswagen Jetta': 'Sedan', 'Volkswagen Polo': 'Hatchback', 'Volkswagen T-ROC': 'SUV', 
    'Volkswagen TAIGUN': 'SUV', 'Volkswagen TIGUAN': 'Lux_SUV', 'Volkswagen Vento': 'Sedan'
}


# --- Create the Encoding Maps ---
def create_mapping(options):
    """Creates a dictionary map from string value to its list index (0-based label)."""
    return {value: index for index, value in enumerate(options)}

CAR_NAME_MAP = create_mapping(CAR_NAMES)
FUEL_MAP = create_mapping(FUEL_OPTIONS)
DRIVE_MAP = create_mapping(DRIVE_OPTIONS)
TYPE_MAP = create_mapping(TYPE_OPTIONS)
LOCATION_MAP = create_mapping(LOCATION_OPTIONS)


# --- CALLBACK FUNCTION to automatically update Car Body Type (Triggered by Car Model change) ---
def update_car_type_on_model_change():
    """Reads the selected car name and updates the car_type state variable."""
    # The new car name is stored in st.session_state.input_car_name (the key of the selectbox)
    car_name = st.session_state.input_car_name
    
    # Look up the corresponding body type, defaulting to the first option ('Hatchback') if not found
    default_type = CAR_TYPE_MAPPING.get(car_name, TYPE_OPTIONS[0])
    
    # Update the session state variable for car_type
    st.session_state.car_type = default_type


# --- REUSABLE PREDICTION FUNCTIONS (Unchanged) ---
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


# --- 2. STREAMLIT UI DESIGN ---

# Custom CSS for high contrast and modern look (Omitted for brevity, assumed to be here)

st.markdown('<p class="main-header">🚗 Used Car Price Predictor</p>', unsafe_allow_html=True)
st.markdown("---") 

# --- 3. INPUT GATHERING ---

# Initialize session state for all inputs and prediction result
initial_car_name = CAR_NAMES[0]
# Use the mapping to ensure the initial car_type is correct
initial_car_type = CAR_TYPE_MAPPING.get(initial_car_name, TYPE_OPTIONS[0]) 

input_defaults = {
    'car_name': initial_car_name, 'year': 2018, 'owner': 1, 'distance': 50000, 
    'fuel': FUEL_OPTIONS[0], 'car_type': initial_car_type, 
    'drive': DRIVE_OPTIONS[0], 
    'location': LOCATION_OPTIONS[0], 'last_prediction': None
}
for key, default in input_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

tab1, tab2, tab3 = st.tabs(["*1. Identity", "2. Specs", "3. Predict*"])

# --- Define a container to hold the Car Body Type selectbox ---
# This allows us to clear and redraw it later if needed, though the fix below 
# should be sufficient.
car_type_container = st.container()

with tab1:
    st.subheader("🚘 Car Identity Details")
    
    st.session_state.car_name = st.selectbox(
        'Select Car Model', 
        options=CAR_NAMES, 
        key='input_car_name', 
        index=CAR_NAMES.index(st.session_state.car_name), 
        help="Choose the exact model name.",
        # Use the callback to update the state variable
        on_change=update_car_type_on_model_change 
    )
    col_t1_1, col_t1_2 = st.columns(2)
    with col_t1_1:
        st.session_state.year = st.slider('Year of Registration', min_value=2010, max_value=2025, value=st.session_state.year, step=1, key='input_year', help="The year the car was first registered.")
    with col_t1_2:
        st.session_state.owner = st.selectbox('Number of Previous Owners', options=OWNER_OPTIONS, key='input_owner', index=OWNER_OPTIONS.index(st.session_state.owner), help="Select 1, 2, or 3+ owners.")
    
with tab2:
    st.subheader("⚙️ Usage and Technical Specifications")
    
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
        # --- THE CORE FIX ---
        # Get the value from the session state, which was updated by the callback in tab1
        current_car_type = st.session_state.car_type
        
        # Calculate the index for the selectbox based on the session state value
        try:
            current_index = TYPE_OPTIONS.index(current_car_type)
        except ValueError:
            current_index = 0
            
        # Display the selectbox using the calculated index
        # The return value updates the session state again, making the box interactive
        st.session_state.car_type = st.selectbox(
            'Car Body Type', 
            options=TYPE_OPTIONS, 
            key='input_type', 
            index=current_index, 
            help="e.g., Hatchback, Sedan, SUV."
        )
        st.session_state.drive = st.selectbox('Transmission Type', options=DRIVE_OPTIONS, key='input_drive', index=DRIVE_OPTIONS.index(st.session_state.drive), help="Manual or Automatic.")
    
    st.markdown("---")
    st.subheader("📍 Location")
    st.session_state.location = st.selectbox('Location (City where the car is being sold)', options=LOCATION_OPTIONS, key='input_location', index=LOCATION_OPTIONS.index(st.session_state.location), help="The city is used as a pricing factor.")


# --- 4. PREDICTION LOGIC ---

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
            
            # Display Prediction
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
