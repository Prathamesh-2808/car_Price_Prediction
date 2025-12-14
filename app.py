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
    'BMW 3 Series': 'Lux_Sedan',
    'Datsun Go': 'Hatchback',
    'Datsun Go Plus': 'Hatchback',
    'Datsun Redi Go': 'Hatchback',
    'Ford Ecosport': 'SUV',
    'Ford Endeavour': 'Lux_SUV',
    'Ford FREESTYLE': 'SUV',
    'Ford Figo Aspire': 'Sedan',
    'Ford New Figo': 'Hatchback',
    'Honda Accord': 'Sedan',
    'Honda Amaze': 'Sedan',
    'Honda BR-V': 'SUV',
    'Honda Brio': 'Hatchback',
    'Honda CRV': 'Lux_SUV',
    'Honda City': 'Sedan',
    'Honda Civic': 'Sedan',
    'Honda Jazz': 'Hatchback',
    'Honda WR-V': 'SUV',
    'Hyundai ALCAZAR': 'SUV',
    'Hyundai AURA': 'Sedan',
    'Hyundai Creta': 'SUV',
    'Hyundai Elite i20': 'Hatchback',
    'Hyundai Eon': 'Hatchback',
    'Hyundai GRAND I10 NIOS': 'Hatchback',
    'Hyundai Grand i10': 'Hatchback',
    'Hyundai NEW I20': 'Hatchback',
    'Hyundai NEW I20 N LINE': 'Hatchback',
    'Hyundai NEW SANTRO': 'Hatchback',
    'Hyundai New Elantra': 'Sedan',
    'Hyundai Santro Xing': 'Hatchback',
    'Hyundai Sonata': 'Sedan',
    'Hyundai Tucson New': 'SUV',
    'Hyundai VENUE': 'SUV',
    'Hyundai Verna': 'Sedan',
    'Hyundai Xcent': 'Sedan',
    'Hyundai i10': 'Hatchback',
    'Hyundai i20': 'Hatchback',
    'Hyundai i20 Active': 'Hatchback',
    'Jeep Compass': 'SUV',
    'Jeep GRAND CHEROKEE': 'Lux_SUV',
    'KIA CARENS': 'SUV',
    'KIA SELTOS': 'SUV',
    'KIA SONET': 'SUV',
    'Mahindra BOLERO NEO': 'SUV',
    'Mahindra Bolero': 'SUV',
    'Mahindra KUV 100 NXT': 'Hatchback',
    'Mahindra Kuv100': 'Hatchback',
    'Mahindra MARAZZO': 'SUV',
    'Mahindra Scorpio': 'SUV',
    'Mahindra Thar': 'SUV',
    'Mahindra TUV 300 PLUS': 'SUV',
    'Mahindra TUV300': 'SUV',
    'Mahindra XUV 3OO': 'SUV',
    'Mahindra XUV500': 'SUV',
    'Mahindra XUV700': 'SUV',
    'Maruti A Star': 'Hatchback',
    'Maruti Alto': 'Hatchback',
    'Maruti Alto 800': 'Hatchback',
    'Maruti Alto K10': 'Hatchback',
    'Maruti Baleno': 'Hatchback',
    'Maruti BREZZA': 'SUV',
    'Maruti Celerio': 'Hatchback',
    'Maruti Celerio X': 'Hatchback',
    'Maruti Ciaz': 'Sedan',
    'Maruti Dzire': 'Sedan',
    'Maruti Eeco': 'Hatchback',
    'Maruti Ertiga': 'SUV',
    'Maruti IGNIS': 'Hatchback',
    'Maruti New Wagon-R': 'Hatchback',
    'Maruti OMNI E': 'Hatchback',
    'Maruti S CROSS': 'SUV',
    'Maruti S PRESSO': 'Hatchback',
    'Maruti Ritz': 'Hatchback',
    'Maruti Swift': 'Hatchback',
    'Maruti Swift Dzire': 'Sedan',
    'Maruti Vitara Brezza': 'SUV',
    'Maruti Wagon R': 'Hatchback',
    'Maruti Wagon R 1.0': 'Hatchback',
    'Maruti Wagon R Stingray': 'Hatchback',
    'Maruti XL6': 'SUV',
    'Maruti Zen Estilo': 'Hatchback',
    'MG HECTOR': 'SUV',
    'MG HECTOR PLUS': 'SUV',
    'Nissan MAGNITE': 'SUV',
    'Nissan Micra': 'Hatchback',
    'Nissan Micra Active': 'Hatchback',
    'Nissan Sunny': 'Sedan',
    'Nissan Terrano': 'SUV',
    'Renault Captur': 'SUV',
    'Renault Duster': 'SUV',
    'Renault Kiger': 'SUV',
    'Renault Kwid': 'Hatchback',
    'Renault Pulse': 'Hatchback',
    'Renault TRIBER': 'SUV',
    'Skoda KUSHAQ': 'SUV',
    'Skoda Octavia': 'Sedan',
    'Skoda RAPID': 'Sedan',
    'Skoda SLAVIA': 'Sedan',
    'Tata ALTROZ': 'Hatchback',
    'Tata Bolt': 'Hatchback',
    'Tata Harrier': 'SUV',
    'Tata Hexa': 'SUV',
    'Tata NEXON': 'SUV',
    'Tata PUNCH': 'SUV',
    'Tata Safari': 'SUV',
    'Tata TIGOR': 'Sedan',
    'Tata TIAGO NRG': 'Hatchback',
    'Tata Tiago': 'Hatchback',
    'Tata Zest': 'Sedan',
    'Toyota Camry': 'Lux_Sedan',
    'Toyota Corolla Altis': 'Sedan',
    'Toyota Etios': 'Sedan',
    'Toyota Etios Liva': 'Hatchback',
    'Toyota Fortuner': 'Lux_SUV',
    'Toyota Glanza': 'Hatchback',
    'Toyota Innova': 'SUV',
    'Toyota Innova Crysta': 'SUV',
    'Toyota URBAN CRUISER': 'SUV',
    'Toyota YARIS': 'Sedan',
    'Volkswagen Ameo': 'Sedan',
    'Volkswagen Jetta': 'Sedan',
    'Volkswagen Polo': 'Hatchback',
    'Volkswagen T-ROC': 'SUV',
    'Volkswagen TAIGUN': 'SUV',
    'Volkswagen TIGUAN': 'Lux_SUV',
    'Volkswagen Vento': 'Sedan',
}

CAR_FUEL_MAPPING = {
    'BMW 3 Series': 'DIESEL',
    'Datsun Go': 'PETROL',
    'Datsun Go Plus': 'PETROL',
    'Datsun Redi Go': 'PETROL',
    'Ford Ecosport': 'DIESEL',
    'Ford Endeavour': 'DIESEL',
    'Ford FREESTYLE': 'DIESEL',
    'Ford Figo Aspire': 'DIESEL',
    'Ford New Figo': 'DIESEL',
    'Honda Accord': 'PETROL',
    'Honda Amaze': 'DIESEL',
    'Honda BR-V': 'PETROL',
    'Honda Brio': 'PETROL',
    'Honda CRV': 'PETROL',
    'Honda City': 'PETROL',
    'Honda Civic': 'PETROL',
    'Honda Jazz': 'PETROL',
    'Honda WR-V': 'DIESEL',
    'Hyundai ALCAZAR': 'DIESEL',
    'Hyundai AURA': 'DIESEL',
    'Hyundai Creta': 'DIESEL',
    'Hyundai Elite i20': 'DIESEL',
    'Hyundai Eon': 'PETROL',
    'Hyundai GRAND I10 NIOS': 'PETROL',
    'Hyundai Grand i10': 'PETROL',
    'Hyundai NEW I20': 'PETROL',
    'Hyundai NEW I20 N LINE': 'PETROL',
    'Hyundai NEW SANTRO': 'PETROL',
    'Hyundai New Elantra': 'PETROL',
    'Hyundai Santro Xing': 'PETROL',
    'Hyundai Sonata': 'AUTOMATIC',
    'Hyundai Tucson New': 'DIESEL',
    'Hyundai VENUE': 'PETROL',
    'Hyundai Verna': 'DIESEL',
    'Hyundai Xcent': 'PETROL',
    'Hyundai i10': 'PETROL',
    'Hyundai i20': 'PETROL',
    'Hyundai i20 Active': 'PETROL',
    'Jeep Compass': 'DIESEL',
    'Jeep GRAND CHEROKEE': 'DIESEL',
    'KIA CARENS': 'PETROL',
    'KIA SELTOS': 'DIESEL',
    'KIA SONET': 'DIESEL',
    'Mahindra BOLERO NEO': 'DIESEL',
    'Mahindra Bolero': 'DIESEL',
    'Mahindra KUV 100 NXT': 'PETROL',
    'Mahindra Kuv100': 'DIESEL',
    'Mahindra MARAZZO': 'DIESEL',
    'Mahindra Scorpio': 'DIESEL',
    'Mahindra Thar': 'DIESEL',
    'Mahindra TUV 300 PLUS': 'DIESEL',
    'Mahindra TUV300': 'DIESEL',
    'Mahindra XUV 3OO': 'DIESEL',
    'Mahindra XUV500': 'DIESEL',
    'Mahindra XUV700': 'DIESEL',
    'Maruti A Star': 'PETROL',
    'Maruti Alto': 'PETROL',
    'Maruti Alto 800': 'PETROL',
    'Maruti Alto K10': 'PETROL',
    'Maruti Baleno': 'PETROL',
    'Maruti BREZZA': 'PETROL',
    'Maruti Celerio': 'PETROL',
    'Maruti Celerio X': 'PETROL',
    'Maruti Ciaz': 'PETROL',
    'Maruti Dzire': 'PETROL',
    'Maruti Eeco': 'PETROL',
    'Maruti Ertiga': 'PETROL',
    'Maruti IGNIS': 'PETROL',
    'Maruti New Wagon-R': 'PETROL',
    'Maruti OMNI E': 'PETROL',
    'Maruti S CROSS': 'PETROL',
    'Maruti S PRESSO': 'PETROL',
    'Maruti Ritz': 'DIESEL',
    'Maruti Swift': 'PETROL',
    'Maruti Swift Dzire': 'DIESEL',
    'Maruti Vitara Brezza': 'DIESEL',
    'Maruti Wagon R': 'PETROL',
    'Maruti Wagon R 1.0': 'PETROL',
    'Maruti Wagon R Stingray': 'PETROL',
    'Maruti XL6': 'PETROL',
    'Maruti Zen Estilo': 'PETROL',
    'MG HECTOR': 'DIESEL',
    'MG HECTOR PLUS': 'DIESEL',
    'Nissan MAGNITE': 'PETROL',
    'Nissan Micra': 'DIESEL',
    'Nissan Micra Active': 'PETROL',
    'Nissan Sunny': 'DIESEL',
    'Nissan Terrano': 'DIESEL',
    'Renault Captur': 'DIESEL',
    'Renault Duster': 'DIESEL',
    'Renault Kiger': 'PETROL',
    'Renault Kwid': 'PETROL',
    'Renault Pulse': 'DIESEL',
    'Renault TRIBER': 'PETROL',
    'Skoda KUSHAQ': 'PETROL',
    'Skoda Octavia': 'PETROL',
    'Skoda RAPID': 'PETROL',
    'Skoda SLAVIA': 'PETROL',
    'Tata ALTROZ': 'PETROL',
    'Tata Bolt': 'DIESEL',
    'Tata Harrier': 'DIESEL',
    'Tata Hexa': 'DIESEL',
    'Tata NEXON': 'DIESEL',
    'Tata PUNCH': 'PETROL',
    'Tata Safari': 'DIESEL',
    'Tata TIGOR': 'PETROL',
    'Tata TIAGO NRG': 'PETROL',
    'Tata Tiago': 'PETROL',
    'Tata Zest': 'DIESEL',
    'Toyota Camry': 'PETROL',
    'Toyota Corolla Altis': 'PETROL',
    'Toyota Etios': 'DIESEL',
    'Toyota Etios Liva': 'DIESEL',
    'Toyota Fortuner': 'DIESEL',
    'Toyota Glanza': 'PETROL',
    'Toyota Innova': 'DIESEL',
    'Toyota Innova Crysta': 'DIESEL',
    'Toyota URBAN CRUISER': 'PETROL',
    'Toyota YARIS': 'PETROL',
    'Volkswagen Ameo': 'DIESEL',
    'Volkswagen Jetta': 'DIESEL',
    'Volkswagen Polo': 'PETROL',
    'Volkswagen T-ROC': 'PETROL',
    'Volkswagen TAIGUN': 'PETROL',
    'Volkswagen TIGUAN': 'DIESEL',
    'Volkswagen Vento': 'DIESEL',
}

CAR_DRIVE_MAPPING = {
    'BMW 3 Series': 'Automatic',
    'Datsun Go': 'Manual',
    'Datsun Go Plus': 'Manual',
    'Datsun Redi Go': 'Manual',
    'Ford Ecosport': 'Manual',
    'Ford Endeavour': 'Automatic',
    'Ford FREESTYLE': 'Manual',
    'Ford Figo Aspire': 'Manual',
    'Ford New Figo': 'Manual',
    'Honda Accord': 'Automatic',
    'Honda Amaze': 'Manual',
    'Honda BR-V': 'Manual',
    'Honda Brio': 'Manual',
    'Honda CRV': 'Automatic',
    'Honda City': 'Automatic',
    'Honda Civic': 'Automatic',
    'Honda Jazz': 'Manual',
    'Honda WR-V': 'Manual',
    'Hyundai ALCAZAR': 'Automatic',
    'Hyundai AURA': 'Manual',
    'Hyundai Creta': 'Automatic',
    'Hyundai Elite i20': 'Manual',
    'Hyundai Eon': 'Manual',
    'Hyundai GRAND I10 NIOS': 'Manual',
    'Hyundai Grand i10': 'Manual',
    'Hyundai NEW I20': 'Manual',
    'Hyundai NEW I20 N LINE': 'Manual',
    'Hyundai NEW SANTRO': 'Manual',
    'Hyundai New Elantra': 'Automatic',
    'Hyundai Santro Xing': 'Manual',
    'Hyundai Sonata': 'Automatic',
    'Hyundai Tucson New': 'Automatic',
    'Hyundai VENUE': 'Manual',
    'Hyundai Verna': 'Automatic',
    'Hyundai Xcent': 'Manual',
    'Hyundai i10': 'Manual',
    'Hyundai i20': 'Manual',
    'Hyundai i20 Active': 'Manual',
    'Jeep Compass': 'Manual',
    'Jeep GRAND CHEROKEE': 'Automatic',
    'KIA CARENS': 'Automatic',
    'KIA SELTOS': 'Automatic',
    'KIA SONET': 'Manual',
    'Mahindra BOLERO NEO': 'Manual',
    'Mahindra Bolero': 'Manual',
    'Mahindra KUV 100 NXT': 'Manual',
    'Mahindra Kuv100': 'Manual',
    'Mahindra MARAZZO': 'Manual',
    'Mahindra Scorpio': 'Manual',
    'Mahindra Thar': 'Manual',
    'Mahindra TUV 300 PLUS': 'Manual',
    'Mahindra TUV300': 'Manual',
    'Mahindra XUV 3OO': 'Manual',
    'Mahindra XUV500': 'Manual',
    'Mahindra XUV700': 'Automatic',
    'Maruti A Star': 'Manual',
    'Maruti Alto': 'Manual',
    'Maruti Alto 800': 'Manual',
    'Maruti Alto K10': 'Manual',
    'Maruti Baleno': 'Manual',
    'Maruti BREZZA': 'Manual',
    'Maruti Celerio': 'Manual',
    'Maruti Celerio X': 'Manual',
    'Maruti Ciaz': 'Manual',
    'Maruti Dzire': 'Manual',
    'Maruti Eeco': 'Manual',
    'Maruti Ertiga': 'Manual',
    'Maruti IGNIS': 'Manual',
    'Maruti New Wagon-R': 'Manual',
    'Maruti OMNI E': 'Manual',
    'Maruti S CROSS': 'Manual',
    'Maruti S PRESSO': 'Manual',
    'Maruti Ritz': 'Manual',
    'Maruti Swift': 'Manual',
    'Maruti Swift Dzire': 'Manual',
    'Maruti Vitara Brezza': 'Manual',
    'Maruti Wagon R': 'Manual',
    'Maruti Wagon R 1.0': 'Manual',
    'Maruti Wagon R Stingray': 'Manual',
    'Maruti XL6': 'Manual',
    'Maruti Zen Estilo': 'Manual',
    'MG HECTOR': 'Manual',
    'MG HECTOR PLUS': 'Automatic',
    'Nissan MAGNITE': 'Manual',
    'Nissan Micra': 'Manual',
    'Nissan Micra Active': 'Manual',
    'Nissan Sunny': 'Manual',
    'Nissan Terrano': 'Manual',
    'Renault Captur': 'Manual',
    'Renault Duster': 'Manual',
    'Renault Kiger': 'Automatic',
    'Renault Kwid': 'Manual',
    'Renault Pulse': 'Manual',
    'Renault TRIBER': 'Manual',
    'Skoda KUSHAQ': 'Manual',
    'Skoda Octavia': 'Automatic',
    'Skoda RAPID': 'Manual',
    'Skoda SLAVIA': 'Manual',
    'Tata ALTROZ': 'Manual',
    'Tata Bolt': 'Manual',
    'Tata Harrier': 'Manual',
    'Tata Hexa': 'Automatic',
    'Tata NEXON': 'Manual',
    'Tata PUNCH': 'Manual',
    'Tata Safari': 'Manual',
    'Tata TIGOR': 'Manual',
    'Tata TIAGO NRG': 'Manual',
    'Tata Tiago': 'Manual',
    'Tata Zest': 'Manual',
    'Toyota Camry': 'Automatic',
    'Toyota Corolla Altis': 'Automatic',
    'Toyota Etios': 'Manual',
    'Toyota Etios Liva': 'Manual',
    'Toyota Fortuner': 'Automatic',
    'Toyota Glanza': 'Manual',
    'Toyota Innova': 'Manual',
    'Toyota Innova Crysta': 'Automatic',
    'Toyota URBAN CRUISER': 'Manual',
    'Toyota YARIS': 'Manual',
    'Volkswagen Ameo': 'Manual',
    'Volkswagen Jetta': 'Automatic',
    'Volkswagen Polo': 'Manual',
    'Volkswagen T-ROC': 'Automatic',
    'Volkswagen TAIGUN': 'Automatic',
    'Volkswagen TIGUAN': 'Automatic',
    'Volkswagen Vento': 'Manual',
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


# --- CALLBACK FUNCTION to automatically update Specs (Triggered by Car Model change) ---
def update_specs_on_model_change():
    """Reads the selected car name and updates car_type, fuel, and drive state variables."""
    car_name = st.session_state.input_car_name
    
    # 1. Update Car Body Type
    new_type = CAR_TYPE_MAPPING.get(car_name, st.session_state.car_type)
    st.session_state.car_type = new_type
    
    # 2. Update Fuel Type
    new_fuel = CAR_FUEL_MAPPING.get(car_name, st.session_state.fuel)
    st.session_state.fuel = new_fuel

    # 3. Update Transmission Type
    new_drive = CAR_DRIVE_MAPPING.get(car_name, st.session_state.drive)
    st.session_state.drive = new_drive


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

# Custom CSS for high contrast and modern look
st.markdown("""
<style>
.main-header { 
    font-size: 3.5em; 
    font-weight: 800; 
    color: #007BFF; 
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
    background-color: #e6f2ff; 
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}
div[data-testid="stMetricValue"] {
    font-size: 2.5em; 
    color: #004085; 
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

# --- 3. INPUT GATHERING ---

# Initialize session state for all inputs and prediction result
initial_car_name = CAR_NAMES[0]
initial_car_type = CAR_TYPE_MAPPING.get(initial_car_name, TYPE_OPTIONS[0])
initial_fuel = CAR_FUEL_MAPPING.get(initial_car_name, FUEL_OPTIONS[0])
initial_drive = CAR_DRIVE_MAPPING.get(initial_car_name, DRIVE_OPTIONS[0])


input_defaults = {
    'car_name': initial_car_name, 'year': 2018, 'owner': 1, 'distance': 50000, 
    'fuel': initial_fuel, 'car_type': initial_car_type, 
    'drive': initial_drive, 
    'location': LOCATION_OPTIONS[0], 'last_prediction': None
}
for key, default in input_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

tab1, tab2, tab3 = st.tabs(["*1. Identity", "2. Specs", "3. Predict*"])

with tab1:
    st.subheader("🚘 Car Identity Details")
    
    st.session_state.car_name = st.selectbox(
        'Select Car Model', 
        options=CAR_NAMES, 
        key='input_car_name', 
        index=CAR_NAMES.index(st.session_state.car_name), 
        help="Choose the exact model name.",
        on_change=update_specs_on_model_change # NEW CALLBACK
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
        
        # Dynamic index for Fuel Type
        current_fuel = st.session_state.fuel
        # We handle case sensitivity here, as the Fuel options are 'Petrol'/'Diesel'/etc
        # but the derived map uses 'PETROL'/'DIESEL'/etc.
        current_fuel_index = 0
        try:
             # Find the index of the capitalized fuel type in the title-case options list
            current_fuel_index = FUEL_OPTIONS.index(current_fuel.title())
        except ValueError:
             # Fallback: try to find exact match
             try:
                current_fuel_index = FUEL_OPTIONS.index(current_fuel)
             except ValueError:
                pass # Use default index 0 if not found
            
        st.session_state.fuel = st.selectbox('Fuel Type', options=FUEL_OPTIONS, key='input_fuel', index=current_fuel_index, help="Is it Petrol, Diesel, LPG, or CNG?")
        
        if st.session_state.distance > 200000:
            st.warning("⚠️ High mileage detected. This may lower the predicted price.")
        elif kms_per_year < 5000 and years_old > 3:
            st.info("ℹ️ Very low mileage for the car's age. (Positive factor)")

    with col_t2_2:
        # Dynamic index for Car Body Type
        current_car_type = st.session_state.car_type
        try:
            current_index = TYPE_OPTIONS.index(current_car_type)
        except ValueError:
            current_index = 0
            
        st.session_state.car_type = st.selectbox(
            'Car Body Type', 
            options=TYPE_OPTIONS, 
            key='input_type', 
            index=current_index, 
            help="e.g., Hatchback, Sedan, SUV."
        )

        # Dynamic index for Transmission Type
        current_drive = st.session_state.drive
        try:
            # We handle case sensitivity here, as the Drive options are 'Manual'/'Automatic'
            current_drive_index = DRIVE_OPTIONS.index(current_drive.title())
        except ValueError:
            # Fallback: try to find exact match
            try:
                current_drive_index = DRIVE_OPTIONS.index(current_drive)
            except ValueError:
                current_drive_index = 0
            
        st.session_state.drive = st.selectbox('Transmission Type', options=DRIVE_OPTIONS, key='input_drive', index=current_drive_index, help="Manual or Automatic.")
    
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
