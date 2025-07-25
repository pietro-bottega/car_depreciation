import pandas as pd
import numpy as np
import pprint
import os
import re

from os import listdir, getcwd

#----------------------
# Creating dataset
#----------------------

def create_dataset() -> pd.DataFrame:
    """
    Get files extracted from Tabela FIPE by month and consolidates into a single dataframe.

    Parameters:
    - folder: where the files are stored

    Returns:
    - pd.DataFrame: fipe_data
    """

    cwd = os.getcwd()
    cwd_up = os.path.dirname(cwd) 
    path = f"{cwd_up}/data/raw-data/"
    data_files = [path + file for file in listdir(path)]
    fipe_data = pd.concat(map(lambda f: pd.read_csv(f, on_bad_lines='skip'), data_files))
    print("Dataframe created")

    return fipe_data


#----------------------
# Extracting features
#----------------------

def extract_features_from_modelo(modelo: str) -> pd.Series:
    """
    Extract version, engine type, and car category from the 'modelo' column.
    Also strips leading and trailing spaces and extract different informations from the model handle.

    Parameters:
    - modelo: str: The car model description.

    Returns:
    - pd.Series: A pandas Series with values:
        version,
        engine_type,
        car_category,
        transmission_type,
        doors,
        horse_power,
        valves
    """
    
    # Strip leading and trailing spaces
    modelo = modelo.strip()
    
    # Extracting version (e.g., "1.8", "2.0", "16V", "1.6/16V", etc.)
    version_pattern = r'(\b([1-9]\.[0-9]{1,2})\b)'
    
    # Extracting engine type (e.g., "Flex", "Diesel", "Turbo", "V6", "8V", "16V", etc.)
    engine_pattern = r'(V\d+|Eletric|Elétrico|Flex|Gasolina|Diesel|Turbo|V6|16V|8V|12V|MPFI|Bi-Turbo|GDI|CV)'
    
    # Extracting car category (e.g., "Hatch", "Sedan", "SUV", "Picape", "Crossover", etc.)
    category_pattern = r'(SUV|Hatch|Sedan|Picape|Pickup|CD|Wagon|Crossover|Avant|MPV|Sport|Luxury|Coupe|Convertible)'

    # Extracting transmission type (e.g., "Mec", "Aut")
    transmission_pattern = r'(Mec|Aut)'

    # Extracting number of doors (e.g. 4p, 5p..)
    doors_pattern = r'(\b[2-5]p\b)'

    # Extracting horse_power (e.g. 340cv)
    hp_pattern = r'(\d{1,3})(?:cv?|c)\b'

    # Extracting valves (e.g. 8V, 12V) - always multiples of 2 or 4
    valves_pattern = r'(\b([0-9]{1,2})V\b)'
    
    # Extract version
    version = re.findall(version_pattern, modelo)
    version = version[0][0] if version else None

    # Extract engine type and clean up (strip spaces)
    engine = re.findall(engine_pattern, modelo)
    engine = engine[0].strip() if engine else None

    # Extract car category
    category = re.findall(category_pattern, modelo)
    category = category[0].strip() if category else None

    # Extract transmission type
    transmission = re.findall(transmission_pattern, modelo)
    transmission = transmission[0].strip() if transmission else None

    # Extract doors number
    doors = re.findall(doors_pattern, modelo)
    doors = doors[0][0].strip() if doors else None

    # Extract horsepower
    horse_power = re.findall(hp_pattern, modelo)
    horse_power = horse_power[0][0].strip()[:3] if horse_power else None

    # Extract valves
    valves = re.findall(valves_pattern, modelo)
    valves = valves[0][0].strip()[:-1] if valves else None
    
    # Classify engine type based on keywords
    def classify_engine_type(engine: str) -> str:
        if engine:
            engine = engine.strip().lower()
            if 'turbo' in engine:
                return 'Turbo'
            elif 'flex' in engine:
                return 'Flex'
            elif 'diesel' in engine:
                return 'Diesel'
            elif 'electric' in engine or 'elétrico' in engine:
                return 'Electric'
            elif 'v6' in engine or 'v8' in engine:
                return 'V6/V8'
            else:
                return 'Other'
        return None

    # Classify car category based on keywords
    def classify_category(category: str) -> str:
        if category:
            category = category.strip().lower()
            if 'suv' in category:
                return 'SUV'
            elif 'hatch' in category:
                return 'Hatchback'
            elif 'sedan' in category:
                return 'Sedan'
            elif 'picape' in category or 'pickup' in category or 'cd' in category:
                return 'Pickup'
            elif 'crossover' in category:
                return 'Crossover'
            elif 'wagon' in category:
                return 'Wagon'
            elif 'luxury' in category:
                return 'Luxury'
            elif 'coupe' in category:
                return 'Coupe'
            elif 'convertible' in category:
                return 'Convertible'
            elif 'sport' in category:
                return 'Sport'
            elif 'mpv' in category or 'avant' in category:
                return 'MPV'
            else:
                return 'Other'
        return None

    # Classify transmission type based on keywords
    def classify_transmision(transmission: str) -> str:
        if transmission:
            transmission = transmission.strip().lower()
            if 'mec' in transmission:
                return 'Manual'
            elif 'aut' in transmission:
                return 'Auto'
            else:
                return 'Other'
        return None

    # Classify door numbers type based on keywords
    def classify_doors(doors: str) -> str:
        if doors:
            doors = doors.strip().lower()
            if 'mec' in doors:
                return 'Manual'
            elif 'aut' in transmission:
                return 'Auto'
            else:
                return 'Other'
        return None

    # Classify engine type and car category
    engine_type = classify_engine_type(engine)
    car_category = classify_category(category)
    transmission_type = classify_transmision(transmission)


    return pd.Series([version, engine_type, car_category, transmission_type, doors, horse_power, valves])


#----------------------
# RUN PIPELINE
#----------------------

fipe_data = create_dataset()
fipe_data.to_csv("../data/output/fipe_data.csv")

fipe_data[['version', 'engine', 'category', 'transmission_type', 'doors', 'hp', 'valves']] = fipe_data['modelo'].apply(extract_features_from_modelo)
fipe_features = fipe_data[['modelo_id','modelo', 'marca', 'comb', 'version', 'engine', 'category', 'transmission_type', 'doors', 'hp', 'valves']].drop_duplicates()
fipe_features.to_csv("../data/output/fipe_features.csv")