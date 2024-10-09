import requests
import pandas as pd
from entities.character import json_to_character
from entities.race import json_to_race

API_URL = "https://zetsuboushii.github.io/tome-of-the-vastlands/api/"

def get_all_data():
    endpoints = {
        "characters_data": "characters.json",
        "general_data": "data.json",
        "races_data": "races.json",
        "places_data": "places.json"
    }

    data = {key: requests.get(API_URL + endpoint).json() for key, endpoint in endpoints.items()}
    return data["characters_data"], data["general_data"], data["races_data"], data["places_data"]

def get_character_df(characters_data):
    character_objects = [json_to_character(char_data) for char_data in characters_data]
    character_dicts = [char.__dict__ for char in character_objects]
    df_characters = pd.DataFrame(character_dicts)
    return df_characters

def get_races_df(races_data):
    races_objects = [json_to_race(race_data) for race_data in races_data]
    race_dicts = [race.__dict__ for race in races_objects]
    df_races = pd.DataFrame(race_dicts)
    return df_races
