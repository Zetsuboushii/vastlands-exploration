import requests
import pandas as pd

from entities.action import json_to_action
from entities.character import json_to_character
from entities.place import json_to_place
from entities.race import json_to_race

API_URL = "https://zetsuboushii.github.io/tome-of-the-vastlands/api/"

def get_all_data():
    endpoints = {
        "characters_data": "characters.json",
        "general_data": "data.json",
        "races_data": "races.json",
        "places_data": "places.json",
        "actions_data": "actions.json",
        "enemies_data": "enemies.json",
    }
    data = {key: requests.get(API_URL + endpoint).json() for key, endpoint in endpoints.items()}
    return data["characters_data"], data["general_data"], data["races_data"], data["places_data"], data["actions_data"], data["enemies_data"]

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

def get_places_df(places_data):
    places_objects = [json_to_place(place_data) for place_data in places_data]
    place_dicts = [place.__dict__ for place in places_objects]
    df_places = pd.DataFrame(place_dicts)
    return df_places

def get_actions_df(actions_data):
    actions_objects = [json_to_action(action_data) for action_data in actions_data]
    action_dicts = [action.__dict__ for action in actions_objects]
    df_actions = pd.DataFrame(action_dicts)
    return df_actions

def get_enemies_df(enemies_data):
    enemies_objects = [json_to_action(enemies_data) for enemies_data in enemies_data]
    enemy_dicts = [enemy.__dict__ for enemy in enemies_objects]
    df_enemies = pd.DataFrame(enemy_dicts)
    return df_enemies