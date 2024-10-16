import os
import shutil
import typing
from typing import List, Dict, Any

import pandas as pd
import requests

from entities import Entity

API_URL = "https://zetsuboushii.github.io/tome-of-the-vastlands/api/"


def get_all_data(faergria_map_url: str):
    endpoints = {
        "characters_data": "characters.json",
        "general_data": "data.json",
        "races_data": "races.json",
        "places_data": "places.json",
        "actions_data": "actions.json",
        "enemies_data": "enemies.json",
    }
    endpoints = {key: API_URL + endpoint for key, endpoint in endpoints.items()}
    faergria_map_endpoints = {
        "markers_data": "/markers"
    }
    faergria_endpoints = {key: faergria_map_url + endpoint for key, endpoint in faergria_map_endpoints.items()}
    data = {key: requests.get(endpoint).json() for key, endpoint in endpoints.items()}
    data |= {key: requests.get(endpoint).json()["data"] for key, endpoint in faergria_endpoints.items()}
    fetch_faergria_map(faergria_map_url)
    return data


def get_df_from_endpoint_data(endpoint_data: List[Dict[str, Any]], type: typing.Type[Entity]):
    objects = [type.from_json(object_data) for object_data in endpoint_data]
    dicts = [object.__dict__ for object in objects]
    df = pd.DataFrame(dicts)
    return df

def fetch_faergria_map(faegria_map_url: str):
    data_dir = os.path.join(os.curdir, "data")
    map_location = os.path.join(data_dir, "faergria-map.png")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(map_location):
        response = requests.get(faegria_map_url + "/src/assets/maps/faergria.png", stream=True)
        with open(map_location, "wb") as file:
            shutil.copyfileobj(response.raw, file)