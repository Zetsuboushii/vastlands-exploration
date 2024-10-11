import typing
from typing import List, Dict, Any

import pandas as pd
import requests

from entities import Entity

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
    return data


def get_df_from_endpoint_data(endpoint_data: List[Dict[str, Any]], type: typing.Type[Entity]):
    objects = [type.from_json(object_data) for object_data in endpoint_data]
    dicts = [object.__dict__ for object in objects]
    df = pd.DataFrame(dicts)
    return df
