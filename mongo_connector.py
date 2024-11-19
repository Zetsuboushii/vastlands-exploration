import json
import os
import re

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database


def _get_env_var(key: str, default=None) -> str:
    value = os.environ.get(key, default)
    return value if value is not None and value != "" else default


host = _get_env_var("MONGODB_HOST", "localhost")
port = int(_get_env_var("MONGODB_PORT", "27017"))
username = _get_env_var("MONGODB_USER", "mongo")
password = _get_env_var("MONGODB_PASSWORD", "abc")
database_name = _get_env_var("MONGODB_DATABASE", "vastlands-exploration")

client = MongoClient(host, port, username=username, password=password)
db: Database = client[database_name]
tierlists: Collection = db.tierlists

def _parse_tierlist_name(name: str):
    match = re.match(r"tierlist_([\w-]+)_(\d+)\.json", name)
    return match.groups() if match else (None, None)

def load_tierlists_into_db():
    tierlists_dir = os.path.join("data", "tierlists")
    tierlist_dicts = []
    for filename in os.listdir(tierlists_dir):
        path = os.path.join(tierlists_dir, filename)
        with open(path, "r") as f:
            data = json.load(f)
        name, session_number = _parse_tierlist_name(filename)
        data["author"] = name
        data["sessionNr"] = session_number
        tierlist_dicts.append(data)
    tierlists.insert_many(tierlist_dicts)

def fetch_tierlists():
    return tierlists.find()