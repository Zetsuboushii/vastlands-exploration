import json
import os
import re

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

host = os.environ["MONGODB_HOST"]
port = int(os.environ["MONGODB_PORT"])
username = os.environ["MONGODB_USER"]
password = os.environ["MONGODB_PASSWORD"]
database_name = os.environ["MONGODB_DATABASE"]

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