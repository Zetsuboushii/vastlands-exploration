from typing import Dict, Any

from entities import Entity


class Marker(Entity):
    name: str
    lat: float
    long: float

    def __init__(self, name: str, lat: float, long: float):
        self.name = name
        self.lat = lat
        self.long = long

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "Marker":
        return Marker(data["m_name"], data["m_lat"], data["m_lng"])