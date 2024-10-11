from typing import Dict, Any


class Entity:
    @staticmethod
    def from_json(data: Dict[str, Any]) -> 'Entity':
        raise NotImplementedError()
