from entities import Entity


class Race(Entity):
    def __init__(self, name: str, ageavg: int, domains: list[str]):
        self.name = name
        self.ageavg = ageavg
        self.domains = domains

    @staticmethod
    def from_json(data):
        return Race(
            name=data.get('name', ''),
            ageavg=data.get('ageavg', ''),
            domains=data.get('domains', [])
        )
