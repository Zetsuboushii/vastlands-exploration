class Race:
    def __init__(self, name: str, ageavg: int, domains: list[str]):
        self.name = name
        self.ageavg = ageavg
        self.domains = domains


def json_to_race(data):
    return Race(
        name=data.get('name', ''),
        ageavg=data.get('ageavg', ''),
        domains=data.get('domains', [])
    )
