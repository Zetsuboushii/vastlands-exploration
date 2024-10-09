class Place:
    def __init__(self, name: str, subtitle: str, supplement: str, natlang: list[str], demography: str,
                 has_init: list[str], placetype: str, system: str, leader: tuple[str, str], capital: str):
        self.name = name
        self.subtitle = subtitle
        self.supplement = supplement
        self.natlang = natlang
        self.demography = demography
        self.has_init = has_init
        self.placetype = placetype
        self.system = system
        self.leader = leader
        self.capital = capital

def json_to_place(data):
    return Place(
        name=data.get("name",""),
        subtitle=data.get("subtitle",""),
        supplement=data.get("supplement",""),
        natlang=data.get("natlang",[]),
        demography=data.get("demography",""),
        has_init=data.get("hasInit",[]),
        placetype=data.get("placetype",""),
        system=data.get("system",""),
        leader=data.get("leader",""),
        capital=data.get("capital","")
    )