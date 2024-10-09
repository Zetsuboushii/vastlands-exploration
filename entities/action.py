class Action:
    def __init__(self, name: str, legendary: bool, cooldown: tuple[int, str], range: str, aoe: str, hitbonus: int,
                 savereq: tuple[str, int], damage: list[tuple[str, str]], effects: list[tuple[str, str]]):
        self.name = name
        self.legendary = legendary
        self.cooldown = cooldown
        self.range = range
        self.aoe = aoe
        self.hitbonus = hitbonus
        self.savereq = savereq
        self.damage = damage
        self.effects = effects


def json_to_action(data):
    return Action(
        name=data.get("name", ""),
        legendary=data.get("legendary", False),
        cooldown=data.get("cooldown", (0, "")),
        range=data.get("range", ""),
        aoe=data.get("aoe", ""),
        hitbonus=data.get("hitbonus", 0),
        savereq=data.get("savereq", (0, "")),
        damage=data.get("damage", []),
        effects=data.get("effects", [])
    )
