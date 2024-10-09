class Enemy:
    def __init__(self, name: str, boss: bool, bosstitle: str, type: str, hp: int, ac: int, movement: list[str],
                 str: int, dex: int, con: int, int: int, wis: int, cha: int, weaknesses: list[str],
                 resistances: list[str], immunities: list[str], actions: list[str]):
        self.name = name,
        self.boss = boss,
        self.bosstitle = bosstitle,
        self.type = type,
        self.hp = hp,
        self.ac = ac,
        self.movement = movement,
        self.str = str,
        self.dex = dex,
        self.con = con,
        self.int = int,
        self.wis = wis,
        self.cha = cha,
        self.weaknesses = weaknesses,
        self.resistances = resistances,
        self.immunities = immunities,
        self.actions = actions

def json_to_enemy(data):
    return Enemy(
        name=data.get('name',""),
        boss=data.get('boss',False),
        bosstitle=data.get('bosstitle',""),
        type=data.get('type',""),
        hp=data.get('hp',0),
        ac=data.get('ac',0),
        movement=data.get('movement',0),
        str=data.get('str',0),
        dex=data.get('dex',0),
        con=data.get('con',0),
        int=data.get('int',0),
        wis=data.get('wis',0),
        cha=data.get('cha',0),
        weaknesses=data.get('weaknesses',[]),
        resistances=data.get('resistances',[]),
        immunities=data.get('immunities',[]),
        actions=data.get('actions',[])
    )
