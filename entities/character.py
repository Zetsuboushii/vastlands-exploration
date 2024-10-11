from entities import Entity


class Character(Entity):
    def __init__(self, name: str, surname: str, title: str, race: str, sex: str, birthday: str,
                 height: str,
                 functions: list[str], character_class: str, subclasses: list[str],
                 masterclass: str, homes: list[str],
                 alignment: str, status: str, relationships: list[tuple[str, str]], lover: str):
        self.name = name
        self.surname = surname
        self.title = title
        self.race = race
        self.sex = sex
        self.birthday = birthday
        self.height = height
        self.function = functions
        self.character_class = character_class
        self.subclasses = subclasses
        self.masterclass = masterclass
        self.homes = homes
        self.alignment = alignment
        self.status = status
        self.relationships = relationships
        self.lover = lover

    @staticmethod
    def from_json(data):
        return Character(
            name=data.get('name', ''),
            surname=data.get('surname', ''),
            title=data.get('title', ''),
            race=data.get('race', ''),
            sex=data.get('sex', ''),
            birthday=data.get('birthday', ''),
            height=data.get('height', ''),
            functions=data.get('functions', []),
            character_class=data.get('class', ''),
            subclasses=data.get('subclasses', []),
            masterclass=data.get('masterclass', ''),
            homes=data.get('homes', []),
            alignment=data.get('alignment', ''),
            status="am Leben" if data.get('status', None) == "true" else "tot" if data.get('status',
                                                                                           None) == "false" else data.get(
                'status', None),
            relationships=data.get('relationships', []),
            lover=data.get('lover', '')
        )
