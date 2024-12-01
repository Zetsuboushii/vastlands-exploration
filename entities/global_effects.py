# The global dictionary to store effects data
EFFECT_RATING_DICT = {
    "exhaustion": 5,
    "paralysed": 5,
    "petrified": 5,
    "bleeding": 5,
    "knockback": 3,
    "stunned": 4,
    "poisoned": 3,
    "grappled": 3,
    "prone": 3,
    "double_damage": 4,
    "heal": 2,
    "leap": 2,
    "frightened": 3,
    "blinded": 3,
    "pull": 2,
    "increased_movement": 3,
    "escape": 3,
    "invisible": 4
}

def update_effects(new_data: dict):
    """ Update the global effects dictionary with new data """
    global global_effects
    EFFECT_RATING_DICT.update(new_data)

