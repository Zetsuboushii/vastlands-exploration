from datetime import datetime

CURRENT_DATE = None

def set_current_date(general_data):
    global CURRENT_DATE
    current_ingame_date = general_data.get('currentIngameDate', '')
    CURRENT_DATE = current_ingame_date

def calculate_age(birthday):
    current_year = int(CURRENT_DATE.split('-')[2])
    try:
        parts = birthday.split(".")
        birth_year = int(parts[2])
        return current_year - birth_year
    except (ValueError, IndexError, AttributeError):
        return None
