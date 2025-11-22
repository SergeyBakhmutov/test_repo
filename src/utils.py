import pandas as pd
from catboost import CatBoostRegressor

def load_model(model_path: str='models/catboost_model.cbm') -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(model_path)
    return model

def prepare_input_data(city: str, district: str, street_house:str, postcode: str,
                       total_square: float, rooms: int, floor: int ):
    input_data = pd.DataFrame(
        {
            "city": [city],
            "district": [district],
            "street_house": [street_house],
            "postcode": [postcode],
            "total_square": [total_square],
            "rooms": [rooms],
            "floor": [floor],
        }
    )
    return input_data


