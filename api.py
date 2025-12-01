import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.utils import load_model, prepare_input_data

app = FastAPI()
model = load_model("models/catboost_model.cbm")


class ModelRequestData(BaseModel):
    postcode: str
    total_square: float
    rooms: int
    floor: int
    city: str
    district: str
    street_house: str


class Result(BaseModel):
    result: float


@app.get("/health")
def health():
    return JSONResponse(content={"message": "It's alive!"}, status_code=200)


@app.get("/predict_get", response_model=Result)
def predict_get(
    postcode: str,
    total_square: float,
    rooms: int,
    floor: int,
    city: str,
    district: str,
    street_house: str,
):
    input_df = prepare_input_data(
        city=city,
        district=district,
        street_house=street_house,
        postcode=postcode,
        total_square=total_square,
        rooms=rooms,
        floor=floor,
    )
    pred = float(model.predict(input_df)[0])
    return Result(result=pred)


@app.post("/predict_post", response_model=Result)
def predict_post(data: ModelRequestData):
    input_df = prepare_input_data(**data.dict())
    pred = float(model.predict(input_df)[0])
    return Result(result=pred)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)