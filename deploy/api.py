from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.enums import AnimalType
from src.predict_dummy import predict
import os

from dotenv import load_dotenv
load_dotenv()


app = FastAPI(title="Image Generation API")

class PredictionRequest(BaseModel):
    prompt: str
    animal_type: AnimalType 

class PredictionResponse(BaseModel):
    image_path: str

@app.post("/predict", response_model=PredictionResponse)
def generate_image(request: PredictionRequest):
    try:
        image_path = predict(request.prompt, request.animal_type)

        if not os.path.exists(image_path):
            raise HTTPException(status_code=500, detail="Image generation failed or file not found.")

        return PredictionResponse(image_path=image_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")
