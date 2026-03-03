from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import predict

app = FastAPI(title="VTU Dropout Prediction API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api", tags=["prediction"])