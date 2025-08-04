'''from fastapi import FastAPI
from routers import predict

app = FastAPI(
    title="Smart Waste Classifier API",
    description="Classifies waste images using a trained deep learning model.",
    version="1.0.0"
)

app.include_router(predict.router, prefix="/api")'''

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from routers import predict

app = FastAPI()

# Static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Home route
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Include the prediction route
app.include_router(predict.router, prefix="/api")
    
