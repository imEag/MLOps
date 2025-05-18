from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the ML API"}

# Placeholder for API routers
from .routers import models
app.include_router(models.router) 