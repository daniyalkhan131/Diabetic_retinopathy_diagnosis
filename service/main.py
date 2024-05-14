from fastapi import FastAPI
from service.api.api import main_router

app = FastAPI (project_name='Diabetic-Retinopathy-Detection')
app.include_router(main_router)
 

@app.get("/")
def root():
    return {'welcome to diabetic retinopathy diagnosis, images': 'grading' }


