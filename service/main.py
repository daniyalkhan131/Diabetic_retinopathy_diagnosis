from fastapi import FastAPI
import onnxruntime as rt
from service.api.api import main_router

app = FastAPI (project_name='Diabetic-Retinopathy-Detection')
app.include_router(main_router)
 

providers= ['CPUExecutionProvider']
output_path = "service/core/logic/inceptn_dia.onnx"


m = rt.InferenceSession(output_path, providers=providers)

@app.get("/")
async def root():
    return {'welcome to diabetic retinopathy diagnosis, images': 'grading' }


