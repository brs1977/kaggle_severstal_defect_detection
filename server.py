import cv2
import os
import io
import numpy as np
from predictor import Predictor
from typing import List
from fastapi import Request, FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
model_path = 'logs/fpn_timm-effb3/trace/traced-best-forward.pth'
predictor = Predictor(model_path)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):    
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.COLOR_BGR2RGB) 
    image = predictor.predict(image)
    return StreamingResponse(io.BytesIO(image), media_type="image/jpeg", 
        headers={'Content-Disposition': 'inline; filename="%s"' %(os.path.basename(file.filename),)})