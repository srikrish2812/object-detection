import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import cv2
import numpy as np
from PIL import Image
import io
import base64
import uvicorn
import os

# Create FastAPI app
app = FastAPI(title="Object Detection API")

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))

    # Perform detection
    results = model(img)

    # Process results
    img_np = np.array(img)
    for det in results.xyxy[0]:  # Process each detection
        x1, y1, x2, y2, conf, cls = det.tolist()
        label = f"{results.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(img_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img_np, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert back to image
    processed_img = Image.fromarray(img_np)

    # Convert to base64 for display
    buffered = io.BytesIO()
    processed_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Save processed image to static folder (optional)
    output_filename = f"static/processed_{file.filename}"
    processed_img.save(output_filename)

    # Created/Modified files during execution:
    print(f"Created: {output_filename}")

    return {"image": f"data:image/jpeg;base64,{img_str}",
            "saved_path": output_filename,
            "detections": results.pandas().xyxy[0].to_dict(orient="records")}

# For direct execution
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)