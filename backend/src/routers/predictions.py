import os
import shutil
import zipfile
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

UPLOAD_DIRECTORY = "/app/data/uploads"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .zip file.")

    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)

    try:
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)

        with zipfile.ZipFile(file_location, 'r') as zip_ref:
            extract_path = os.path.join(UPLOAD_DIRECTORY, os.path.splitext(file.filename)[0])
            if not os.path.exists(extract_path):
                os.makedirs(extract_path)
            zip_ref.extractall(extract_path)

        os.remove(file_location)

        return JSONResponse(status_code=200, content={"message": "File uploaded and decompressed successfully.", "path": extract_path})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
