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
            zip_ref.extractall(UPLOAD_DIRECTORY)

        os.remove(file_location)

        return JSONResponse(status_code=200, content={"message": "File uploaded and decompressed successfully.", "path": UPLOAD_DIRECTORY})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

def get_directory_tree(path):
    tree = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            tree.append({
                "title": item,
                "key": item_path,
                "children": get_directory_tree(item_path)
            })
        else:
            tree.append({
                "title": item,
                "key": item_path,
                "isLeaf": True
            })
    return tree

@router.get("/files/")
async def get_files():
    try:
        tree = get_directory_tree(UPLOAD_DIRECTORY)
        return JSONResponse(status_code=200, content=tree)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.delete("/files/")
async def delete_file(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        return JSONResponse(status_code=200, content={"message": "File deleted successfully."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
