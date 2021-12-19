from typing import List

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

import uvicorn

app = FastAPI()

@app.get('/files/')
def create_files(files: List[bytes] = File(...)):
    return {"file_sizes": [len(file) for file in files]}


@app.post('/uploadfiles/')
def create_upload_files(files: List[UploadFile] = File(...)):
    return {"filenames": [file.filename for file in files]}

@app.get('/')
def main():
    content = """
    <body>
    <form action="/files/" enctype="multipart/form-data" method="post">
    <input name="files" type="file" multiple>
    <input type="submit">
    </form>
    <form action="/uploadfiles/" enctype="multipart/form-data" method="post">
    <input name="files" type="file" multiple>
    <input type="submit">
    </form>
    </body>    
    """
    return HTMLResponse(content=content)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)