from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get('/')
def read_root():
    return {'hello: world'}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)