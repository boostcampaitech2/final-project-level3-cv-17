from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get('/users/{user_id}')
def get_user(user_id):
    return {'user_id': user_id}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)