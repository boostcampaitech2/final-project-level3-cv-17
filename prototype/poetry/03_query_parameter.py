from typing import Optional
from fastapi import FastAPI
import uvicorn

app = FastAPI()

fake_items_db = [{'item_name': 'Foo'}, {'item_name': "Bar"}, {'item_name': 'Baz'}, {'item_id': '2'} ]

# @app.get('/items/')
# def read_item(skip: int=0, limit: int=10):
#     return fake_items_db[skip: skip + limit]

@app.get('/items/{item_id}')
def read_item(item_id: str, q: Optional[str] = None):
    if q:
        return {'item_id': item_id, 'q': q}
    return {'item_id': item_id}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)