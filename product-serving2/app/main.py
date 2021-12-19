import io
import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any

from datetime import datetime

from model import efficientnet_b0 
from predict import run, get_class_model, get_detect_model, predict_from_image_byte, get_big_model, predict_big_class
from utils import get_config

app = FastAPI()

Det_Model = get_detect_model()
Big_Model = get_big_model()
orders = []

class Food(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str

class Order(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    products: List[Food] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def bill(self):
        return sum([product.price for product in self.products])

    def add_product(self, product: Food):
        if product.id in [existing_product.id for existing_product in self.products]:
            return self

        self.products.append(product)
        self.updated_at = datetime.now()
        return self

class InferenceImageProduct(Food):
    name: str = Optional[str]
    # result: Optional[List]

class DetectedImage(Food):
    name: str = Optional[str]
    xywh: Optional[List]
    result: Optional[List]

@app.get("/order/{order_id}", description="Order 정보를 가져옵니다")
async def get_order(order_id: UUID) -> Union[Order, dict]:
    order = get_order_by_id(order_id=order_id)
    if not order:
        return {"message": "주문 정보를 찾을 수 없습니다"}
    return order

def get_order_by_id(order_id: UUID) -> Optional[Order]:
    return next((order for order in orders if order.id == order_id), None)


@app.get("/order", description="주문 리스트를 가져옵니다")
async def get_orders() -> List[Order]:
    return orders

@app.post("/detect", description="Detecting...")
async def make_order(files: List[UploadFile] = File(...)):
    xywhs = []
    for file in files:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).resize((640,640))
        img = img.convert('RGB')

        inference_result = run(Det_Model, img0=np.array(img))

        for xywh in inference_result:
            new_food = DetectedImage(name='detection',xywh=xywh)
            xywhs.append(new_food)

    new_order = Order(products=xywhs)
    orders.append(new_order)
    return new_order

@app.post("/order", description="Big class Classifying...")
async def update_order(file: bytes = File(...)):    
    image_bytes = file
    inference_result = predict_big_class(model=Big_Model, img=image_bytes)
    # InferenceImageProduct Class 생성해서 product로 정의
    product = InferenceImageProduct(name=inference_result)
  
    # orders.append(new_order)
    return product

# @app.post("/order", description="주문을 요청합니다")
# async def classify(files: List[UploadFile] = File(...),
#                      model: efficientnet_b0 = Depends(get_class_model()),
#                      config: Dict[str, Any] = Depends(get_config)):
#     products = []
#     for file in files:
#         image_bytes = await file.read()
#         inference_result = predict_from_image_byte(model=model, image_bytes=image_bytes, config=config)
#         product = InferenceImageProduct(name = '김치',result=inference_result)
#         products.append(product)

#     new_order = Order(products=products)
#     orders.append(new_order)
#     return new_order
