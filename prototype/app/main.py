import io
import numpy as np
from PIL import Image
import torch 

from fastapi import FastAPI, UploadFile, File
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any

from datetime import datetime

from model import efficientnet_b0 
from predict import run, get_class_model, get_detect_model, predict_from_image_byte, get_big_model, predict_big_class
from utils import get_config, transform_image

app = FastAPI()

Det_Model = get_detect_model()
Big_Model = get_big_model()
orders = []

class Food(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    big_label: str
    # small_label: str
    xyxy: list
    # info: dict

class Intake(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    Foods: List[Food] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def add_food(self, food: Food):
        if food.id in [existing_product.id for existing_product in self.products]:
            return self

        self.Foods.append(food)
        self.updated_at = datetime.now()
        return self


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

@app.get("/order", description="주문 리스트를 가져옵니다")
async def get_orders() -> List[Order]:
    return orders

@app.post("/detect", description="Detecting...")
async def make_order(files: List[UploadFile] = File(...)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for file in files:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        img_np = np.array(img)
        h, w, c = img_np.shape

        inference_result = run(Det_Model, img0=np.array(img.resize((640, 640))))

        foods = []
        for xyxy in inference_result:            
            x1, y1 = int(w*xyxy[0]), int(h*xyxy[1])
            x2, y2 = int(w*xyxy[2]), int(h*xyxy[3])

            cropped_img = img.crop((x1, y1, x2, y2))   
            cropped_img = transform_image(cropped_img).to(device)

            big_label = predict_big_class(model=Big_Model, img=cropped_img)
            food = Food(big_label=big_label, xyxy=[x1, y1, x2, y2])
            foods.append(food)

        new_order = Intake(Foods=foods)
        orders.append(new_order)

    return new_order   
  
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