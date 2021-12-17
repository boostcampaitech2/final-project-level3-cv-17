
import io
import os
import numpy as np
from PIL import Image
from os.path import join
from pathlib import Path

import logging
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any

from detect import run, load_det_model
from datetime import datetime

# from app.model import MyEfficientNet, get_model, get_config, predict_from_image_byte
from model import efficientnet_b0
from predict import get_big_prediction, get_small_prediction, get_quantity_prediction, load_big_model, load_small_model, load_quantity_model

import uvicorn


app = FastAPI()

MODEL_DIR_PATH = os.path.join(Path(__file__).parent.parent, "models")

Det_Model = load_det_model(weights=join(MODEL_DIR_PATH, 'best.torchscript.pt'))
Big_Model = load_big_model()
Small_Model  = load_small_model()
Quantity_Model = load_quantity_model()


@app.get("/")
def hello_world():
    return {"hello": "world"}

class xywh(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str

class Det(BaseModel):
    id: UUID = Field(default_factory=uuid4) 
    xywhs: List[xywh] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class Product(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    # Field : 모델 스키마 또는 복잡한 Validation 검사를 위해 필드에 대한 추가 정보를 제공할 때 사용
    # uuid : 고유 식별자, Universally Unique Identifier
    # default_factory : Product Class가 처음 만들어질 때 호출되는 함수를 uuid4로 하겠다 => Product 클래스를 생성하면 uuid4를 만들어서 id에 저장
    name: str

class DetectImage(xywh):
    name: str = "inference_image"
    result: Optional[List]

class ClassificationImage(Product):
    name: str = "classification_image"
    result: Optional[Union[int, str, list]]


class Order(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    products: List[Product] = Field(default_factory=list)
    # 최초에 빈 list를 만들어서 저장한다
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def add_product(self, product: Product):
        # add_product는 Product를 인자로 받아서, 해당 id가 이미 존재하는지 체크 => 없다면 products 필드에 추가
        # 업데이트할 때 updated_at을 현재 시각으로 업데이트
        if product.id in [existing_product.id for existing_product in self.products]:
            return self

        self.products.append(product)
        self.updated_at = datetime.now()
        return self


orders = []
# 실무에서는 보통 이 경우에 데이터베이스를 이용해서 주문을 저장하지만, 데이터베이스를 따로 학습하지 않았으므로 In Memory인 리스트에 저장

@app.post("/detect", description="Detecting...")
async def detect(files: List[UploadFile] = File(...)):
    xywhs = []
    for file in files:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).resize((640,640))
        img = img.convert('RGB')

        inference_result = run(Det_Model, img0=np.array(img))
        xywh = DetectImage(result=inference_result)
        xywhs.append(xywh)

    return Det(xywhs=xywhs)

@app.post("/order", description="음식 분류")
async def make_order(file: bytes = File(...)):
    # Depends : 의존성 주입
    # 반복적이고 공통적인 로직이 필요할 때 사용할 수 있음
    # 모델을 Load, Config Load
    # async, Depends 검색해서 또 학습해보기!
    products = []
    
    image_bytes = file
    inference_result = get_big_prediction(model=Big_Model, img=image_bytes)
    # InferenceImageProduct Class 생성해서 product로 정의
    product = ClassificationImage(result=inference_result)
    products.append(product)

    new_order = Order(products=products)
    orders.append(new_order)
    return new_order

@app.post("/order/{cls}", description="음식 세부 분류")
async def make_order(cls:str,
                    file: bytes = File(...)
                     ):
    # Depends : 의존성 주입
    # 반복적이고 공통적인 로직이 필요할 때 사용할 수 있음
    # 모델을 Load, Config Load
    # async, Depends 검색해서 또 학습해보기!
    products = []
    image_bytes = file
    inference_result = get_small_prediction(img=image_bytes, model_info=Small_Model, cls=cls)
    # InferenceImageProduct Class 생성해서 product로 정의
    product = ClassificationImage(result=inference_result)
    products.append(product)

    new_order = Order(products=products)
    orders.append(new_order)
    return new_order

@app.post("/quant", description="음식 양 추정")
async def make_order(file: bytes = File(...)):
    # Depends : 의존성 주입
    # 반복적이고 공통적인 로직이 필요할 때 사용할 수 있음
    # 모델을 Load, Config Load
    # async, Depends 검색해서 또 학습해보기!
    products = []
    image_bytes = file
    inference_result = get_quantity_prediction(model=Quantity_Model, img=image_bytes)
    # InferenceImageProduct Class 생성해서 product로 정의
    print(inference_result, type(inference_result))
    product = ClassificationImage(result=inference_result)
    print(product)
    products.append(product)

    new_order = Order(products=products)
    orders.append(new_order)
    return new_order

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# TODO: 주문 구현, 상품 구현, 결제 구현
    # TODO: 주문(Order) = Request
    # TODO: 상품(Product) = 마스크 분류 모델 결과
    # TODO: 결제 = Order.bill
    # 2개의 컴포넌트
# TODO: Order, Product Class 구현
    # TODO: Order의 products 필드로 Product의 List(하나의 주문에 여러 제품이 있을 수 있음)

# TODO: get_orders(GET) : 모든 Order를 가져옴
# TODO: get_order(GET) : order_id를 사용해 Order를 가져옴
# TODO: get_order_by_id : get_order에서 사용할 함수
# TODO: make_order(POST) : model, config를 가져온 후 predict => Order products에 넣고 return
# TODO: update_order(PATCH) : order_id를 사용해 order를 가져온 후, update
# TODO: update_order_by_id : update_order에서 사용할 함수
# TODO: get_bill(GET) : order_id를 사용해 order를 가져온 후, order.bill return