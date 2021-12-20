import io
import numpy as np
import torch
from PIL import Image
import torch 

from fastapi import FastAPI, UploadFile, File
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any

from datetime import datetime

from model import efficientnet_b0 
from predict import run, load_small_model, load_det_model, load_big_model, load_quantity_model, get_big_prediction, get_small_predicitions, get_quantity_prediction
from utils import get_config, transform_image

import uvicorn

app = FastAPI()

Det_Model = load_det_model()
Big_Model = load_big_model()
Quantity_Model = load_quantity_model()

# Small_Model = load_small_model()
orders = []

class Food(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    big_label: str
    small_label: str
    xyxy: list
    info: dict

class Intake(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    Foods: List[Food] = Field(default_factory=list)
    Total: Dict
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def add_food(self, food: Food):
        # add_product는 Product를 인자로 받아서, 해당 id가 이미 존재하는지 체크 => 없다면 products 필드에 추가
        # 업데이트할 때 updated_at을 현재 시각으로 업데이트
        if food.id in [existing_product.id for existing_product in self.products]:
            return self

        self.Foods.append(food)
        self.updated_at = datetime.now()
        return self

@app.post("/intake", description="Detecting...")
async def make_order(files: List[UploadFile] = File(...)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for file in files:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        img_np = np.array(img)
        h, w, c = img_np.shape
        print(f'img shape : {w, h}')

        xyxys = run(Det_Model, img0=np.array(img.resize((640, 640))))
        total = {'carbohydrate': 0, 'protein': 0, 'fat': 0, 'sugar': 0, 'kcal': 0}
        foods = []
        for xyxy in xyxys:            
            x1, y1 = int(w*xyxy[0]), int(h*xyxy[1])
            x2, y2 = int(w*xyxy[2]), int(h*xyxy[3])

            cropped_img = img.crop((x1, y1, x2, y2))   
            cropped_img = transform_image(cropped_img).to(device)

            big_label = get_big_prediction(model=Big_Model, img=cropped_img)
            food_info = get_small_predicitions(bigclass= big_label, model=load_small_model(big_label), img=cropped_img)
            quantity = get_quantity_prediction(model=Quantity_Model, img=cropped_img) + 1
            
            name, carbohydrate, protein, fat, sugar, kcal = food_info
            c, p, f, s, k = [round(float(v) * quantity * 0.2, 2) for v in [carbohydrate, protein, fat, sugar, kcal]]
            info = {'quantity': quantity, 'carbohydrate': c, 'protein': p, 'fat': f, 'sugar': s, 'kcal': k}
            for k, v in zip(total, [c, p, f, s, k]):
                total[k] += v

            food = Food(big_label=big_label, small_label=name, xyxy=[x1, y1, x2, y2], info=info)
            foods.append(food)

        new_order = Intake(Foods=foods, Total=total)
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
