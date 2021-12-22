# 최종 프로젝트 : Do You Know Kimchi?

## Project 개요 
**업로드한 음식 사진을 종류와 양에 따라 영양 성분을 분석하고 김치로 환산하는 웹 서비스**
+ Input : 한식 음식 이미지
+ Output : 음식 영양, 양 정보

## Data
### AI hub [음식 이미지 및 영양정보 텍스트](https://aihub.or.kr/aidata/30747)
* 총 12개의 대분류
* 총 339개의 소분류
### 식품의약품안전처 [식품영양성분 데이터베이스](https://www.foodsafetykorea.go.kr/fcdb/)
* 음식별 1회 제공량 / 탄수화물 / 단백질 / 지방 / 당 함량 정보 custom

## Evaluation
### Detection
* **Food Detection**
mAP: 0.917

### Classification (Recall)
* **양 추정 모델**
val/acc: 0.9312

* **대분류 모델**
val/acc: 0.9474

* **소분류 모델**
각 소분류의 수가 많아서 평균 accuracy로 기입

    | 분류 | val/acc | 분류 | val/acc |
    | -------- | -------- | -------- | -------- |
    |deopbab|0.9612|noodle|0.9283|
    |dumpling|0.9568|rice|0.9084|
    |fried|0.9229|seafood|0.9447|
    |herb|0.9528|stew|0.922|
    |kimchi|0.9402|sushi|0.9461|
    |meat|0.9129|vegetable|0.9652|
    
## Model
### YOLO v5 - small
![](https://i.imgur.com/mAvMai8.png)

* Detect 모델: 음식의 접시 기준으로 Bbox를 표시합니다.

### EfficientNet B0
![](https://i.imgur.com/oBkza6u.png)


* Classification 모델
    * 대분류 모델: 각 음식의 대분류를 예측합니다.
    * 양 추정 모델: 음식의 양을 1~5로 분류합니다.
    * 소분류 모델: 각 음식의 소분류를 예측합니다.




## Structure
```
.
├── README.md
├── classification
│   ├── requirements.txt
│   ├── data
│   ├── custom.csv
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── inference.ipynb
│   └── inference_with_torch_script.ipynb
├── detection
│   └── yolov5
│       ├── utils
│       ├── requirements.txt
│       ├── val.py
│       ├── train.py
│       └── visualization.ipynb
├── intermediate
│   ├── Link.ipynb
│   ├── datasheet.csv
│   └── model.py
├── quantity_est
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── inference.ipynb
├── prototype
│   ├── app
│   │   ├── models
│   │   ├── Makefile
│   │   ├── __main__.py
│   │   ├── frontend.py
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── predict.py
│   │   ├── requirements.txt
│   │   └── utils.py
│   └── assets
│       ├── config.yaml
│       ├── Food_info.csv
│       ├── 김치맨1.jpg
│       ├── 김치맨2.png
│       ├── headerbg.jpg
│       └── NanumSquareB.ttf
└── tools
    ├── interactive_EDA.ipynb
    ├── mean_std.ipynb
    ├── resize_and_write_pool.py
    ├── sampling.py
    ├── sampling2.py
    └── split_test_set.py
```

## Service
### How to Use
키, 몸무게, 활동지수를 입력하면 권장 칼로리가 출력됩니다.
이후 원하는 칼로리 범위와 탄수화물, 단백질, 지방의 비율을 설정하실 수 있습니다.

Choose an image 탭에 자신의 식단 사진을 업로드 하면 모델이 음식의 종류와 영양성분을 추정해서 출력합니다.

마지막으로 당신이 먹은 음식이 몇 김치인지 이미지로 나타납니다.
음식의 칼로리가 과도하게 많거나 적으면 **멈춰 김치맨**이 출력되고 적절하면 **발차기 김치맨**이 출력됩니다.

### Result
![](https://i.imgur.com/TVMpMBZ.png)


## Quick Start

**가상환경에서 실행 시키시는걸 추천합니다.**

1. Makefile 설치
2. Packages
```  
pip install -r requirements.txt
```

3. 현재 위치 변경
```
cd {clone directory}/final-project-level3-cv-17/prototype/app
```
4. 실행
```
make -j 2 app_run
```

---
## Members

|   <div align="center">김주영 </div>	|  <div align="center">오현세 </div> 	|  <div align="center">채유리 </div> 	|  <div align="center">배상우 </div> 	|  <div align="center">최세화 </div>  | <div align="center">송정현 </div> |
|---	|---	|---	|---	|---	|---	|
| <img src="https://avatars.githubusercontent.com/u/61103343?s=120&v=4" alt="0" width="200"/>	|  <img src="https://avatars.githubusercontent.com/u/79178335?s=120&v=4" alt="1" width="200"/> 	|  <img src="https://avatars.githubusercontent.com/u/78344298?s=120&v=4" alt="1" width="200"/> 	|   <img src="https://avatars.githubusercontent.com/u/42166742?s=120&v=4" alt="1" width="200"/>	| <img src="https://avatars.githubusercontent.com/u/43446451?s=120&v=4" alt="1" width="200"/> | <img src="https://avatars.githubusercontent.com/u/68193636?v=4" alt="1" width="200"/> |
|   <div align="center">[Github](https://github.com/JadeKim042386)</div>	|   <div align="center">[Github](https://github.com/5Hyeons)</div>	|   <div align="center">[Github](https://github.com/yoorichae)</div>	|   <div align="center">[Github](https://github.com/wSangbae)</div>	| <div align="center">[Github](https://github.com/choisaywhy)</div> | <div align="center">[Github](https://github.com/pirate-turtle)</div>|
