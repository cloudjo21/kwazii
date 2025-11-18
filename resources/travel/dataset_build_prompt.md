```
아래 정보들을 참고해서 City, Accommodatino, POI 데이터셋을 각각 만들어줘
- tab seperation을 쓰는 csv 형식의 데이터셋으로
- 한글로 작성
- region.csv, accommodation.csv, poi.csv 세개 파일로 만들어서 다운로드 받도록 준비해줘
- python코드 예제는 보여줄 필요 없어
- 각 데이터셋에서 만들어야할 기준을 잘 지켜서 데이터셋 만들어
- accommodation과 poi는 theme와 address을 형식적으로 만들지 말고 실제 검색 데이터를 기반해서 만들어줘

## City
- 대상 지역들: 제주, 도쿄, 오사카, 교토, 나라, 취리히, 인터라켄, 루체른, 베른, 마드리드, 세비야, 그라나다, 바르셀로나, 나트랑, 달랏
### Fields
- name
- intro: 여행자 관점에서 매력적인 소개로 30자 이내로 작성해줘

## Accommodation
각 city마다 10개씩 다양한 옵션으로 추가해줘. 
### Fields
- name: 숙소 이름
- city_name
- address
- room_type: 인원 중심으로 category를 몇개 정해서 할당해줘
- price
- theme: 숙소 분위기는 감성적으로 어매니티는 정보전달성 위주로 대략 100자 이내로 작성해줘
- rating: 5점 평점 중에 적당히 할당해줘

## POI
각 city마다 10개씩 다양한 옵션으로 추가해줘.
### Fields
- city_name
- address
- attraction_type: POI가 될 수 있는 category를 몇개 정해서 할당해줘
- theme: 장소 분위기는 감성적으로 제공하는 가치는 정보전달성 위주로 대략 100자 이내로 작성해줘
```