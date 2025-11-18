import pydantic


class Accommodation(pydantic.BaseModel):
    name: str
    city_name: str
    address: str
    room_type: str
    price: int
    theme: str

class PointOfInterst(pydantic.BaseModel):
    city_name: str
    address: str
    attraction_type: str
    theme: str

class Region(pydantic.BaseModel):
    name: str
    intro: str

class AccommodationSearchResponse(pydantic.BaseModel):
    accommodations: list[Accommodation]

class PointOfInterestSearchResponse(pydantic.BaseModel):
    point_of_interests: list[PointOfInterst]

class RegionSearchResponse(pydantic.BaseModel):
    regions: list[Region]

class TravelSearchResponse(pydantic.BaseModel):
    accommodations: list[Accommodation] = []
    point_of_interests: list[PointOfInterst] = []
    regions: list[Region] = []
