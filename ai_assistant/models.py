from pydantic import BaseModel, Field
from enum import Enum
from datetime import date, datetime

class TripType(str, Enum):
    flight = "FLIGHT"
    bus = "BUS"

class TripReservation(BaseModel):
    trip_type: TripType
    date: date
    departure: str
    destination: str
    cost: int
    reservation_type: str = Field(default="TripReservation")

class HotelReservation(BaseModel):
    checkin_date: date
    checkout_date: date
    hotel_name: str
    city: str
    cost: int
    reservation_type: str = Field(default="HotelReservation")

class RestaurantReservation(BaseModel):
    reservation_time: datetime
    restaurant: str
    city: str
    dish: str
    cost: int
    reservation_type: str = Field(default="RestaurantReservation")

class AgentAPIResponse(BaseModel):
    status: str
    agent_response: str
    timestamp: datetime = Field(default_factory=datetime.now)
