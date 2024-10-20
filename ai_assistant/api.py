from fastapi import FastAPI, Depends, Query, Body
from llama_index.core.agent import ReActAgent
from ai_assistant.agent import TravelAgent
from ai_assistant.models import AgentAPIResponse
from ai_assistant.tools import (
    reserve_flight,
    reserve_bus,
    reserve_hotel,
    reserve_restaurant,
    generate_trip_summary,
)
from datetime import datetime

def get_agent() -> ReActAgent:
    return TravelAgent().get_agent()


app = FastAPI(title="AI Agent")

agent_dependency = Depends(get_agent)

@app.get("/recommendations/cities")
def recommend_cities(
    notes: list[str] = Query(...), agent: ReActAgent = agent_dependency
):
    prompt = f"recommend cities in bolivia with the following notes: {notes}"
    return AgentAPIResponse(status="OK", agent_response=str(agent.chat(prompt)))

@app.get("/recommendations/places")
def recommend_places(
    city: str = Query(..., description="City to get recommendations for"),
    notes: list[str] = Query(None, description="Optional notes to guide the recommendations"),
    agent: ReActAgent = agent_dependency,
):
    prompt = f"Recomienda lugares para visitar en {city}."
    if notes:
        prompt += f" Considera las siguientes notas: {', '.join(notes)}."
    response = agent.chat(prompt)
    return AgentAPIResponse(status="OK", agent_response=str(response))

@app.get("/recommendations/hotels")
def recommend_hotels(
    city: str = Query(..., description="City to get hotel recommendations for"),
    notes: list[str] = Query(None, description="Optional notes to guide the recommendations"),
    agent: ReActAgent = agent_dependency,
):
    prompt = f"Recomienda hoteles para alojarse en {city}."
    if notes:
        prompt += f" Considera las siguientes notas: {', '.join(notes)}."
    response = agent.chat(prompt)
    return AgentAPIResponse(status="OK", agent_response=str(response))

@app.get("/recommendations/activities")
def recommend_activities(
    city: str = Query(..., description="City to get activity recommendations for"),
    notes: list[str] = Query(None, description="Optional notes to guide the recommendations"),
    agent: ReActAgent = agent_dependency,
):
    prompt = f"Recomienda actividades interesantes para realizar en {city}."
    if notes:
        prompt += f" Considera las siguientes notas: {', '.join(notes)}."
    response = agent.chat(prompt)
    return AgentAPIResponse(status="OK", agent_response=str(response))

@app.post("/reserve/flight")
def reserve_flight_endpoint(
    departure: str = Body(..., description="Departure city"),
    destination: str = Body(..., description="Destination city"),
    date_str: str = Body(..., description="Flight date in 'YYYY-MM-DD' format"),
):
    reservation = reserve_flight(date_str, departure, destination)
    return {"status": "OK", "reservation": reservation.model_dump()}

@app.post("/reserve/bus")
def reserve_bus_endpoint(
    departure: str = Body(..., description="Departure city"),
    destination: str = Body(..., description="Destination city"),
    date_str: str = Body(..., description="Travel date in 'YYYY-MM-DD' format"),
):
    reservation = reserve_bus(date_str, departure, destination)
    return {"status": "OK", "reservation": reservation.model_dump()}

@app.post("/reserve/hotel")
def reserve_hotel_endpoint(
    checkin_date_str: str = Body(..., description="Check-in date in 'YYYY-MM-DD' format"),
    checkout_date_str: str = Body(..., description="Check-out date in 'YYYY-MM-DD' format"),
    hotel_name: str = Body(..., description="Name of the hotel"),
    city: str = Body(..., description="City where the hotel is located"),
):
    reservation = reserve_hotel(checkin_date_str, checkout_date_str, hotel_name, city)
    return {"status": "OK", "reservation": reservation.model_dump()}

@app.post("/reserve/restaurant")
def reserve_restaurant_endpoint(
    reservation_date: str = Body(..., description="Reservation date in 'YYYY-MM-DD' format"),
    reservation_time: str = Body(..., description="Reservation time in 'HH:MM:SS' format"),
    restaurant: str = Body(..., description="Name of the restaurant"),
    city: str = Body(..., description="City where the restaurant is located"),
    dish: str = Body(None, description="Specific dish to reserve (optional)"),
):
    reservation_datetime_str = f"{reservation_date}T{reservation_time}"
    reservation = reserve_restaurant(reservation_datetime_str, restaurant, city, dish)
    return {"status": "OK", "reservation": reservation.model_dump()}


@app.get("/trip/report")
def trip_report(agent: ReActAgent = agent_dependency):
    prompt = "Genera un reporte detallado del viaje basado en las reservas realizadas."
    response = agent.chat(prompt)
    return AgentAPIResponse(status="OK", agent_response=str(response))
