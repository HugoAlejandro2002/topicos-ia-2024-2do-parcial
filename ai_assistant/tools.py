import json
from random import randint
import wikipedia
from datetime import date, datetime
from llama_index.core.tools import QueryEngineTool, FunctionTool, ToolMetadata
from ai_assistant.rags import TravelGuideRAG
from ai_assistant.prompts import travel_guide_qa_tpl, travel_guide_description
from ai_assistant.config import get_agent_settings
from ai_assistant.models import (
    TripReservation,
    TripType,
    HotelReservation,
    RestaurantReservation,
)
from ai_assistant.utils import save_reservation

SETTINGS = get_agent_settings()

travel_guide_tool = QueryEngineTool(
    query_engine=TravelGuideRAG(
        store_path=SETTINGS.travel_guide_store_path,
        data_dir=SETTINGS.travel_guide_data_path,
        qa_prompt_tpl=travel_guide_qa_tpl,
    ).get_query_engine(),
    metadata=ToolMetadata(
        name="travel_guide", description=travel_guide_description, return_direct=False
    ),
)

def reserve_flight(date_str: str, departure: str, destination: str) -> TripReservation:
    """
    Reserves a flight from a departure city to a destination city on a specific date.

    Args:
        date_str (str): Flight date in 'YYYY-MM-DD' format.
        departure (str): Departure city.
        destination (str): Destination city.

    Returns:
        TripReservation: Flight reservation object with reservation details.
    """
    print(f"Making flight reservation from {departure} to {destination} on date: {date_str}")
    reservation = TripReservation(
        trip_type=TripType.flight,
        departure=departure,
        destination=destination,
        date=date.fromisoformat(date_str),
        cost=randint(200, 700),
    )

    save_reservation(reservation)
    return reservation

flight_tool = FunctionTool.from_defaults(fn=reserve_flight, return_direct=False)

def reserve_bus(date_str: str, departure: str, destination: str) -> TripReservation:
    """
    Reserves a bus ticket from a departure city to a destination city on a specific date.

    Args:
        date_str (str): Travel date in 'YYYY-MM-DD' format.
        departure (str): Departure city.
        destination (str): Destination city.

    Returns:
        TripReservation: Bus reservation object with reservation details.
    """
    print(f"Making bus reservation from {departure} to {destination} on date: {date_str}")
    reservation = TripReservation(
        trip_type=TripType.bus,
        departure=departure,
        destination=destination,
        date=date.fromisoformat(date_str),
        cost=randint(50, 200),
    )

    save_reservation(reservation)
    return reservation

bus_tool = FunctionTool.from_defaults(fn=reserve_bus, return_direct=False)

def reserve_hotel(
    checkin_date_str: str, checkout_date_str: str, hotel_name: str, city: str
) -> HotelReservation:
    """
    Reserves a hotel room in a specific city.

    Args:
        checkin_date_str (str): Check-in date in 'YYYY-MM-DD' format.
        checkout_date_str (str): Check-out date in 'YYYY-MM-DD' format.
        hotel_name (str): Name of the hotel.
        city (str): City where the hotel is located.

    Returns:
        HotelReservation: Hotel reservation object with reservation details.
    """
    print(f"Making hotel reservation at {hotel_name} in {city} from {checkin_date_str} to {checkout_date_str}")
    checkin_date = date.fromisoformat(checkin_date_str)
    checkout_date = date.fromisoformat(checkout_date_str)
    num_nights = (checkout_date - checkin_date).days
    cost_per_night = randint(100, 300)
    total_cost = num_nights * cost_per_night

    reservation = HotelReservation(
        checkin_date=checkin_date,
        checkout_date=checkout_date,
        hotel_name=hotel_name,
        city=city,
        cost=total_cost,
    )

    save_reservation(reservation)
    return reservation

hotel_tool = FunctionTool.from_defaults(fn=reserve_hotel, return_direct=False)

def reserve_restaurant(
    reservation_datetime_str: str, restaurant: str, city: str, dish: str = "not specified"
) -> RestaurantReservation:
    """
    Reserves a table at a restaurant in a specific city.

    Args:
        reservation_datetime_str (str): Reservation date and time in 'YYYY-MM-DDTHH:MM:SS' format.
        restaurant (str): Name of the restaurant.
        city (str): City where the restaurant is located.
        dish (str, optional): Specific dish to reserve.

    Returns:
        RestaurantReservation: Restaurant reservation object with reservation details.
    """
    print(f"Making restaurant reservation at {restaurant} in {city} on {reservation_datetime_str}")
    reservation_time = datetime.fromisoformat(reservation_datetime_str)
    cost = randint(20, 100)

    reservation = RestaurantReservation(
        reservation_time=reservation_time,
        restaurant=restaurant,
        city=city,
        dish=dish,
        cost=cost,
    )

    save_reservation(reservation)
    return reservation

restaurant_tool = FunctionTool.from_defaults(fn=reserve_restaurant, return_direct=False)

def generate_trip_summary() -> str:
    """
    Generates a detailed summary of the trip based on the reservations in trip.json.

    Returns:
        str: A detailed trip report including activities organized by place and date,
             a summary of the total budget, and comments on the places and activities.
    """
    try:
        with open(SETTINGS.log_file, "r") as file:
            reservations = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return "No travel reservations found. Please make some reservations first."

    total_cost = 0
    summary = {}

    for res in reservations:
        res_type = res.get("reservation_type")
        if res_type == "TripReservation":
            date_str = res["date"]
            date_obj = datetime.fromisoformat(date_str)
            city = res["destination"]
            activity = f"Travel by {res['trip_type']} from {res['departure']} to {res['destination']} on {date_str}"
        elif res_type == "HotelReservation":
            date_str = res["checkin_date"]
            date_obj = datetime.fromisoformat(date_str)
            city = res["city"]
            activity = f"Stay at {res['hotel_name']} from {res['checkin_date']} to {res['checkout_date']}"
        elif res_type == "RestaurantReservation":
            date_str = res["reservation_time"]
            date_obj = datetime.fromisoformat(date_str)
            city = res["city"]
            activity = f"Reservation at {res['restaurant']} on {res['reservation_time']}"
        else:
            continue

        total_cost += res.get("cost", 0)

        date_key = date_obj.strftime("%Y-%m-%d")
        if date_key not in summary:
            summary[date_key] = []
        summary[date_key].append({
            "city": city,
            "activity": activity,
            "cost": res.get("cost", 0)
        })

    report = "Trip Summary:\n"
    for date in sorted(summary.keys()):
        report += f"\nDate: {date}\n"
        for item in summary[date]:
            report += f"- City: {item['city']}\n"
            report += f"  Activity: {item['activity']}\n"
            report += f"  Cost: ${item['cost']}\n"

    report += f"\nEstimated Total Cost: ${total_cost}\n"

    report += "\nWe hope you enjoy your trip! Remember to visit local attractions and immerse yourself in the culture."

    return report

trip_summary_tool = FunctionTool.from_defaults(fn=generate_trip_summary, return_direct=False)


def get_department_info(department_name: str) -> str:
    """
    Fetches information from Wikipedia about a specified Bolivian department.
    
    Args:
        department_name (str): Name of the Bolivian department.
    
    Returns:
        str: A summary of information about the department.
    """
    try:
        wikipedia.set_lang("es")  # Set language to Spanish
        page = wikipedia.page(department_name + " (departamento de Bolivia)")
        summary = page.summary
        return summary
    except wikipedia.exceptions.PageError:
        return f"No se encontró una página de Wikipedia para {department_name}."
    except Exception as e:
        return f"Ocurrió un error al obtener la información: {e}"

department_info_tool = FunctionTool.from_defaults(fn=get_department_info, return_direct=False)