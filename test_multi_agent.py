"""Test script for multi-agent session support with coordinator pattern."""

from strands import Agent, tool
from strands.models.bedrock import BedrockModel
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig

# Tools for sub-agents
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    weather_data = {
        "seattle": "Rainy, 55°F", "san francisco": "Foggy, 62°F", 
        "new york": "Sunny, 68°F", "tokyo": "Clear, 72°F",
    }
    return weather_data.get(location.lower(), f"No weather data for {location}")

@tool
def search_restaurants(cuisine: str, city: str) -> list:
    """Search for restaurants by cuisine type in a city."""
    restaurants = {
        ("italian", "seattle"): ["Altura", "Spinasse", "Il Corvo"],
        ("japanese", "seattle"): ["Shiro's", "Sushi Kashiba"],
        ("mexican", "new york"): ["Cosme", "Atla"],
    }
    return restaurants.get((cuisine.lower(), city.lower()), [f"No {cuisine} restaurants in {city}"])

@tool
def calculate_tip(bill_amount: float, tip_percentage: float = 18.0) -> str:
    """Calculate tip and total for a bill."""
    tip = bill_amount * (tip_percentage / 100)
    return f"Tip: ${tip:.2f}, Total: ${bill_amount + tip:.2f}"

@tool
def get_directions(from_loc: str, to_loc: str) -> str:
    """Get directions between two locations."""
    return f"Directions from {from_loc} to {to_loc}: Head north, turn right. ~15 min."

# Global sub-agents (initialized in main)
weather_agent = None
restaurant_agent = None
travel_agent = None

# Coordinator tools to delegate to sub-agents
@tool
def ask_weather_specialist(question: str) -> str:
    """Delegate weather-related questions to the weather specialist."""
    return str(weather_agent(question))

@tool
def ask_restaurant_specialist(question: str) -> str:
    """Delegate restaurant/food questions to the restaurant specialist."""
    return str(restaurant_agent(question))

@tool
def ask_travel_specialist(question: str) -> str:
    """Delegate travel/directions questions to the travel specialist."""
    return str(travel_agent(question))

def main():
    global weather_agent, restaurant_agent, travel_agent
    
    print("🚀 Multi-Agent Coordinator Test\n")
    
    config = AgentCoreMemoryConfig(
        memory_id="CustomerSupport-pkP616GF9D",
        session_id="coordinator-test-002",
        actor_id="coordinator-user",
    )

    print("Creating session manager...")
    session_manager = AgentCoreMemorySessionManager(
        agentcore_memory_config=config,
        region_name="us-west-2",
    )
    print(f"Root event ID: {session_manager._root_event_id}\n")

    model = BedrockModel(model_id="anthropic.claude-3-haiku-20240307-v1:0")

    # Create sub-agents
    print("Creating sub-agents...")
    weather_agent = Agent(
        agent_id="weather",
        system_prompt="You are a weather specialist. Use get_weather tool. Be concise.",
        session_manager=session_manager,
        model=model,
        tools=[get_weather],
    )
    
    restaurant_agent = Agent(
        agent_id="restaurant",
        system_prompt="You are a restaurant specialist. Use search_restaurants and calculate_tip tools. Be concise.",
        session_manager=session_manager,
        model=model,
        tools=[search_restaurants, calculate_tip],
    )
    
    travel_agent = Agent(
        agent_id="travel",
        system_prompt="You are a travel specialist. Use get_directions tool. Be concise.",
        session_manager=session_manager,
        model=model,
        tools=[get_directions],
    )

    # Create coordinator
    print("Creating coordinator agent...")
    coordinator = Agent(
        agent_id="coordinator",
        system_prompt="""You are a helpful coordinator. Route requests to specialists:
- Weather questions → ask_weather_specialist
- Restaurant/food/tip questions → ask_restaurant_specialist  
- Travel/directions questions → ask_travel_specialist
Always use the appropriate specialist tool. Be concise.""",
        session_manager=session_manager,
        model=model,
        tools=[ask_weather_specialist, ask_restaurant_specialist, ask_travel_specialist],
    )

    print(f"Branches created: {session_manager._created_branches}\n")

    # Interactive loop
    print("="*60)
    print("🤖 Coordinator Ready! (delegates to weather/restaurant/travel)")
    print("="*60)
    print("Examples: 'weather in Seattle', 'Italian restaurants in Seattle',")
    print("          'calculate 20% tip on $50', 'directions to downtown'")
    print("Commands: 'branches', 'messages', 'quit'")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() == 'quit':
                break
            if user_input.lower() == 'branches':
                print(f"Branches: {session_manager._created_branches}\n")
                continue
            if user_input.lower() == 'messages':
                for aid in ["coordinator", "weather", "restaurant", "travel"]:
                    msgs = session_manager.list_messages(config.session_id, aid)
                    print(f"  {aid}: {len(msgs)} messages")
                print()
                continue

            response = coordinator(user_input)
            print(f"Coordinator: {response}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")

    print("Goodbye!")

if __name__ == "__main__":
    main()
