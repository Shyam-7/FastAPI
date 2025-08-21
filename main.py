import os
from dotenv import load_dotenv
from ai.vectorstores.chroma_store import get_chroma_vector_store
from ai.chains.ingestion_chain import ingest_data
from ai.graphs.multi_agent_flow import create_multi_agent_flow
from langchain_openai import OpenAI

def main():
    """
    Main function to run the AI service recommendation system.
    """
    load_dotenv()

    # Create a vector store
    vector_store = get_chroma_vector_store(path="./car_service_db")

    # Ingest data
    data = """
    Service history for a Toyota Camry:
    - 2021-01-01: Oil change
    - 2021-06-01: Tire rotation
    - 2022-01-01: Oil change, brake inspection
    """
    ingest_data(data, vector_store)

    # Create a language model
    llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Create a multi-agent flow
    app = create_multi_agent_flow(llm, vector_store)

    # Run the flow
    car_details = {
        "brand": "Toyota",
        "model": "Camry",
        "age": "3 years",
        "mileage": "30000 miles"
    }

    final_state = None
    for output in app.stream(car_details):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
            if key == "analysis_node":
                final_state = value
        print("\n---\n")

    if final_state:
        print("Final Recommendations:")
        print(final_state['recommendations'])


if __name__ == "__main__":
    main()
