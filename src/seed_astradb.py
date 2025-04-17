import os
from astrapy import DataAPIClient, Database, Collection
from dotenv import load_dotenv

import json
from typing import Any, Callable, Dict

from astrapy.constants import VectorMetric
from astrapy.info import (
    CollectionDefinition,
    CollectionVectorOptions,
    VectorServiceOptions
)

load_dotenv()

def connect_to_database() -> Database:
    endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")  
    token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

    if not token or not endpoint:
        raise RuntimeError(
            "Environment variables ASTRA_DB_API_ENDPOINT and ASTRA_DB_APPLICATION_TOKEN must be defined"
        )
        
    if not token or not endpoint:
        raise RuntimeError(
            "Environment variables ASTRA_DB_API_ENDPOINT and ASTRA_DB_APPLICATION_TOKEN must be defined"
        )

    # Create an instance of the `DataAPIClient` class
    client = DataAPIClient()
    
    database = client.get_database(endpoint, token=token)

    print(f"Connected to database {database.info().name}")

    return database

def create_collection(database: Database, collection_name: str) -> Collection:
    collection = database.create_collection(
        collection_name,
        definition=CollectionDefinition(
            vector=CollectionVectorOptions(
                metric=VectorMetric.COSINE,
                service=VectorServiceOptions(
                    provider="nvidia",
                    model_name="NV-Embed-QA",
                )
            )
        )
    )
    return collection
    
def upload_json_data(
    collection: Collection,
    data_file_path: str,
    embedding_string_creator: Callable[[Dict[str, Any]], str],
) -> None:
    
    # Read the JSON file and parse it into a JSON array.
    with open(data_file_path, "r", encoding="utf8") as file:
        json_data = json.load(file)

    # Add a $vectorize field to each piece of data. 
    documents = [
        {
            **data,
            "$vectorize": embedding_string_creator(data.content),
        }
        for data in json_data
    ]

    # Upload the data.
    inserted = collection.insert_many(documents)
    print(f"Inserted {len(inserted.inserted_ids)} items.")

def main() -> None:
    database = connect_to_database()

    collection = create_collection(database, "law_data")  

    upload_json_data(
        collection,
        "../src/metadata/metadata.json", 
        lambda data: ( 
            f"summary: {data['summary']} | "
            f"genres: {', '.join(data['genres'])}"
        ),
    )

if __name__ == "__main__":
    main()