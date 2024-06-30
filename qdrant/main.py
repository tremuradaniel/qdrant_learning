from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct

COLLECTION_NAME = "test_collection"

# connect to qdrant
client = QdrantClient(url="http://localhost:6333")



# create collection
# client.create_collection(
#     collection_name=COLLECTION_NAME,
#     vectors_config=VectorParams(size=4, distance=Distance.DOT),
# )


# create vector
## Payloads are other data you want to associate with the vector
operation_info = client.upsert(
    collection_name=COLLECTION_NAME,
    wait=True,
    points=[
        PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"city": "Berlin"}),
        PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London"}),
        PointStruct(id=3, vector=[0.36, 0.55, 0.47, 0.94], payload={"city": "Moscow"}),
        PointStruct(id=4, vector=[0.18, 0.01, 0.85, 0.80], payload={"city": "New York"}),
        PointStruct(id=5, vector=[0.24, 0.18, 0.22, 0.44], payload={"city": "Beijing"}),
        PointStruct(id=6, vector=[0.35, 0.08, 0.11, 0.44], payload={"city": "Mumbai"}),
    ],
)

print(operation_info)


# Run a query
# Let’s ask a basic question - Which of our stored vectors are most similar to the query vector [0.2, 0.1, 0.9, 0.7]?


search_result = client.search(
    collection_name=COLLECTION_NAME, query_vector=[0.2, 0.1, 0.9, 0.7], limit=3
)

print(search_result)


# Add a filter
# We can narrow down the results further by filtering by payload. Let’s find the closest results that include “London”.

from qdrant_client.models import Filter, FieldCondition, MatchValue

search_result = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=[0.2, 0.1, 0.9, 0.7],
    query_filter=Filter(
        must=[FieldCondition(key="city", match=MatchValue(value="London"))]
    ),
    with_payload=True,
    limit=3,
)

print("\n")
print(search_result)