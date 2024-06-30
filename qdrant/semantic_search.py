# Import the models - 
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

# The Sentence Transformers framework contains many embedding models. However, all-MiniLM-L6-v2 is the fastest encoder for this tutorial.
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# add the dataset
documents = [
    {
        "name": "The Time Machine",
        "description": "A man travels through time and witnesses the evolution of humanity.",
        "author": "H.G. Wells",
        "year": 1895,
    },
    {
        "name": "Ender's Game",
        "description": "A young boy is trained to become a military leader in a war against an alien race.",
        "author": "Orson Scott Card",
        "year": 1985,
    },
    {
        "name": "Brave New World",
        "description": "A dystopian society where people are genetically engineered and conditioned to conform to a strict social hierarchy.",
        "author": "Aldous Huxley",
        "year": 1932,
    },
    {
        "name": "The Hitchhiker's Guide to the Galaxy",
        "description": "A comedic science fiction series following the misadventures of an unwitting human and his alien friend.",
        "author": "Douglas Adams",
        "year": 1979,
    },
    {
        "name": "Dune",
        "description": "A desert planet is the site of political intrigue and power struggles.",
        "author": "Frank Herbert",
        "year": 1965,
    },
    {
        "name": "Foundation",
        "description": "A mathematician develops a science to predict the future of humanity and works to save civilization from collapse.",
        "author": "Isaac Asimov",
        "year": 1951,
    },
    {
        "name": "Snow Crash",
        "description": "A futuristic world where the internet has evolved into a virtual reality metaverse.",
        "author": "Neal Stephenson",
        "year": 1992,
    },
    {
        "name": "Neuromancer",
        "description": "A hacker is hired to pull off a near-impossible hack and gets pulled into a web of intrigue.",
        "author": "William Gibson",
        "year": 1984,
    },
    {
        "name": "The War of the Worlds",
        "description": "A Martian invasion of Earth throws humanity into chaos.",
        "author": "H.G. Wells",
        "year": 1898,
    },
    {
        "name": "The Hunger Games",
        "description": "A dystopian society where teenagers are forced to fight to the death in a televised spectacle.",
        "author": "Suzanne Collins",
        "year": 2008,
    },
    {
        "name": "The Andromeda Strain",
        "description": "A deadly virus from outer space threatens to wipe out humanity.",
        "author": "Michael Crichton",
        "year": 1969,
    },
    {
        "name": "The Left Hand of Darkness",
        "description": "A human ambassador is sent to a planet where the inhabitants are genderless and can change gender at will.",
        "author": "Ursula K. Le Guin",
        "year": 1969,
    },
    {
        "name": "The Three-Body Problem",
        "description": "Humans encounter an alien civilization that lives in a dying system.",
        "author": "Liu Cixin",
        "year": 2008,
    },
]


# using qdrant as a vector DB


# define storage location
# You need to tell Qdrant where to store embeddings. This is a basic demo, so your local computer will use its memory as temporary storage.
client = QdrantClient(":memory:")

#create collection
# Use recreate_collection if you are experimenting and running the script several times. 
# This function will first try to remove an existing collection with the same name.

# The vector_size parameter defines the size of the vectors for a specific collection. 
# If their size is different, it is impossible to calculate the distance between them. 
# 384 is the encoder output dimensionality. 
# You can also use model.get_sentence_embedding_dimension() to get the dimensionality of the model you are using.

# The distance parameter lets you specify the function used to measure the distance between two points.
client.recreate_collection(
    collection_name="my_books",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
        distance=models.Distance.COSINE,
    ),
)

# Upload data to collection

# Tell the database to upload documents to the my_books collection. This will give each record an id and a payload. The payload is just the metadata from the dataset.
client.upload_records(
    collection_name="my_books",
    records=[
        models.PointStruct(
            id=idx, 
            vector=encoder.encode(doc["description"]).tolist(), 
            payload=doc
        )
        for idx, doc in enumerate(documents)
    ],
)

# Ask the engine a question
hits = client.search(
    collection_name="my_books",
    query_vector=encoder.encode("alien invasion").tolist(),
    limit=3,
)
for hit in hits:
    print(hit.payload, "score:", hit.score)
    
# Narrow down the query
hits = client.search(
    collection_name="my_books",
    query_vector=encoder.encode("alien invasion").tolist(),
    query_filter=models.Filter(
        must=[models.FieldCondition(key="year", range=models.Range(gte=2000))]
    ),
    limit=1,
)

print('\n')
for hit in hits:
    print(hit.payload, "score:", hit.score)
