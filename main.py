from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition, MatchValue




class DataBaseQDrant:
    def __init__(self, collection_name, model, client):
        self.collection_name = collection_name
        self.model = model
        self.client = client

    def insert(self, text, id):
        vector = self.model.encode(text).astype("float32")

        # Tworzymy jeden punkt (wiersz)
        point = rest.PointStruct(
            id=id,  # unikalne ID
            vector=vector,  # embedding
            payload={"text": text}  # dodatkowe dane (jak kolumny)
        )

        # Wstawienie do bazy
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

        print(f"Dane zostały wstawione: {text}")

    def select_by_id(self, id):
        result = self.client.retrieve( collection_name=collection_name, ids=[id], with_vectors=True)

        print(f"Oczytano wiesz danych: {result}")
        return result

    def select_text(self, text):
        vector = self.model.encode(text).astype("float32")

        filter = Filter(
            must=[
                FieldCondition(
                    key="text",  # klucz w payload
                    match=MatchValue(value=text) # dokładne dopasowanie
                )
            ]
        )

        results = client.query_points(
            collection_name=self.collection_name,
            query=PointQuery(
                vector=query_vector,
                query_filter=filter,
                limit=10
            )
        )

        print(f"Oczytano wiesz danych: {results}")
        return results

    def select_like(self, text):
        vector = self.model.encode(text).astype("float32")

        results = client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=50  # większy limit, żeby mieć z czego filtrować
        )

        filtered = [
            r for r in results
            if text in r.payload.get("text", "").lower()
        ]

        for r in filtered:
            print(r.payload["text"], r.score)

        return filtered

    def select_semantic(self, text, score = 0):
        vector = self.model.encode(text).astype("float32")

        results = client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=50  # większy limit, żeby mieć z czego filtrować
        )

        if score > 0:
            filtered = [r for r in results if r.score > score]
            for r in filtered:
                print(f"{r.payload['text']} (score: {r.score:.3f})")

            return filtered

        for r in results:
            print(r.payload["text"], r.score)

        return results


# 1. Łączenie się z lokalnym Qdrant
client = QdrantClient(host="localhost", port=6333)

# 2. Ładowanie modelu do embeddingów
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Tworzymy kolekcję (nadpisujemy jeśli istnieje)
collection_name = "demo_collection"

# 4 Sprawdzenie i usunięcie istniejącej kolekcji
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

# 5 Tworzenie nowej kolekcji
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

print(f"Kolekcja '{collection_name}' gotowa.")


dataBase = DataBaseQDrant(collection_name, model, client)
dataBase.insert("Sebastian ma piękne włosy", 1)
dataBase.insert("Koty lubią mleko", 2)
dataBase.insert("Alicja lubi koty i psy.", 3)
dataBase.insert("W naszym hotelu jest bardzo przyjemnie", 4)
dataBase.insert("Nie akceptujemy zwierząt domowych", 5)
dataBase.insert("Adaś ma małego psa który się wabi Reksio", 6)
dataBase.insert("W zasadzie do papuga też należy do zwierząt domowych", 7)
dataBase.insert("Pies mojej sąsiadki ugryzł małe dziecko, duży problem z teog", 8)

dataBase.select_by_id(1)

dataBase.select_like("psy")

dataBase.select_semantic("zwierzęta domowe", 0.47)


