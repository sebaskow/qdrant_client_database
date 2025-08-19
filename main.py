from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition, MatchValue, OptimizersConfigDiff, PayloadSchemaType

## #############################################################################################################
##
## Uwaga u mnie musiałem odinstalować pakiety poleceniem:
## > pip uninstall torch torchvision torchaudio -y
## następnie zainstlować zgodne z cuda12.9 poleceniem:
## > pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu129 torch==2.9.0.dev20250729+cu129
## przy dostępnych bibliotekach nie trzeba tego robić
##
## #############################################################################################################


## insert           – pojedynczy rekord
## insert_many      – szybki batch insert
## select_by_id     – po ID
## select_text      – dokładne dopasowanie payloadu
## select_like      – semantyczne + LIKE
## select_semantic  – czyste semantyczne z filtrem score


class DataBaseQDrant:
    def __init__(self, collection_name, model, client: QdrantClient):
        self.collection_name = collection_name
        self.model = model
        self.client = client

    def vector_sentence(self, text: str):
        """Zamienia tekst na wektor float32 (GPU jeśli model na cuda)."""
        return self.model.encode(
            text,
            batch_size=256,
            show_progress_bar=False,
            convert_to_numpy=True
        ).astype("float32")

    # --- INSERT ---
    def insert(self, text: str, id: int):
        point = rest.PointStruct(
            id=id,
            vector=self.vector_sentence(text),
            payload={"text": text, "info": "dane testowe"}
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
            wait=False
        )

        print(f"[INSERT] Wstawiono: {text}")

    def insert_many(self, texts: list[str], ids: list[int]):
        """Hurtowe wstawianie wielu rekordów na raz (szybsze)."""
        vectors = self.model.encode(
            texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True
        ).astype("float32")

        points = [
            rest.PointStruct(id=ids[i], vector=vectors[i], payload={"text": texts[i]})
            for i in range(len(texts))
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=False
        )
        print(f"[INSERT_MANY] Wstawiono {len(points)} rekordów.")

    # --- SELECT ---
    def select_by_id(self, id: int):
        result = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[id],
            with_vectors=True
        )
        print(f"[SELECT_BY_ID] {result}")
        return result

    def select_text(self, text: str):
        """Dokładne dopasowanie payloadu text == value"""
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="text",
                    match=MatchValue(value=text)
                )
            ]
        )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=self.vector_sentence(text),
            query_filter=query_filter,
            limit=10
        )

        print(f"[SELECT_TEXT] znaleziono {len(results.points)} wyników")
        return results.points

    def select_like(self, text: str):
        """Wyszukiwanie semantyczne + LIKE w payload."""
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=self.vector_sentence(text),
            limit=50
        )

        points = results.points
        filtered = [
            r for r in points if text.lower() in r.payload.get("text", "").lower()
        ]

        for r in filtered:
            print(f"[SELECT_LIKE] {r.payload['text']} (score={r.score:.3f})")

        return filtered

    def select_semantic(self, text: str, score: float = 0):
        """Czyste wyszukiwanie semantyczne z progiem score."""
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=self.vector_sentence(text),
            limit=50
        )

        points = results.points

        if score > 0:
            filtered = [r for r in points if r.score > score]
            for r in filtered:
                print(f"[SEMANTIC] {r.payload['text']} (score={r.score:.3f})")
            return filtered

        for r in points:
            print(f"[SEMANTIC] {r.payload['text']} (score={r.score:.3f})")

        return points


# 1. Łączenie się z lokalnym Qdrant
client = QdrantClient(host="localhost", port=6333)

# 2. Ładowanie modelu do embeddingów
model = SentenceTransformer("sentence-transformers/LaBSE", device="cuda")

# 3. Tworzymy kolekcję (nadpisujemy jeśli istnieje)
collection_name = "demo_collection"

# 4 Sprawdzenie i usunięcie istniejącej kolekcji
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

# 5 Tworzenie nowej kolekcji
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=768,                           # LaBSE embeddings = 768
        distance=Distance.COSINE,           # najlepsze dla NLP
        on_disk=False                       # embeddings przechowywane na dysku True (oszczędzasz RAM)
    ),
    shard_number=4,                         # rozdziel dane na kilka shardów (lepszy parallelizm)
    replication_factor=1,                   # 1 kopia (dla lokalnego środowiska wystarczy)
    optimizers_config=OptimizersConfigDiff(
        memmap_threshold=20000,             # powyżej tylu punktów trzyma wektory na dysku
        indexing_threshold=10000            # buduj index dopiero od 10k punktów
    )
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

texts = [
    "Kot siedzi na płocie",
    "Pies biega po parku",
    "Samochód jedzie szybko",
    "Lubię programować w Pythonie",
    "Przemysł wydobywczy ulega zagładzie, przez ogromny negatywny wpływ na środowisko naturalne"
]
ids = [9, 10, 11, 12, 13]
dataBase.insert_many(texts, ids)


print("--------[SELECT BY ID]--------------------")
dataBase.select_by_id(1)
print("--------[LIKE]--------------------")
dataBase.select_like("psy")
print("--------[SEMANTIC]--------------------")
dataBase.select_semantic("zwierzęta domowe", 0.27)


