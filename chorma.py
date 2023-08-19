import chromadb

phrases = [
    "Amanda baked cookies and will bring Jerry some tomorrow.",
    "Olivia and Olivier are voting for liberals in this election.",
    "Sam is confused, because he overheard Rick complaining about him as a roommate. Naomi thinks Sam should talk to Rick. Sam is not sure what to do.",
    "John's cookies were only half-baked but he still carries them for Mary."
]

ids = ["001", "002", "003", "004"]

metadatas = [{"source": "pdf-1"}, {"source": "doc-1"},
             {"source": "pdf-2"}, {"source": "txt-1"}]

chroma_client = chromadb.Client()

chroma_client = chromadb.PersistentClient(path="./chromadb/")

collection = chroma_client.create_collection(name="tns_tutorial")
collection.add(
    documents=phrases,
    metadatas=metadatas,
    ids=ids
)

# collection = chroma_client.get_collection("tns_tutorial")

collection.peek()

results = collection.query(
    query_texts=["Mary got half-baked cake from John"],
    n_results=2
)

print(results)

results = collection.query(
    query_texts=["cookies"],
    where={"source": "pdf-1"},
    n_results=1
)
print(results)

collection.delete()
