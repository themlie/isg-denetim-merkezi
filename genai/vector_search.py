import faiss
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from genai.document_loader import load_documents

VECTORSTORE_DIR = Path(__file__).parent.parent / "vectorstore"
FAISS_INDEX_PATH = VECTORSTORE_DIR / "isg_faiss.index"
CHUNKS_PATH = VECTORSTORE_DIR / "chunks.pkl"

MODEL_NAME = 'intfloat/multilingual-e5-small'
model = SentenceTransformer(MODEL_NAME)

index = None
chunks = None

def load_store():
    global index, chunks
    if index is not None and chunks is not None:
        return True
    if FAISS_INDEX_PATH.exists() and CHUNKS_PATH.exists():
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        return True
    return False

# Modül ilk yüklendiğinde hafızaya al
load_store()


def build_index():
    global index, chunks
    print("Dokümanlar yükleniyor ve parçalanıyor...")
    docs = load_documents()
    
    if not docs:
        print("Hata: Parça bulunamadı!")
        return

    print(f"{len(docs)} parça vektörleştiriliyor ({MODEL_NAME})...")
    texts = [doc.page_content for doc in docs]
    
    # E5 modeli için passage: öneki ve kosinüs benzerliği normalizasyonu
    embeddings = model.encode(["passage: " + t for t in texts], normalize_embeddings=True)
    
    # FAISS indeksi oluştur (Inner Product - Kosinüs Benzerliği)
    dimension = embeddings.shape[1]
    new_index = faiss.IndexFlatIP(dimension)
    
    # Vektörleri ekle
    new_index.add(np.array(embeddings, dtype=np.float32))
    
    # Kaydet
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(new_index, str(FAISS_INDEX_PATH))
    
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(docs, f)
        
    print(f"İndeks {FAISS_INDEX_PATH} adresine kaydedildi.")
    
    index = new_index
    chunks = docs

def search(query, k=3):
    global index, chunks
    if index is None or chunks is None:
        if not load_store():
            print("İndeks bulunamadı, önce build_index() çalıştırın.")
            return []

    # E5 modeli için query: öneki ve kosinüs benzerliği normalizasyonu
    query_text = "query: " + query if not query.startswith("query: ") else query
    query_embedding = model.encode([query_text], normalize_embeddings=True)

    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        
        chunk = chunks[idx]
        distance = distances[0][i]
        results.append({
            "chunk": chunk,
            "distance": float(distance)
        })

    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_index()
    else:
        build_index()
        test_query = "Yüksekte çalışmalarda baret ve koruyucu donanım kullanılması zorunlu mudur?"
        print(f"\nTest Sorgusu: '{test_query}'")
        print("-" * 50)
        results = search(test_query, k=3)
        for i, res in enumerate(results):
            doc = res['chunk']
            print(f"--- Sonuç {i+1} (Skor: {res['distance']:.4f}) ---")
            print(f"Kaynak: {doc.metadata.get('source')} - Sayfa: {doc.metadata.get('page')}")
            print(f"Metin:\n{doc.page_content[:200]}...\n")
