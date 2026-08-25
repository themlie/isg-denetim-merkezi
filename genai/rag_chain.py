import textwrap
from genai.vector_search import search
from genai.llm_client import ask_llm, stream_llm

def get_rag_context_and_prompt(question: str):
    results = search(question, k=3)
    
    if not results:
        return None, [], "Bu bilgi mevzuat dokümanlarında yer almamaktadır."
        
    baglam_metinleri = []
    sources = []
    for i, res in enumerate(results, 1):
        doc = res["chunk"]
        kaynak = doc.metadata.get("source", "6331 Sayılı Kanun")
        sayfa = doc.metadata.get("page", "1")
        sources.append({"source": kaynak, "page": sayfa, "score": res.get("distance", 0.0)})
        baglam_metinleri.append(f"[{kaynak} - Sayfa {sayfa}]:\n{doc.page_content.strip()}")
        
    baglam = "\n\n".join(baglam_metinleri)
    
    rag_prompt = f"""Aşağıdaki resmi İSG mevzuat maddelerini ve bağlamı inceleyerek soruyu yanıtla.

BAĞLAM:
{baglam}

SORU:
{question}

GÖREV:
- Yukarıdaki mevzuat bağlamına dayanarak soruyu net, doğru ve kurumsal bir dille 2-3 cümlede yanıtla.
- Cümlelerini daima tam ve eksiksiz olarak bitir.
- İşverenin ve çalışanların yasal sorumluluklarını veya KKD zorunluluklarını açıkça belirt.
- Eğer konu mevzuatla tamamen alakasızsa (örneğin hava durumu, yemek tarifi vb.), "Bu bilgi mevzuat dokümanlarında yer almamaktadır." şeklinde belirt.

CEVAP:"""
    return rag_prompt, sources, None

def ask_mevzuat_with_sources(question: str):
    prompt, sources, fallback = get_rag_context_and_prompt(question)
    if fallback:
        return fallback, sources
    cevap = ask_llm(prompt)
    return cevap, sources

def stream_mevzuat_with_sources(question: str):
    prompt, sources, fallback = get_rag_context_and_prompt(question)
    if fallback:
        def fallback_gen():
            yield fallback
        return fallback_gen(), sources
    return stream_llm(prompt), sources

def ask_mevzuat(question: str) -> str:
    ans, _ = ask_mevzuat_with_sources(question)
    return ans

# Aliases
query_rag = ask_mevzuat_with_sources
stream_rag = stream_mevzuat_with_sources

if __name__ == "__main__":
    soru = "Yüksekte çalışmalarda baret ve koruyucu donanım kullanılması zorunlu mudur?"
    print(f"SORU:\n{soru}\n")
    tokens, srcs = stream_mevzuat_with_sources(soru)
    print("YANIT (STREAM):")
    for t in tokens:
        print(t, end="", flush=True)
    print("\n\nKAYNAKLAR:", srcs)
