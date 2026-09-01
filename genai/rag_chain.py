import textwrap
from genai.vector_search import search
from genai.llm_client import ask_llm, stream_llm

import re

def get_rag_context_and_prompt(question: str):
    q_clean = question.strip()
    q_lower = q_clean.lower()
    
    # Kapsam dışı doğrudan tetikleyiciler (yemek, eğlence, günlük sohbet vs.)
    out_of_scope_patterns = [
        r"\b(menemen|yemek|kahvaltı|akşam yemeği|tatlı|çay|kahve|pizza|hamburger|çorba)\b",
        r"\b(hava durumu|yağmur yağacak mı|yarın hava|sıcaklık)\b",
        r"\b(futbol|maç|fenerbahçe|galatasaray|beşiktaş|şampiyon)\b",
        r"\b(nasılsın|naber|kimsin|sen kimsin|günaydın|iyi akşamlar|espri|fıkra)\b"
    ]
    for pattern in out_of_scope_patterns:
        if re.search(pattern, q_lower):
            fallback_msg = "Bu soru 6331 sayılı İSG Kanunu ve Yapı İşleri Yönetmeliği kapsamı dışındadır. Lütfen iş sağlığı, güvenliği, KKD standartları veya saha mevzuatı ile ilgili bir soru yöneltiniz."
            return None, [], fallback_msg

    # İSG Konusu Kontrolü (Tam kelime eşleşmesi)
    isg_pattern = r"\b(isg|iş sağlığı|iş güvenliği|güvenlik|baret|yelek|kkd|şantiye|inşaat|iskele|yüksekte çalışma|iş kazası|ceza|kanun|yönetmelik|koruyucu donanım|tehlike|risk|işveren|çalışan|eldiven|gözlük|emniyet kemeri|yangın|denetim|mevzuat)\b"
    if not re.search(isg_pattern, q_lower):
        fallback_msg = "Bu soru 6331 sayılı İSG Kanunu ve Yapı İşleri Yönetmeliği kapsamı dışındadır. Lütfen iş sağlığı, güvenliği, KKD standartları veya saha mevzuatı ile ilgili bir soru yöneltiniz."
        return None, [], fallback_msg
    
    results = search(question, k=3)
    top_score = results[0]["distance"] if results else 0.0
    if not results or top_score < 0.60:
        fallback_msg = "Bu soru 6331 sayılı İSG Kanunu ve Yapı İşleri Yönetmeliği kapsamı dışındadır. Lütfen iş sağlığı, güvenliği, KKD standartları veya saha mevzuatı ile ilgili bir soru yöneltiniz."
        return None, [], fallback_msg

        
    baglam_metinleri = []
    sources = []
    for i, res in enumerate(results, 1):
        doc = res["chunk"]
        kaynak = doc.metadata.get("source", "6331 Sayılı Kanun")
        sayfa = doc.metadata.get("page", "1")
        sources.append({"source": kaynak, "page": sayfa, "score": res.get("distance", 0.0)})
        content_preview = doc.page_content.strip()
        baglam_metinleri.append(f"[{kaynak} - Madde/Sayfa {sayfa}]:\n{content_preview}")
        
    baglam = "\n\n".join(baglam_metinleri)
    
    rag_prompt = f"""Sen 6331 Sayılı İş Sağlığı ve Güvenliği Kanunu ve Yapı İşleri Yönetmeliği uzmanısın. Aşağıdaki resmi mevzuat maddelerine dayanarak soruyu net, doğru ve kurumsal bir dille 2-3 cümlede yanıtla.

RESMİ MEVZUAT METİNLERİ:
{baglam}

SORU:
{question}

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
