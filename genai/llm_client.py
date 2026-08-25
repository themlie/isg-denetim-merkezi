import ollama
import time

MODEL_NAME = "qwen2.5:3b"
SYSTEM_PROMPT = "Sen bir İş Sağlığı ve Güvenliği (İSG) uzmanısın. Yalnızca resmi kanun ve yönetmeliklere dayanarak profesyonel, eksiksiz ve tamamlanmış cümlelerle yanıtlar ver."

def ask_llm(question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    options = {
        "num_predict": 512,
        "temperature": 0.15,
        "num_ctx": 2048,
        "top_k": 30,
        "top_p": 0.9
    }
    response = ollama.chat(model=MODEL_NAME, messages=messages, options=options)
    return response["message"]["content"]

def stream_llm(question: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    options = {
        "num_predict": 512,
        "temperature": 0.15,
        "num_ctx": 2048,
        "top_k": 30,
        "top_p": 0.9
    }
    response = ollama.chat(model=MODEL_NAME, messages=messages, stream=True, options=options)
    for chunk in response:
        content = chunk.get("message", {}).get("content", "")
        if content:
            yield content

if __name__ == "__main__":
    sorular = [
        "Şantiyede baret takmak neden zorunludur?"
    ]
    for soru in sorular:
        print(f"Soru: {soru}")
        t0 = time.time()
        for token in stream_llm(soru):
            print(token, end="", flush=True)
        print(f"\nSüre: {time.time() - t0:.2f} sn")
