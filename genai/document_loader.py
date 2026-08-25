from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

MEVZUAT_DIR = Path(__file__).parent.parent / "data" / "mevzuat"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

def load_documents():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = []
    chunk_id = 0
    
    for pdf_path in MEVZUAT_DIR.glob("*.pdf"):
        reader = PdfReader(pdf_path)
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            
            if not text or not text.strip():
                continue
                
            # PDF'den gelen \r (carriage return) ve \n (yeni satır) karakterlerini temizle
            # Bu sayede terminalde yazılar birbirinin üstüne binmeyecek
            text = text.replace('\r', '').replace('\n', ' ')
            # Fazladan boşlukları tek boşluğa indir
            import re
            text = re.sub(' +', ' ', text)
                
            page_chunks = splitter.split_text(text)
            
            for chunk_text in page_chunks:
                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_num,
                        "chunk_id": chunk_id
                    }
                )
                chunks.append(doc)
                chunk_id += 1
                
    return chunks

if __name__ == "__main__":
    print("Dokümanlar yükleniyor ve parçalanıyor...")
    chunks = load_documents()
    
    print(f"Toplam parça sayısı: {len(chunks)}")
    
    if chunks:
        print("\n--- İlk Parça (0. index) ---")
        print(f"Metadata: {chunks[0].metadata}")
        print(f"Metin:\n{chunks[0].page_content}")
        
    if len(chunks) > 50:
        print("\n--- 50. Parça (50. index) ---")
        print(f"Metadata: {chunks[50].metadata}")
        print(f"Metin:\n{chunks[50].page_content}")
