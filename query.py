from functools import lru_cache
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import torch
import faiss
import pickle
import requests
import os
import numpy as np

# 1. Cache Model dan Index (Hidup selama aplikasi berjalan)
@lru_cache(maxsize=None)
def load_resources():
    # Muat model embedding sekali saja
    model_name = "intfloat/multilingual-e5-large"
    huggingFaceKey = os.getenv("HUGGING_FACE_KEY")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingFaceKey)
    model = AutoModel.from_pretrained(model_name, token=huggingFaceKey)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    # Muat index dan dokumen sekali saja
    index = faiss.read_index("./build/index.faiss")
    with open("./build/docs.pkl", "rb") as f:
        docs = pickle.load(f)
    
    return tokenizer, model, device, index, docs

# 2. Fungsi Embedding Terpadu
def generate_embeddings(texts, tokenizer, model, device, is_query=False):
    prefix = "query: " if is_query else "passage: "
    texts = [prefix + text for text in texts]
    
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Pooling canggih dengan normalisasi
    last_hidden = outputs.last_hidden_state
    attention_mask = inputs.attention_mask.unsqueeze(-1)
    embeddings = (last_hidden * attention_mask).sum(1) / attention_mask.sum(1)
    return torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu().numpy()

# 3. Fungsi Jawaban yang Dioptimalkan
def getAnswer(query, top_k=3):
    try:
        docsPkl = Path('./build/docs.pkl')
        faissIndexFile = Path('./build/index.faiss')

        if not docsPkl.exists() or not faissIndexFile.exists():
            raise FileNotFoundError("sistem belum dilatih")

        # Muat sumber daya yang diperlukan
        tokenizer, model, device, index, docs = load_resources()
        
        # 1. Embedding Query (Ultra Cepat)
        query_embedding = generate_embeddings(
            [query], tokenizer, model, device, is_query=True
        ).astype("float32")
        
        # 2. Semantic Search dengan Optimasi GPU
        scores, indices = index.search(query_embedding, top_k)
        
        # 3. Konstruksi Context Cerdas
        context = "\n".join([
            f"[Doc {i+1} | Relevansi: {score:.2f}]: {docs[idx]}"
            for i, (idx, score) in enumerate(zip(indices[0], scores[0]))
        ])
        
        # 4. Dynamic Prompt Engineering
        prompt = f"""
        Anda bertugas menjawab pertanyaan berdasarkan *hanya* dari konteks yang diberikan di bawah. 
        Jika informasi yang dibutuhkan untuk menjawab pertanyaan tidak tersedia atau tidak ditemukan dalam konteks, 
        jawablah secara eksplisit bahwa Anda tidak mengetahui jawabannya. Jangan membuat asumsi atau menambahkan informasi 
        di luar konteks, bahkan jika pengetahuan tersebut terdengar umum atau logis.
        Berikan jawaban yang jelas dan mendalam jika memungkinkan, sejauh yang bisa didukung oleh konteks. 
        Hindari penggunaan frasa pembuka seperti "berdasarkan konteks tersebut" atau kalimat serupa. 
        Fokuskan penjelasan hanya pada pertanyaan, jangan keluar dari topik.
        Konteks:
        {context}
        Pertanyaan:
        {query}
        """

        # 5. Gemini API Call dengan Optimasi
        geminiKey = os.getenv("GEMINI_API_KEY")
        # Kirim ke Gemini
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={geminiKey}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        res = requests.post(url, json=data, headers=headers)
        if res.status_code != 200:
            return f"API Error: {res.status_code} - {res.text}"

        res_json = res.json()
        return res_json["candidates"][0]["content"]["parts"][0]["text"]
            
    except FileNotFoundError:
        return "sistem belum dilatih"
    except Exception as e:
        return f"System Error: {str(e)}"
