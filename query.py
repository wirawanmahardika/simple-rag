from functools import lru_cache
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import torch
import faiss
import pickle
import requests
import os
import numpy as np

# 1. Cache Model dan Index - Dioptimalkan untuk small model
@lru_cache(maxsize=None)
def load_resources():
    model_name = "intfloat/multilingual-e5-small"  # Pastikan menggunakan small
    huggingFaceKey = os.getenv("HUGGING_FACE_KEY")
    
    # Optimasi loading model
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingFaceKey)
    
    # Konfigurasi device otomatis berdasarkan VRAM
    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 4e9 else "cpu")
    
    # Load model dengan low_memory=True jika tersedia
    model = AutoModel.from_pretrained(model_name, token=huggingFaceKey).to(device)
    model.eval()
    
    # Handle missing index files
    index_path = "./build/index.faiss"
    docs_path = "./build/docs.pkl"
    
    if not Path(index_path).exists() or not Path(docs_path).exists():
        raise FileNotFoundError("Index files not found")
    
    index = faiss.read_index(index_path)
    with open(docs_path, "rb") as f:
        docs = pickle.load(f)
    
    return tokenizer, model, device, index, docs

# 2. Fungsi Embedding - Dioptimalkan untuk small model
def generate_embeddings(texts, tokenizer, model, device, is_query=False):
    prefix = "query: " if is_query else "passage: "
    texts = [prefix + text for text in texts]
    
    # Optimasi tokenizer untuk small model
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,  # Diperpendek untuk small model
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        try:
            outputs = model(**inputs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                return generate_embeddings(texts, tokenizer, model, device, is_query)
            raise
    
    # Pooling yang lebih efisien
    last_hidden = outputs.last_hidden_state
    attention_mask = inputs.attention_mask.unsqueeze(-1)
    
    # Mean pooling dengan optimasi memory
    sum_embeddings = (last_hidden * attention_mask).sum(dim=1)
    sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)
    embeddings = sum_embeddings / sum_mask
    
    return torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu().numpy()

# 3. Fungsi Jawaban - Tetap sama dengan pengecekan tambahan
def getAnswer(query, top_k=3, temperature=0.3):
    try:
        # Pengecekan file lebih robust
        if not Path('./build/docs.pkl').exists() or not Path('./build/index.faiss').exists():
            raise FileNotFoundError("Index files not found. Please train the system first.")
        
        tokenizer, model, device, index, docs = load_resources()
        
        # Validasi dimensi
        if index.d != 384:  # Dimensi e5-small adalah 384
            raise ValueError("Index dimension mismatch! Expected 384 for e5-small")
        
        # Generate embedding dengan fallback ke CPU jika perlu
        try:
            query_embedding = generate_embeddings(
                [query], tokenizer, model, device, is_query=True
            ).astype("float32")
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                device = torch.device("cpu")
                model = model.to(device)
                query_embedding = generate_embeddings(
                    [query], tokenizer, model, device, is_query=True
                ).astype("float32")
            else:
                raise
        
        # Pencarian dengan pengecekan hasil
        scores, indices = index.search(query_embedding, top_k)
        
        if len(indices) == 0 or len(scores) == 0:
            return "No relevant documents found"
        
        # Konstruksi context
        context = "\n".join(
            f"[Doc {i+1} | Score: {score:.2f}]: {docs[idx] if idx < len(docs) else 'INVALID_INDEX'}"
            for i, (idx, score) in enumerate(zip(indices[0], scores[0]))
        )
        
        # Prompt engineering untuk small model
        prompt = f"""Berdasarkan konteks berikut, jawab pertanyaan dengan singkat dan tepat:
        
        Konteks:
        {context}

        Pertanyaan: {query}

        Jika informasi tidak cukup, jawab: "Saya tidak menemukan informasi yang relevan"."""
        
        # Gemini API call dengan timeout
        geminiKey = os.getenv("GEMINI_API_KEY")
        if not geminiKey:
            return "Gemini API key not configured"
            
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={geminiKey}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": temperature,
                        "topP": 0.95,
                        "maxOutputTokens": 512
                    }
                },
                headers={"Content-Type": "application/json"},
                timeout=15
            )
            response.raise_for_status()
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except requests.exceptions.RequestException as e:
            return f"API Error: {str(e)}"
            
    except FileNotFoundError:
        return "Error: Sistem belum dilatih. Silakan jalankan training terlebih dahulu."
    except Exception as e:
        return f"Error: {str(e)}"