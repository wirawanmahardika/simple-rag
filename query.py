from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from dotenv import load_dotenv
import torch
import faiss
import pickle
import requests
import os

load_dotenv()
def load_resources(name: str = "nasi padang"):
    model_name = "intfloat/multilingual-e5-base"
    huggingFaceKey = os.getenv("HUGGING_FACE_KEY")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingFaceKey)
    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 4e9 else "cpu")
    model = AutoModel.from_pretrained(model_name, token=huggingFaceKey).to(device)
    model.eval()
    index_path = f"./build/{name}/index.faiss"
    docs_path = f"./build/{name}/docs.pkl"
    if not Path(index_path).exists() or not Path(docs_path).exists():
        raise FileNotFoundError("Index files not found")
    index = faiss.read_index(index_path)
    with open(docs_path, "rb") as f:
        docs = pickle.load(f)
    # Validasi sinkronisasi dokumen dan index
    if len(docs) != index.ntotal:
        raise ValueError(f"Jumlah dokumen ({len(docs)}) dan index ({index.ntotal}) tidak sinkron. Silakan rebuild index.")
    return tokenizer, model, device, index, docs

def generate_embeddings(texts, tokenizer, model, device, is_query=False):
    prefix = "query: " if is_query else "passage: "
    texts = [prefix + text for text in texts]
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,  # Harus sama dengan build.py
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden = outputs.last_hidden_state
    attention_mask = inputs.attention_mask.unsqueeze(-1)
    sum_embeddings = (last_hidden * attention_mask).sum(dim=1)
    sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)
    embeddings = sum_embeddings / sum_mask
    return torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu().numpy()

def getAnswer(query, top_k=15, temperature=0.5, nprobe=10, name: str = "nasi padang"):
    try:
        if not Path(f'./build/{name}/docs.pkl').exists() or not Path(f'./build/{name}/index.faiss').exists():
            return "Error: Sistem belum dilatih. Silakan jalankan training terlebih dahulu."
        tokenizer, model, device, index, docs = load_resources(name=name)
        if hasattr(index, 'nprobe'):
            index.nprobe = nprobe  # optimasi pencarian cluster
        if index.d != 768:
            return "Error: Index dimension mismatch! Expected 768 for e5-base"
        query_embedding = generate_embeddings([query], tokenizer, model, device, is_query=True).astype("float32")
        scores, indices = index.search(query_embedding, top_k)
        if len(indices) == 0 or len(scores) == 0:
            return "No relevant documents found"
        # Pastikan index valid
        context = "\n".join(
            f"[Doc {i+1} | Score: {score:.2f}]: {docs[idx] if 0 <= idx < len(docs) else 'INVALID_INDEX'}"
            for i, (idx, score) in enumerate(zip(indices[0], scores[0]))
        )
        print("context ditemukan", context)
        prompt = f"""Anda adalah LLM dalam sistem RAG yang bertugas menjawab pertanyaan berdasarkan konteks berikut. Jika tidak ada informasi yang relevan, jawab: "Saya tidak menemukan informasi yang relevan". Anda boleh menambahkan informasi terkait, tapi jangan menyimpang dari konteks. Perbaiki jika ada informasi yang kurang tepat. Tulis jawaban secara linear, jangan ada simbol yang tidak nyaman dibaca manusia dan tidak boleh juga ada list. Jangan menjawab dengan awalan seperti "berdasarkan konteks" dan semacamnya

        Konteks:
        {context}

        Pertanyaan: {query}
        """
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
            jawaban = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            return jawaban.strip()
        except requests.exceptions.RequestException as e:
            return f"API Error: {str(e)}"
    except FileNotFoundError:
        return "Error: Sistem belum dilatih. Silakan jalankan training terlebih dahulu."
    except Exception as e:
        return f"Error: {str(e)}"