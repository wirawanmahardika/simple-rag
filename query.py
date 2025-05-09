from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import pickle
import requests
import os

def getAnswer(query):
    # Konfigurasi model
    model_name = "intfloat/multilingual-e5-large"
    huggingFaceKey = os.getenv("HUGGING_FACE_KEY")
    geminiKey = os.getenv("GEMINI_API_KEY")

    # Load tokenizer dan model
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingFaceKey)
    model = AutoModel.from_pretrained(model_name, token=huggingFaceKey)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    def get_query_embedding(query):
        inputs = tokenizer(f"query: {query}", return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        mask = inputs['attention_mask']
        last_hidden = outputs.last_hidden_state.masked_fill(~mask[..., None].bool(), 0.0)
        embeddings = last_hidden.sum(dim=1) / mask.sum(dim=1)[..., None]
        return embeddings[0].detach().cpu().numpy()

    try:
        # Load FAISS index dan dokumen
        index = faiss.read_index("./build/index.faiss")
        with open("./build/docs.pkl", "rb") as f:
            docs = pickle.load(f)

        # Buat embedding dari query
        embedding = get_query_embedding(query)
        D, I = index.search(embedding.reshape(1, -1), k=2)

        # Ambil dokumen relevan
        retrieved_docs = "\n".join([docs[i] for i in I[0]])

        # Buat prompt untuk Gemini
        prompt = f"""
        Anda bertugas menjawab pertanyaan berdasarkan *hanya* dari konteks yang diberikan di bawah. 
        Jika informasi yang dibutuhkan untuk menjawab pertanyaan tidak tersedia atau tidak ditemukan dalam konteks, 
        jawablah secara eksplisit bahwa Anda tidak mengetahui jawabannya. Jangan membuat asumsi atau menambahkan informasi 
        di luar konteks, bahkan jika pengetahuan tersebut terdengar umum atau logis.

        Berikan jawaban yang jelas dan mendalam jika memungkinkan, sejauh yang bisa didukung oleh konteks. 
        Hindari penggunaan frasa pembuka seperti "berdasarkan konteks tersebut" atau kalimat serupa—langsung masuk ke inti jawaban. 
        Fokuskan penjelasan hanya pada pertanyaan, jangan keluar dari topik.

        Konteks:
        {retrieved_docs}

        Pertanyaan:
        {query}
        """

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

    except Exception as e:
        return f"Terjadi Kesalahan: {str(e)}"
