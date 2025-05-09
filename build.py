from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import pickle
import numpy as np
import os

def runBuild(docs):
    huggingFaceKey = os.getenv("HUGGING_FACE_KEY")
    model_name = "intfloat/multilingual-e5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingFaceKey)
    model = AutoModel.from_pretrained(model_name, token=huggingFaceKey)
    
    # Pastikan model di CPU (atau ganti jadi 'cuda' jika Anda pakai GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    def get_embedding(text):
        inputs = tokenizer(f"passage: {text}", return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        mask = inputs['attention_mask']
        last_hidden = outputs.last_hidden_state.masked_fill(~mask[..., None].bool(), 0.0)
        embeddings = last_hidden.sum(dim=1) / mask.sum(dim=1)[..., None]
        
        # Kembalikan ke CPU sebelum konversi ke numpy
        return embeddings[0].detach().cpu().numpy()

    try:
        embeddings = [get_embedding(doc) for doc in docs]

        # Simpan FAISS index
        dim = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype("float32"))  # FAISS butuh float32

        os.makedirs("./build", exist_ok=True)

        with open("./build/docs.pkl", "wb") as f:
            pickle.dump(docs, f)
        faiss.write_index(index, "./build/index.faiss")
        return None
    except Exception as e:
        return f"Terjadi Kesalahan: {str(e)}"
