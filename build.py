from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
import faiss
import numpy as np
import os
import pickle

def runBuild(docs):
    # 1. Load Model Embedding
    huggingFaceKey = os.getenv("HUGGING_FACE_KEY")
    model_name = "intfloat/multilingual-e5-large"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingFaceKey)
    model = AutoModel.from_pretrained(model_name, token=huggingFaceKey)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Mode inferensi

    # 2. Fungsi Embedding dengan Batch (lebih efisien)
    def get_embeddings(texts, batch_size=8):
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Memproses dokumen"):
            batch = texts[i:i+batch_size]
            
            # Format input sesuai spesifikasi model E5
            batch = ["passage: " + text for text in batch]
            
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Mean Pooling dengan normalisasi L2
            last_hidden = outputs.last_hidden_state
            attention_mask = inputs.attention_mask.unsqueeze(-1)
            mean_embeddings = (last_hidden * attention_mask).sum(1) / attention_mask.sum(1)
            mean_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)
            
            embeddings.append(mean_embeddings.cpu().numpy())
        
        return np.concatenate(embeddings, axis=0)

    try:
        # 3. Generate Embeddings
        embeddings = get_embeddings(docs)
        
        # 4. Gunakan IndexFlatL2 (PALING COCOK untuk dataset kecil)
        dim = embeddings.shape[1]  # Dimensionality vector
        index = faiss.IndexFlatL2(dim)  # <-- INI PERUBAHAN UTAMA
        
        # Konversi ke float32 (wajib untuk FAISS)
        index.add(embeddings.astype('float32'))
        
        # 5. Simpan Index dan Dokumen
        os.makedirs("./build", exist_ok=True)
        
        # Simpan dokumen asli (untuk ditampilkan saat retrieval)
        with open("./build/docs.pkl", "wb") as f:
            pickle.dump(docs, f)
        
        # Simpan index FAISS
        faiss.write_index(index, "./build/index.faiss")
        
        print("Sukses membangun index dengan IndexFlatL2!")
        return None
    
    except Exception as e:
        return f"Error: {str(e)}"
    