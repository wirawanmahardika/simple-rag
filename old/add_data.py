from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
import faiss
import numpy as np
import os
import pickle

def addMoreData(docs, append=False):
    # 1. Load Model Embedding
    huggingFaceKey = os.getenv("HUGGING_FACE_KEY")
    model_name = "intfloat/multilingual-e5-large"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingFaceKey)
    model = AutoModel.from_pretrained(model_name, token=huggingFaceKey)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 2. Fungsi Embedding dengan Batch
    def get_embeddings(texts, batch_size=2):
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Memproses dokumen"):
            batch = texts[i:i+batch_size]
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
            
            last_hidden = outputs.last_hidden_state
            attention_mask = inputs.attention_mask.unsqueeze(-1)
            mean_embeddings = (last_hidden * attention_mask).sum(1) / attention_mask.sum(1)
            mean_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)
            
            embeddings.append(mean_embeddings.cpu().numpy())
        
        return np.concatenate(embeddings, axis=0)

    try:
        # 3. Generate Embeddings untuk dokumen baru
        new_embeddings = get_embeddings(docs)
        dim = new_embeddings.shape[1]
        
        # 4. Handle Append Mode
        os.makedirs("./build", exist_ok=True)
        
        if append and os.path.exists("./build/index.faiss"):
            # Load existing index and docs
            index = faiss.read_index("./build/index.faiss")
            with open("./build/docs.pkl", "rb") as f:
                existing_docs = pickle.load(f)
            
            # Check dimensionality match
            assert index.d == dim, f"Dimensionality mismatch: Index has {index.d}D, new embeddings have {dim}D"
            
            # Append new data
            index.add(new_embeddings.astype('float32'))
            updated_docs = existing_docs + docs
        else:
            # Create new index
            index = faiss.IndexFlatL2(dim)
            index.add(new_embeddings.astype('float32'))
            updated_docs = docs
        
        # 5. Save Updated Data
        with open("./build/docs.pkl", "wb") as f:
            pickle.dump(updated_docs, f)
        
        faiss.write_index(index, "./build/index.faiss")
        
        print(f"Sukses {'menambahkan' if append else 'membangun'} index!")
        print(f"Total dokumen sekarang: {len(updated_docs)}")
        return None
    
    except Exception as e:
        return f"Error: {str(e)}"
