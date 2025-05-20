from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
import faiss
import numpy as np
import os
import pickle

def runBuild(docs):
    # 1. Load Model dan Tokenizer
    huggingFaceKey = os.getenv("HUGGING_FACE_KEY")
    model_name = "intfloat/multilingual-e5-small"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingFaceKey)
    model = AutoModel.from_pretrained(model_name, token=huggingFaceKey)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 2. Fungsi Mean Pooling sesuai dokumentasi E5
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden_size)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    # 3. Fungsi untuk menghasilkan embedding dokumen
    def get_embeddings(texts, batch_size=8):
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Memproses dokumen"):
            batch = texts[i:i+batch_size]

            # Tambahkan prefix "passage: " sesuai format E5
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

            # Gunakan mean pooling dan normalisasi
            embedding = mean_pooling(outputs, inputs["attention_mask"])
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

            embeddings.append(embedding.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    try:
        # 4. Generate Embeddings dari dokumen
        embeddings = get_embeddings(docs)

        # 5. Buat FAISS Index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype("float32"))

        # 6. Simpan Index dan Dokumen
        os.makedirs("./build", exist_ok=True)

        with open("./build/docs.pkl", "wb") as f:
            pickle.dump(docs, f)

        faiss.write_index(index, "./build/index.faiss")

        print("Sukses membangun index dengan IndexFlatL2 dan E5-small.")
        return None

    except Exception as e:
        return f"Error: {str(e)}"
