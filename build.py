from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
import faiss
import numpy as np
import os
import pickle

def runBuild(docs):
    huggingFaceKey = os.getenv("HUGGING_FACE_KEY")
    model_name = "intfloat/multilingual-e5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingFaceKey)
    model = AutoModel.from_pretrained(model_name, token=huggingFaceKey)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_embeddings(texts, batch_size=8):
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
            embedding = mean_pooling(outputs, inputs["attention_mask"])
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            embeddings.append(embedding.cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    try:
        embeddings = get_embeddings(docs)
        print("Jumlah dokumen baru:", len(docs))
        print("Shape embeddings baru:", embeddings.shape)
        os.makedirs("./build", exist_ok=True)
        index_path = "./build/index.faiss"
        docs_path = "./build/docs.pkl"

        # Gabungkan dengan data lama jika ada
        if os.path.exists(index_path) and os.path.exists(docs_path):
            with open(docs_path, "rb") as f:
                old_docs = pickle.load(f)
            old_index = faiss.read_index(index_path)
            old_embeddings = np.zeros((old_index.ntotal, embeddings.shape[1]), dtype="float32")
            for i in range(old_index.ntotal):
                old_embeddings[i] = old_index.reconstruct(i)
            all_docs = old_docs + docs
            all_embeddings = np.vstack([old_embeddings, embeddings.astype("float32")])
        else:
            all_docs = docs
            all_embeddings = embeddings.astype("float32")

        print("Total dokumen setelah digabung:", len(all_docs))
        print("Total embeddings:", all_embeddings.shape)

        dim = all_embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(all_embeddings)

        with open(docs_path, "wb") as f:
            pickle.dump(all_docs, f)
        faiss.write_index(index, index_path)

        print("Sukses membangun/menambah index (IndexFlatL2).")
        return None

    except Exception as e:
        print(f"Error: {str(e)}")
        return f"Error: {str(e)}"