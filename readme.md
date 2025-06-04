# Penjelasan Kode Project simple-rag

## Overview
Project ini menerapkan konsep Retrieval-Augmented Generation (RAG), yaitu menggabungkan kemampuan pencarian dokumen (retrieval) dengan Large Language Model (LLM) untuk menghasilkan jawaban yang relevan berdasarkan data yang dimiliki.

---

## Struktur Utama Kode

### 1. Proses Indexing (`build.py`)
- **Fungsi utama:** Mengubah kumpulan dokumen menjadi vektor embedding menggunakan model transformer, lalu menyimpan embedding tersebut ke dalam index FAISS untuk pencarian cepat.
- **Langkah-langkah:**
  - Memuat model dan tokenizer dari HuggingFace.
  - Mengubah setiap dokumen menjadi embedding vektor menggunakan mean pooling dan normalisasi.
  - Jika sudah ada index lama, dokumen dan embedding lama digabung dengan yang baru.
  - Semua embedding disimpan ke index FAISS (`index.faiss`), dan dokumen ke file (`docs.pkl`).
- **Hasil:** Sistem siap melakukan pencarian dokumen berbasis vektor.

### 2. Proses Query dan Jawaban (`query.py`)
- **Fungsi utama:** Menjawab pertanyaan user dengan pipeline RAG.
- **Langkah-langkah:**
  - Memuat model, tokenizer, index FAISS, dan dokumen dari file.
  - Mengubah pertanyaan user menjadi embedding vektor.
  - Melakukan pencarian dokumen paling relevan dari index FAISS.
  - Menggabungkan dokumen hasil pencarian ke dalam prompt untuk LLM.
  - Mengirim prompt ke LLM (Gemini API) untuk menghasilkan jawaban.
  - Mengembalikan jawaban ke user.
- **Hasil:** User mendapatkan jawaban yang relevan berdasarkan dokumen yang sudah diindex.

---

## Penjelasan Fungsi Penting

### `load_resources()` (`query.py`)
- Memuat model, tokenizer, index FAISS, dan dokumen dari file.
- Melakukan validasi agar jumlah dokumen dan index sinkron.
- Menggunakan cache agar resource hanya dimuat sekali.

### `generate_embeddings()` (`query.py`)
- Mengubah teks (query atau dokumen) menjadi embedding vektor.
- Proses: tokenisasi, padding, proses model, pooling, dan normalisasi.

### `getAnswer()` (`query.py`)
- Pipeline utama untuk menjawab pertanyaan user.
- Langkah:
  1. Cek ketersediaan index dan dokumen.
  2. Memuat resource yang diperlukan.
  3. Mengubah query menjadi embedding.
  4. Melakukan pencarian dokumen relevan.
  5. Membuat prompt dari dokumen hasil pencarian.
  6. Mengirim prompt ke LLM (Gemini) dan mengambil jawaban.
  7. Mengembalikan jawaban ke user.
- Menangani berbagai error seperti file tidak ditemukan, dimensi index salah, atau error API.

### `runBuild()` (`build.py`)
- Fungsi utama untuk membangun index dari kumpulan dokumen.
- Melakukan embedding dokumen secara batch, normalisasi, dan menyimpan index serta dokumen.
- Mendukung penambahan dokumen baru ke index yang sudah ada.

---

## Alur Kerja Singkat

1. **Indexing:** Jalankan `build.py` untuk membuat index dari dokumen.
2. **Menjawab Pertanyaan:** User bertanya → Query diubah jadi embedding → Cari dokumen relevan → Prompt ke Gemini → Jawaban dihasilkan.

---

## Catatan
- Model yang digunakan: `intfloat/multilingual-e5-base` dari HuggingFace.
- Index pencarian: FAISS (`IndexFlatL2`).
- LLM untuk jawaban: Gemini API.
- Semua proses embedding dan pencarian dilakukan secara efisien dan otomatis.