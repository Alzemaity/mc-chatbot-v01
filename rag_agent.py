import google.generativeai as genai
import faiss
import numpy as np
import json
from typing import List
import os
from PyPDF2 import PdfReader
from docx import Document

genai.configure(api_key="AIzaSyDnuoCp69zfUxizJ5NzqJap-WWl9X7vfpc")

embed_model = "models/embedding-001"
llm_model = "gemini-1.5-flash"

class RAGPipeline:
    def __init__(self, dimension=768):
        self.index = faiss.IndexFlatL2(dimension)
        self.docs: List[str] = []
        self.dimension = dimension

    def embed_text(self, text: str) -> np.ndarray:
        result = genai.embed_content(model=embed_model, content=text)
        return np.array(result["embedding"], dtype="float32")

    def add_document(self, text: str):
        if text.strip():
            embedding = self.embed_text(text)
            self.index.add(np.array([embedding]))
            self.docs.append(text)

    def load_documents(self, path="data.json"):
        with open(path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        for d in docs:
            self.add_document(d)

    def load_files(self, folder="knowledge_base"):
        """Load extra knowledge base files (PDF, DOCX, TXT)"""
        for file in os.listdir(folder):
            filepath = os.path.join(folder, file)
            if file.endswith(".pdf"):
                reader = PdfReader(filepath)
                for page in reader.pages:
                    self.add_document(page.extract_text())
            elif file.endswith(".docx"):
                doc = Document(filepath)
                for para in doc.paragraphs:
                    self.add_document(para.text)
            elif file.endswith(".txt"):
                with open(filepath, "r", encoding="utf-8") as f:
                    self.add_document(f.read())

    def retrieve(self, query: str, k=3) -> List[str]:
        q_emb = self.embed_text(query)
        D, I = self.index.search(np.array([q_emb]), k)
        return [self.docs[i] for i in I[0] if i < len(self.docs)]

    def answer(self, query: str) -> str:
        retrieved_docs = self.retrieve(query)
        context = "\n".join(retrieved_docs)

        prompt = f"""
You are a helpful, friendly assistant for Mansoura College International School.

Rules:
- Answer naturally, as if you know the information yourself.
- Do NOT say things like "Based on the context" or "According to the text".
- If no relevant info is found, simply say:
  "Sorry, I don’t have information about that. Please check with the school administration."

Context:
{context}

Question:
{query}

Answer:
"""

        model = genai.GenerativeModel(llm_model)
        response = model.generate_content(prompt)
        return response.text.strip()
