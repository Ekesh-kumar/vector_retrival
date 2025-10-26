import os
from typing import Optional, List, Dict
from pathlib import Path
# FastAPI imports
from fastapi import UploadFile
# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
# Environment
from dotenv import load_dotenv
load_dotenv()
# PDF parsing
import fitz  # PyMuPDF
# Google Generative AI
from google import genai

# -------------------- GEMINI EMBEDDINGS --------------------
  
class GeminiEmbeddings(Embeddings):
   
    def __init__(self, model_name: str = "gemini-embedding-001", api_key: Optional[str] = None):
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=texts
            )
            return [embedding.values for embedding in result.embeddings]
        except Exception as e:
            raise RuntimeError(f"Error embedding documents: {str(e)}")

    def embed_query(self, text: str) -> List[float]:
       
        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=[text]
            )
            return result.embeddings[0].values
        except Exception as e:
            raise RuntimeError(f"Error embedding query: {str(e)}")


class RAGVectorStoreCreator:
   
    def __init__(
        self,
        embedding_model: Optional[GeminiEmbeddings] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.embedding_model = embedding_model or self._get_default_embeddings()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def _get_default_embeddings(self) -> GeminiEmbeddings:
       
        return GeminiEmbeddings(
            model_name=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
        )


    async def _load_pdf_with_pymupdf(self, file_content: bytes, filename: str) -> List[Document]:
        
        documents = []
        try:
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                text = page.get_text()
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": page_num + 1,
                            "total_pages": len(pdf_document)
                        }
                    ))
            pdf_document.close()
        except Exception as e:
            raise ValueError(f"Error reading PDF: {str(e)}")
        return documents

    async def _load_text_file(self, file_content: bytes, filename: str) -> List[Document]:
        
        try:
            text = file_content.decode("utf-8")
        except UnicodeDecodeError:
            for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                try:
                    text = file_content.decode(encoding)
                    break
                except Exception:
                    continue
            else:
                raise ValueError("Unable to decode text file with supported encodings")

        return [Document(page_content=text, metadata={"source": filename})]


    async def create_rag_from_file(
        self,
        file: UploadFile,
        vector_store_name: Optional[str] = None,
        persist_directory: Optional[str] = None
    ) -> dict:

        file_content = await file.read()
        filename = file.filename
        file_extension = Path(filename).suffix.lower()

        print(f"📄 Loading file: {filename} ({len(file_content)} bytes)")

        # Load text data
        if file_extension == ".pdf":
            documents = await self._load_pdf_with_pymupdf(file_content, filename)
        elif file_extension == ".txt":
            documents = await self._load_text_file(file_content, filename)
        else:
            raise ValueError(f"Unsupported file type '{file_extension}'. Use .pdf or .txt")

        if not documents:
            raise ValueError("No text content extracted from file")

        print(f"🧩 Splitting {len(documents)} documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")

        # Prepare persistence paths
        vector_store_name = vector_store_name or f"{Path(filename).stem}_chroma"
        persist_dir = persist_directory or f"./chroma_db/{vector_store_name}"
        os.makedirs(persist_dir, exist_ok=True)

        # Create and persist Chroma database
        print(f"🧠 Creating Chroma DB at: {persist_dir}")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory=persist_dir,
            collection_name=vector_store_name
        )
        vector_store.persist()

        print(f"✅ Chroma DB '{vector_store_name}' saved successfully at {persist_dir}")

        return {
            "vector_store_name": vector_store_name,
            "vector_store": vector_store,
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "filename": filename,
            "persist_directory": persist_dir
        }

    def load_existing_vector_store(
        self,
        vector_store_name: str,
        persist_directory: Optional[str] = None
    ):
        persist_dir = persist_directory or f"./chroma_db/{vector_store_name}"
        if not os.path.exists(persist_dir):
            raise FileNotFoundError(f"Chroma DB not found at {persist_dir}")

        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embedding_model,
            collection_name=vector_store_name
        )
        print(f"✅ Loaded Chroma DB: {vector_store_name}")
        return vector_store


    def retrieve_chunks(self, vector_store, query: str, k: int = 4, retrieval_type: str = "similarity") -> List[Document]:
        if retrieval_type == "similarity":
            return vector_store.similarity_search(query, k=k)
        elif retrieval_type == "mmr":
            return vector_store.max_marginal_relevance_search(query, k=k)
        else:
            raise ValueError(f"Unknown retrieval type: {retrieval_type}")

    def retrieve_chunks_with_scores(self, vector_store, query: str, k: int = 4) -> List[tuple]:
        return vector_store.similarity_search_with_score(query, k=k)

    def retrieve_chunks_by_threshold(
        self, vector_store, query: str, score_threshold: float = 0.7, k: int = 10
    ) -> List[Document]:
        docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
        filtered_docs = [doc for doc, score in docs_with_scores if score <= (1 - score_threshold)]
        return filtered_docs

    def retrieve_chunks_formatted(
        self, vector_store, query: str, k: int = 4, include_metadata: bool = True, include_scores: bool = False
    ) -> List[Dict]:
      
        if include_scores:
            docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
        else:
            docs_with_scores = [(doc, None) for doc in vector_store.similarity_search(query, k=k)]

        results = []
        for idx, (doc, score) in enumerate(docs_with_scores, start=1):
            entry = {"rank": idx, "content": doc.page_content}
            if include_metadata:
                entry["metadata"] = doc.metadata
            if include_scores and score is not None:
                entry["similarity_score"] = float(score)
            results.append(entry)
        return results

    def retrieve_as_context(self, vector_store, query: str, k: int = 4, separator: str = "\n\n---\n\n") -> str:
        docs = vector_store.similarity_search(query, k=k)
        return separator.join([doc.page_content for doc in docs])

    def get_retriever(self, vector_store, search_type: str = "similarity", k: int = 4, **kwargs):
        return vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k, **kwargs})
