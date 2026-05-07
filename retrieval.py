from langchain_community.document_loaders import PyPDFLoader #Reader
from langchain_text_splitters import RecursiveCharacterTextSplitter # Chunks dividieren
from langchain_huggingface import HuggingFaceEmbeddings # Embedding Modell
from langchain_chroma import Chroma # Vector DB
import os

persist_directory = './my_vector_db'

# 1. Initialize the Embedding Model
# This downloads a small model (~100MB) to your computer
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",cache_folder = "./my_model_cache")


def retrievalData(query,file_path):
    vector_db = Chroma(persist_directory=persist_directory,embedding_function=embeddings) # Loads DB
    existing = vector_db.get(where={"source" : file_path})
    if not existing['ids']:
      loader = PyPDFLoader(file_path) # Give Instructions, how to Read
      docs = loader.load() # Reads

    # 2. Initialize the splitter
    # chunk_size: The target length of each piece of text.
    # chunk_overlap: A small 'buffer' so the end of one chunk 
    #                matches the start of the next for context.
      text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 200,
        add_start_index = True # Helps track exactly where in the PDF this chunk came from
    )

      chunks = text_splitter.split_documents(docs)

    # print(chunks[0].page_content)

    # for i, chunk in enumerate(chunks):
    #     print(f"{i+1} -> {chunk}")


    # 2. Initialize the Vector DB and save your chunks
    # 'persist_directory' saves the data to your folder so it's not lost

      vector_db = Chroma.from_documents(
        documents=chunks,
        embedding = embeddings,
        persist_directory= "./my_vector_db"
      )
      return vector_db.similarity_search(query,k=10,filter={"source" : file_path})
    else:
      # Loads already created DB
      vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
        )
    
    # Find the top 10 most relevant chunks
      return vector_db.similarity_search(query,k=10,filter={"source" : file_path})

    # for i,res in enumerate(results):
    #     print(f"{i+1} => {res.page_content}")