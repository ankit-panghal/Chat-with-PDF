from langchain_community.document_loaders import PyPDFDirectoryLoader #Reader
from langchain_text_splitters import RecursiveCharacterTextSplitter # Chunks dividieren
from langchain_huggingface import HuggingFaceEmbeddings # Embedding Modell
from langchain_chroma import Chroma # Vector DB


persist_directory = './my_vector_db'

# 1. Initialize the Embedding Model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",cache_folder = "./my_model_cache")

def retrievalData(query,file_paths):
    # Loads DB
    vector_db = Chroma(persist_directory=persist_directory,embedding_function=embeddings) 
    if file_paths:
       for path in file_paths:
        existing = vector_db.get(where={"source" : path})
        if not existing['ids']:
          loader = PyPDFDirectoryLoader(path='./uploads')
          # loader = PyPDFLoader(path) # Give Instructions, how to Read
          docs = loader.load() # Reads

        # 2. Initialize the splitter
        # chunk_size: The target length of each piece of text.
        # chunk_overlap: A small 'buffer' so the end of one chunk 
        #                matches the start of the next for context.
          text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 2500,
            chunk_overlap = 150,
            add_start_index = True # Helps track exactly where in the PDF this chunk came from
        )

          chunks = text_splitter.split_documents(docs)
          vector_db.add_documents(chunks)
        # print(chunks[0].page_content)

        # for i, chunk in enumerate(chunks):
        #     print(f"{i+1} -> {chunk}")

        else:
      # Loads already created DB
         vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
        )
    
    # Find the top 10 most relevant chunks
    return vector_db.similarity_search(query,k=10)