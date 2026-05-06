import retrieval
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os, shutil
from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

llm = ChatGoogleGenerativeAI(model='gemini-3-flash-preview',api_key = os.getenv("API_KEY"))

system_prompt = (
    "You are a helpful assistant. "
    "Your goal is to extract and list EVERY data mentioned in the provided context. "
    "Break the line when there in '\n\n' instead of writing it. Make Pointers to understand the output"
    "\n\n"
    "Context:\n{context}"
)
    
prompt = ChatPromptTemplate.from_messages([
    ("system" , system_prompt),
    ("human" , "{input}")
])

@app.get('/')
def main():
 return {"message" : "home"}

upload_dir = './uploads'

# Create the RAG chain once
combine_docs_chain = create_stuff_documents_chain(llm,prompt)

@app.post('/chat-with-pdf')
def chat(user_input : str,file: UploadFile = File(...)):
 # 1. Save the file locally so PyPDFLoader can read it
 file_path = os.path.join(upload_dir, str(file.filename))
 with open(file_path,"wb") as buffer:
    shutil.copyfileobj(file.file,buffer)

 result = retrieval.retrievalData(user_input,file_path)
#  print(result)
 response = combine_docs_chain.invoke({
  "input" : user_input,
  "context" : result
 })
 return {"output" : response}