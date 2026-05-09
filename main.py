import retrieval
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os, shutil
from fastapi import FastAPI, UploadFile as UF, File
from dotenv import load_dotenv
from typing import List,Annotated
from pydantic import WithJsonSchema
load_dotenv()

app = FastAPI()

model = ChatGoogleGenerativeAI(model='gemini-3-flash-preview',api_key = os.getenv("API_KEY"))

system_prompt = (
    "You are a helpful assistant. "
    "Your goal is to extract and list related data mentioned in the provided context. "
    "\n\n"
    "Context:\n{context}"
)
    
prompt = ChatPromptTemplate.from_messages([
    ("system" , system_prompt),
    ("human" , "{input}")
])

@app.get('/')
def main():
 return {"message" : "Server Running...."}

upload_dir = './uploads'
# Create the RAG chain once
combine_docs_chain = create_stuff_documents_chain(model,prompt)

#Type Hint to specify explicitly the data type
#Annotated -> adds metadata to type, UF -> alias for type, JsonSchema-> to instruct type and format to API Docs
UploadFile = Annotated[UF, WithJsonSchema({"type": "string", "format": "binary"})]

@app.post('/chat-with-pdf')
async def chat(user_input:str,files: Annotated[List[UploadFile] , File(description="Upload multiple PDF files")]):
 file_paths = []
 if files:
   for file in files:
    path = os.path.join(upload_dir, str(file.filename))
 # 1. Save the file locally so PyPDFLoader can read it
    with open(path,"wb") as buffer:
     #shutil -> moving,copying and archiving files and directories
     shutil.copyfileobj(file.file,buffer)
     file_paths.append(path)

 result = retrieval.retrievalData(user_input,file_paths)
#  print(result)
# Does -> Variable Filling(ChatPromptTemplate), Formatting(Doc objs in result -> one giant string), API Call(bundles prompt -> AI Model), Response
 response = combine_docs_chain.invoke({
  "input" : user_input,
  "context" : result
 })
 return {"output" : response}