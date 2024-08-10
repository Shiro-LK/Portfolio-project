from fastapi import FastAPI, UploadFile, File, HTTPException
import binascii
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import io, os, ast
from utils import extract_text, get_text_chunks_langchain, get_existed_sources, delete_source
from q_a import merging_search_results, highlight_pdf, create_message, extract_json
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter #,  TokenTextSplitter
from langchain.schema.document import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from llama_index.core.node_parser import TokenTextSplitter
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import pipeline
import torch
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
#from langchain_chroma import Chroma 
from langchain_community.vectorstores.chroma import Chroma

os.environ['API_TOKEN']= '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'
from huggingface_hub import login
login(token=os.environ['API_TOKEN'])

class CommandInput(BaseModel):
    command: str
    query: Optional[str] = None
    binary: Optional[bytes] = None
    filename: Optional[str] = None
    delete: Optional[List] = None


app = FastAPI()

topk = 20
chunk_size = 128
chunk_overlap = 0.5

MAX_LENGTH=4096

# doctr model
model = ocr_predictor(pretrained=True ).to('cuda')
checkpoint = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)
model_llm = AutoModelForCausalLM.from_pretrained(checkpoint, token=os.environ['API_TOKEN']).to('cuda')
 
# embeddings

model_kwargs = {'trust_remote_code': True}
model_name = "Alibaba-NLP/gte-base-en-v1.5"
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs
)

# Chroma

db_chroma = Chroma(persist_directory="chroma_db", embedding_function=hf_embeddings, 
                   collection_name = "full_docs",
                   collection_metadata={"hnsw:space": "cosine"}
                   )

    


@app.post("/process")
async def process_command(input_data: CommandInput):
    global db_chroma
    
    if input_data.command == "upload":
        if input_data.binary is None:
            raise HTTPException(status_code=400, detail="Binary data is required for upload command" )
            
        if input_data.filename is None:
            raise HTTPException(status_code=400, detail="filename is required for upload command" )
        binary_data = binascii.unhexlify(input_data.binary)
        # save file locally
        with open(f"tmp/{input_data.filename}", "wb") as f:
            f.write(binary_data)

        # apply processing on pdf
        doc = DocumentFile.from_pdf('tmp/' + input_data.filename)
        result = model(doc)
        output = result.export()
        pages, bbxs_pages, bbxs_dict = extract_text(output)
        documents = get_text_chunks_langchain(text=pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap, filename=input_data.filename,
                                 bounding_boxes_dict=bbxs_dict)
        
        db_chroma.add_documents(documents)
        db_chroma.persist()
        # os.remove(f"tmp/{input_data.filename}")
        return JSONResponse(content={"success": True, 'output':''})
    
    elif input_data.command == "listing":
        items = get_existed_sources(db_chroma) 
        return JSONResponse(content={"success": True, "output":items})
    elif input_data.command == "delete":
        if input_data.delete is None:
            raise HTTPException(status_code=400, detail="delete is required for remove command" )
        if type(input_data.delete) != list:
            raise HTTPException(status_code=400, detail="delete should be a list for remove command" )
            
        source_name = set(input_data.delete)
        
        db_chroma = delete_source(db=db_chroma, source_name=source_name)
        items = get_existed_sources(db_chroma) 
        return JSONResponse(content={"success": True, "output":items})
    
    elif input_data.command == "q&a":
        if input_data.query is None:
            raise HTTPException(status_code=400, detail="Query string is required for q&a command" )
        try:
            os.remove("tmp/temp_highlighted.pdf")
        except:
            pass
        question = input_data.query
        # Here you can implement your Q&A logic
        # For simplicity, we'll just return the query as the answer
         
        srs = db_chroma.similarity_search_with_relevance_scores(question, k = topk )
        
        srs_merged =  merging_search_results(srs)
        search_results = 'Search Results:\n\n' 
        count = len(tokenizer.encode(search_results))
        for i, x in enumerate(srs_merged):
            s = f'Document index {i}:\n' + x.page_content + '\n\n----\n\n'
            curr = len(tokenizer.encode(s))
            if count + curr < MAX_LENGTH:
                count += curr
                search_results += s
            else:
                break
         
        print(search_results)
        print('number tokens:', count)
        tokenized_chat = create_message(search_results, question, tokenizer)
        start_input = len(tokenized_chat[0])
        #print('start_input:', start_input)
        generated_ids = model_llm.generate(tokenized_chat.to('cuda'), max_length=4096, do_sample=False)
        outputs = tokenizer.batch_decode([generated_ids[0][start_input:]], skip_special_tokens=True)[0]

        outputs = extract_json(outputs)
        print(type(outputs))
        print(outputs)
        if type(outputs) == dict:
            response, indexes = outputs['response'], outputs['document']
            # find overlapping
            #starts, totals = get_overlapping(search_results=srs_merged, indexes=indexes, texts=" ".join(texts))
            # highligh in pdf
            doc_name = 'tmp/' + srs_merged[indexes[0]].metadata['filename']
            bbxs_dict = ast.literal_eval(srs_merged[indexes[0]].metadata['_bounding_boxes'])
        #    print(bbxs_dict)
            page_min = highlight_pdf(filename=doc_name, bbxs_dict=bbxs_dict )


      
        
        return JSONResponse(content={"success":True,"output": response, "page":page_min})
    
    else:
        raise HTTPException(status_code=400, detail="Invalid command" )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
