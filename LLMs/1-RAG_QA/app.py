import gradio as gr
from gradio_pdf import PDF
from transformers import pipeline
from pathlib import Path
from llama_index.core import VectorStoreIndex
from llama_index.core import Document, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever, AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.retrievers import AutoMergingRetriever
import os, ast
from typing import List, Optional
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import pipeline
import torch
import fitz 

if not os.path.exists('tmp/'):
    os.makedirs('tmp')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['API_TOKEN']= 'PUT_YOUR_TOKEN'

login(token=os.environ['API_TOKEN'])

dir_ = Path(__file__).parent

# OCR
ocr_model = ocr_predictor(pretrained=True ).to('cpu')

# EMBEDDINGS
embedding_model =  HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
Settings.embed_model = embedding_model

# LLM Model Llama
checkpoint = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#pipe = pipeline("text-generation", checkpoint, device="cuda", torch_dtype=torch.float16)
# load model in 4-bit
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)
model = AutoModelForCausalLM.from_pretrained(checkpoint, token=os.environ['API_TOKEN']).to('cuda')

# global var
filename_pdf = ''
texts = []
bbxs = []
bbxs_dict = {}
base_retriever = None


def my_id_func(index, document):
    return f"test-{index}"

def extract_json(response):
    """
        extract the json format of the response from the LLM
    """
    st = response.find('{')
    end = response.find('}')
    
    if st != -1 and end != -1:
        return ast.literal_eval(response[st:end+1])
    return response

def extract_text(output:dict):
    """
        convert output of OCR into text
        return:
            @texts: text for each page (list of str)
            @bbxs: bounding boxes for each pages (same order than texts if you applied .split()
            @bbxs_dict: dict of bounding boxes, where keys in the index of the bounding boxes 
    """
    texts = []
    bbxs = []
    bbxs_dict = {}

    count = 0
    for i, page in enumerate(output['pages']):

        text = []
        bbx = []
        dimension =  page['dimensions']
        for blocks in page['blocks']:
            for line in blocks['lines']:
                for word in line['words']:
                    text.append(word['value'])
                    (x1, y1), (x2, y2) = word['geometry']
                    bbx.append([x1 , y1,x2 , y2 ])
                    bbxs_dict[count] = {'page':i+1, 'bbx': bbx[-1]}
                    count += 1
                text[-1] = text[-1] + '\n'
            text[-1] = text[-1] + '\n'
        text = ' '.join(text)
        texts.append(text)
        bbxs.append(bbx)
    return texts, bbxs, bbxs_dict

def get_overlapping(search_results, indexes, texts):
    """
        Search for overlapping between the search results and the entire text from the document. The goal is to find the start index of the overlapping and how much tokens are overlapping when it exists. Usefull for highlighting relevant search results in pdf
    """
    srs = [search_results[i].node.text.split() for i in indexes]
    starts = []
    totals = []
    texts = texts.split()
    for sr in srs:
        N = len(sr )
        for i in range(0, len(texts) - N):
            if sr  == texts[i:i+N]:
                starts.append(i)
                totals.append(N)
                break
    return starts, totals

def highlight_pdf(filename, bbxs_dict, start_indexes, totals):
    doc = fitz.open(filename)
    page_min = float('inf')
    for start_index, total in zip(start_indexes, totals):
        for i in range(start_index, start_index+total):
            bbx = bbxs_dict[i]
            page_number = bbx['page'] - 1
            page_min = min(page_min, page_number)
            page = doc[page_number]
            w, h = page.rect.width, page.rect.height
            inst = fitz.Rect(bbx['bbx'][0] * w, bbx['bbx'][1] * h, bbx['bbx'][2] * w, bbx['bbx'][3] * h)  
            highlight = page.add_highlight_annot(inst)
            highlight.update()
    doc.save(r"tmp/temp.pdf")    
    return page_min


def isOverlap(s1,e1,s2,e2):
    if (s1 <= e2 and e1 >= s2) or (e1 >= s2 and s1 <= e2):
        return True
    if (s1 <= s2 and e1 >= e2) or (s1 >= s2 and e1 <= e2):
        return True


def merging_nodes(results):
    # merge chunks when there is overlapping
    
    starts = [r.node.start_char_idx for r in results]
    sorted_results = [x for _, x in sorted(zip(starts, results), key=lambda pair: pair[0])]
    saved = []
    for i, r in enumerate(sorted_results):
        if i == 0:
            saved.append(r)
            continue
        previous_st = saved[-1].node.start_char_idx
        previous_end = saved[-1].node.end_char_idx
        st = r.node.start_char_idx
        end = r.node.end_char_idx
        print(previous_st, previous_end, st, end)
        if isOverlap(previous_st, previous_end, st, end):
            previous_r = saved.pop(-1)
            new_score = max(previous_r.score, r.score)
            new_text = previous_r.node.text[:(st-previous_st)] + r.node.text
            new = TextNode(text=new_text, id_=r.node.id_, metadata=r.node.metadata, start_char_idx=previous_st, end_char_idx=end)
            saved.append(NodeWithScore.from_dict({'node':new, 'score':new_score}))
        else:
            saved.append(r)
    saved.sort(key=lambda x: x.score, reverse=True)
    return saved

def create_message(search_results, question):
    # use format of LLAMA instruct
    messages = [
        {
            "role": "system",
            "content": """
        You are an assistant which will answer a question based on Search Results provided only. Answer based stricly from the search results only. 
        Do not use your own knowledge. 
        You will return the response in a json format {'response':text, 'document':list} where:
        - response is the answer to the question
        - document is a list of integer and refer to the document number where the response come from. Document number can be found as : 'Document NUMBER:\n\n'
        """,
        },
        {"role": "user", "content": search_results + '\n\nQuestion:' +  question },
     ]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    return tokenized_chat



def qa(question: str, doc_name: str) -> str:
    """
        1. extract text using ocr for the pdf
        2. chunk the text
        3. use vectore store (in memory) and extracting embedding for every chunk and take the top-15
        4. Merge relevant chunks when necessary => search result
        5. Use LLMs (Llama 3.1 8b) to answer the question from the search results
    """
    global filename_pdf  
    global texts
    global bbxs, bbxs_dict
    global base_retriever
    global pdf
    if filename_pdf != doc_name: # if ocr has not been extracted yet
        bbxs_dict = {}
        print('extract text and create VectoreStoreIndex...')
        doc = DocumentFile.from_pdf(doc_name)
        # Analyze
        ocr_model.to(device)
        result = ocr_model(doc)
        ocr_model.to('cpu')
        output = result.export()
        texts, bbxs, bbxs_dict = extract_text(output)
        print('ocr done')
        print('creating VectorStore')
        filename_pdf = doc_name
        doc = Document(text=' '.join(texts))
        index = VectorStoreIndex.from_documents([doc], embed_model=embedding_model, transformations=[TokenTextSplitter(
            chunk_size=128, chunk_overlap=50, id_func=my_id_func 
        )]  )
        base_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=15,
        )
    srs = base_retriever.retrieve(question)
    srs_merged =  merging_nodes(srs)
    search_results = 'Search Results:\n\n' + '\n\n---\n\n'.join([f'Document number {i}:\n' + x.node.text for i, x in enumerate(srs_merged)])
    tokenized_chat = create_message(search_results, question)
    start_input = len(tokenized_chat[0])
    print('start_input:', start_input)
    generated_ids = model.generate(tokenized_chat.to('cuda'), max_length=4096, do_sample=False)
    outputs = tokenizer.batch_decode([generated_ids[0][start_input:]], skip_special_tokens=True)[0]
     
    outputs = extract_json(outputs)
    print(type(outputs))
    print(outputs)
    if type(outputs) == dict:
        response, indexes = outputs['response'], outputs['document']
        # find overlapping
        starts, totals = get_overlapping(search_results=srs_merged, indexes=indexes, texts=" ".join(texts))
        # highligh in pdf

        highlight_pdf(filename=doc_name, bbxs_dict=bbxs_dict, start_indexes=starts, totals=totals)
        return 'tmp/temp.pdf', response
    return doc_name, outputs

 
with gr.Blocks() as demo:
    with gr.Row(): 
        with gr.Column():
            pdf =  PDF(label="PDF")
            btn = gr.Button("Run")

        with gr.Column():
            #inp2 = gr.File(label='Document')
            inp = gr.Textbox(label="Question")
            out = gr.Textbox(label="Response")
        
    
       
    btn.click(fn=qa, inputs=[inp, pdf], outputs=[pdf, out])

if __name__ == "__main__":
    demo.launch(share=True, debug=True)