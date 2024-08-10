from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter #,  TokenTextSplitter
from langchain.schema.document import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from llama_index.core.node_parser import TokenTextSplitter


# --------- process text ----------- #

def extract_text(output):
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

def get_text_chunks_langchain(text, chunk_size:int, chunk_overlap:float, filename:str, bounding_boxes_dict:dict):
    def return_indexes_char(chunk, text):
        start = text.find(chunk)
        end = start + len(chunk) - 1
        if start == -1:
            raise('Error')
        return start, end
    
    def find_chunk_indices(text, chunk):
        words = text.split()
        chunk_words = chunk.split()
        chunk_length = len(chunk_words)

        start_idx = -1
        end_idx = -1

        for i in range(len(words) - chunk_length + 1):
            if words[i:i + chunk_length] == chunk_words:
                start_word_idx = i
                end_word_idx = i + chunk_length    # Calculate end index
                return (start_word_idx, end_word_idx - 1)  # -1 to adjust for the last space

        return (start_idx, end_idx)

    
    if type(text) == list:
        text = ' '.join(text)
        
 
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=int(chunk_overlap * chunk_size))# TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=int(chunk_overlap * chunk_size))
    docs = []
    count = 0
    for ind, x in enumerate(text_splitter.split_text(text)):
        st, end = return_indexes_char(x, text)
        st_word, end_word = find_chunk_indices(text, chunk=x)
        docs.append(Document(page_content=x, metadata={'char_start':st, 'char_end':end, 'filename':filename, 'word_start':st_word , 'word_end':end_word,
                                                      'chunk_id':filename + '-' + str(ind),
                                                      '_bounding_boxes':str([bounding_boxes_dict[i] for i in range(st_word, end_word+1)])}))
 
    return docs


# ----- get and delete document on Chroma DB ------- #

def get_existed_sources(db):
    sources = {}
    for x in db.get()['metadatas']:
        if x['filename'] not in sources:
            sources[x['filename']] = 0
        sources[x['filename']] += 1
    return sources

def delete_source(db, source_name:set):
    indexes = []
    print(source_name)
    for i, x in enumerate(db.get()['metadatas']):
        if x['filename'] in source_name:
            indexes.append(i)
    indexes = list(reversed(indexes))
    if len(indexes) > 0:
        db._collection.delete(ids=[db.get()['ids'][i] for i in indexes])
    return db