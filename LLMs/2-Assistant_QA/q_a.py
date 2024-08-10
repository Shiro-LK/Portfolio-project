import copy, ast, fitz
from langchain.schema.document import Document
def isOverlap(s1,e1,s2,e2):
    if (s1 <= e2 and e1 >= s2) or (e1 >= s2 and s1 <= e2):
        return True
    if (s1 <= s2 and e1 >= e2) or (s1 >= s2 and e1 <= e2):
        return True

def extract_json(response):
    """
        extract the json format of the response from the LLM
    """
    st = response.find('{')
    end = response.find('}')
    
    if st != -1 and end != -1:
        return ast.literal_eval(response[st:end+1])
    return response
def merging_search_results(results_with_scores):
    
    def merge_search_results_per_source(results, starts):
        sorted_results = [x for _, x in sorted(zip(starts, results), key=lambda pair: pair[0])]
        saved = []
        for i, r in enumerate(sorted_results):
            if i == 0:
                saved.append(r)
                continue
            previous_st = saved[-1].metadata['char_start']
            previous_end = saved[-1].metadata['char_end']
            st = r.metadata['char_start']
            end = r.metadata['char_end']
            #print(previous_st, previous_end, st, end)
            if isOverlap(previous_st, previous_end, st, end):
                previous_r = saved.pop(-1)
                new_score = max(previous_r.metadata['score'], r.metadata['score'])
                new_text = previous_r.page_content[:(st-previous_st)] + r.page_content
                
                metadata = copy.deepcopy(r.metadata)
                metadata['score'] = new_score
                metadata['word_start'] = previous_r.metadata['word_start']
                metadata['char_start'] = previous_r.metadata['char_start']
                previous_st_word = previous_r.metadata['word_start']
                st_word = r.metadata['word_start']
                print(previous_r.metadata['word_start'], previous_r.metadata['word_end'], r.metadata['word_start'], r.metadata['word_end'])
                metadata['_bounding_boxes'] = ast.literal_eval(previous_r.metadata['_bounding_boxes'])[:st_word-previous_st_word] + ast.literal_eval(r.metadata['_bounding_boxes'])
                # merge boundingboxes
                #new = TextNode(text=new_text, id_=r.node.id_, metadata=r.node.metadata, start_char_idx=previous_st, end_char_idx=end)
                print('new merging', len(metadata['_bounding_boxes']), len(new_text.split()))
 
                metadata['_bounding_boxes'] = str(metadata['_bounding_boxes'])
                new = Document(page_content=new_text , metadata=metadata)
                saved.append(new)
                
            else:
                saved.append(r)
        
        return saved
    
    results_per_filename = {}
    starts_per_filename = {}
    for result, score in results_with_scores:
        filename = result.metadata['filename']
        result.metadata['score'] = score
        if filename not in results_per_filename:
            results_per_filename[filename] = []
        results_per_filename[filename].append(result)
        
        if filename not in starts_per_filename:
            starts_per_filename[filename] = []
        starts_per_filename[filename].append(result.metadata['char_start'])
    
    new_results = []
    for k in results_per_filename.keys():
        saved = merge_search_results_per_source(results=results_per_filename[k], starts=starts_per_filename[k])
        new_results += saved
    
    new_results.sort(key=lambda x: x.metadata['score'], reverse=True)
    return new_results

def create_message(search_results, question, tokenizer):
    messages = [
        {
            "role": "system",
            "content": """
        You are an assistant which will answer a question based on Search Results provided only. Answer based stricly from the search results only. 
        Do not use your own knowledge. 
        You will return the response in a json format {"response":text, "document":list} where:
        - "response" is the answer to the question
        - "document" is a list of integer and refers to the document index where the answer comes from. 
        To identify the document number you will strictly look for the following string: 'Document index INT:\n\n'.
        """,
        },
        {"role": "user", "content": search_results + '\n\nQuestion:' +  question },
     ]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    return tokenized_chat


def highlight_pdf(filename, bbxs_dict ):
    doc = fitz.open(filename)
    page_min = float('inf')
     
    for bbx in bbxs_dict:

        page_number = bbx['page'] - 1
        page_min = min(page_min, page_number)
        page = doc[page_number]
        w, h = page.rect.width, page.rect.height
        inst = fitz.Rect(bbx['bbx'][0] * w, bbx['bbx'][1] * h, bbx['bbx'][2] * w, bbx['bbx'][3] * h)  
        highlight = page.add_highlight_annot(inst)
        highlight.update()
    
    doc.save(r"tmp/temp_highlighted.pdf")    
    return page_min