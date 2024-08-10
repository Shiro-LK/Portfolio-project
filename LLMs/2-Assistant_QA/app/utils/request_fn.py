import requests

def make_request( to_remove=[], mode='listing'):
    url = "http://127.0.0.1:8000/process"
    data = {
        "command": "delete",
        "query": None,
        "binary": None,  # Convert binary data to a hex string for JSON serialization
        "filename": None,
        "delete": to_remove
    }
    
    data2 = {
        "command": "listing",
        "query": None,
        "binary": None,  # Convert binary data to a hex string for JSON serialization
        "filename": None,
        "delete": None
    }
    

    
    if mode == 'listing':
        response = requests.post(url, json=data2)
    else:
        response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()['output']
    return "An error occured."

def make_request_upload( binary_data, filename):
    url = "http://127.0.0.1:8000/process"
    data = {
        "command": "upload",
        "query": None,
        "binary": binary_data.hex(),  # Convert binary data to a hex string for JSON serialization
        "filename": filename
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return True
    return False

def make_request_query(query):
    url = "http://127.0.0.1:8000/process"
    data = {
        "command": "q&a",
        "query": query,
        "binary": None,  # Convert binary data to a hex string for JSON serialization
        "filename": None,
        "delete": None
    }
   
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()['output']
    return "An error occured."