import requests

url = "http://127.0.0.1:8000/process"

"""
# Read the binary content of a PDF file
with open("../Easy_recipes.pdf", "rb") as file:
    binary_data = file.read()

data = {
    "command": "upload",
    "query": None,
    "binary": binary_data.hex(),  # Convert binary data to a hex string for JSON serialization
    "filename": "Easy_recipes.pdf"
}



 
response = requests.post(url, json=data)
print(response)
print(response.json())
 

# Read the binary content of a PDF file
with open("../FAQ_Vitality.pdf", "rb") as file:
    binary_data = file.read()

data = {
    "command": "upload",
    "query": None,
    "binary": binary_data.hex(),  # Convert binary data to a hex string for JSON serialization
    "filename": "FAQ_Vitality.pdf"
}

response = requests.post(url, json=data)
print(response)
print(response.json())
 
"""
"""
data = {
    "command": "remove",
    "query": None,
    "binary": None,  # Convert binary data to a hex string for JSON serialization
    "filename": None,
    "to_remove": ['FAQ_Vitality.pdf']
}


"""


data = {
    "command": "listing",
    "query": None,
    "binary": None,  # Convert binary data to a hex string for JSON serialization
    "filename": None,
    "delete": None
}
"""
data = {
    "command": "remove",
    "query": None,
    "binary": None,  # Convert binary data to a hex string for JSON serialization
    "filename": None,
    "delete": ['Easy_recipes.pdf.pdf']
}
"""
response = requests.post(url, json=data)
print(response)
print(response.json())
 
data = {
    "command": "q&a",
    "query": "What are the ingredients for making pad thai?",
    "binary": None,  # Convert binary data to a hex string for JSON serialization
    "filename": None,
    "delete": None
}

response = requests.post(url, json=data)
print(response)
print(response.json())