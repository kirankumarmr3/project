import json
import requests
json_data=requests.get("https://api.smith.langchain.com/openapi.json").json()  # load the json data into json_data
from langchain_text_splitters import RecursiveJsonSplitter
json_splitter=RecursiveJsonSplitter(max_chunk_size=40) # not sure how length has been calculated
json_chunks=json_splitter.split_json(json_data)

## The splitter can also output documents
docs=json_splitter.create_documents(texts=[json_data])
for doc in docs[:3]:
    print(doc)