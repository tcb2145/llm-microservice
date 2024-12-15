from fastapi import FastAPI, HTTPException, BackgroundTasks, status, Request
from pydantic import BaseModel
from fastapi_pagination import Page, add_pagination, paginate
from fastapi_pagination.utils import disable_installed_extensions_check
disable_installed_extensions_check()
from typing import List, Optional
import mysql.connector
from discord import SyncWebhook
import time
import uuid
from fastapi.middleware.cors import CORSMiddleware
import requests
from urllib.parse import quote, urlencode
import os

db_config = {
    'user': 'root',
    'password': 'dbuserdbuser',
    'host': os.environ.get('DB_HOST', '34.46.34.153'),
    'database': 'w4153'
}

app = FastAPI(title='llm')
add_pagination(app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BasicResponse(BaseModel):
    message: str

class LLMResponse(BaseModel):
    llm: str
    content: str

with open('openaiapikey.txt', 'r') as f:
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', f.read())

from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)
print(OPENAI_API_KEY[:10])

# middleware to log all requests
@app.middleware("http")
async def sql_logging(request: Request, call_next):
    start_time = time.perf_counter()

    headers = dict(request.scope['headers'])

    if b'x-correlation-id' not in headers:
        correlation_id = str(uuid.uuid4())
        headers[b'x-correlation-id'] = correlation_id.encode('latin-1')
        request.scope['headers'] = [(k, v) for k, v in headers.items()]
    else:
        correlation_id = request.headers['x-correlation-id']

    response = await call_next(request)

    process_time = time.perf_counter() - start_time
    
    with mysql.connector.connect(**db_config) as conn:
        with conn.cursor(dictionary=True) as cursor:
            query = "INSERT INTO logs (microservice, request, response, elapsed, correlation_id) VALUES (%s, %s, %s, %s, %s)"
            values = ('llm', str(request.url.path), str(response.status_code), int(process_time), correlation_id)
            cursor.execute(query, values)
            conn.commit()
        
    return response

# basic hello world for the microservice
@app.get("/")
def get_microservice() -> BasicResponse:
    """
    Simple endpoint to test and return which microservice is being connected to.
    """
    return BasicResponse(message="hello world from llm microservice")

# get info about llm
@app.get("/llm/info")
def get_llm_info() -> BasicResponse:
    """
    Get basic information about the LLM being used for this API.
    """
    message = str(client.models.retrieve("gpt-4o"))

    return BasicResponse(message=message)

# post response from llm
@app.post("/llm/response")
def post_llm_response(query: str) -> LLMResponse:
    """
    Post a new response from the LLM based on the query.
    """
    llm = "gpt-4o"

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Be generally concise in your responses."},
            {"role": "user", "content": query}
        ]
    )

    content = completion.choices[0].message.content

    return LLMResponse(llm=llm, content=content)
    
# main microservice run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)