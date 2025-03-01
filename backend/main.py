from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import time
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware
import random
import string
import json  # Importing json to serialize data

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow requests from the frontend (localhost:3000)
    allow_credentials=True,
    allow_methods=["GET"],  # Allow only GET method
    allow_headers=["*"],  # Allow all headers
)

# Function to generate a random word
def generate_random_word(length: int = 5) -> str:
    return ''.join(random.choices(string.ascii_lowercase, k=length))

# Streaming function to yield messages continuously with a random word and a number
def event_stream():
    while True:
        # Create an object with a random word and a number from 1 to 10
        random_word = generate_random_word()
        random_number = random.randint(1, 3)
        data = {"word": random_word, "number": random_number}
        
        # Yield the data as a JSON string
        yield f"data: {json.dumps(data)}\n\n"
        time.sleep(2)  # Adjust the sleep time as per the frequency of updates

@app.get("/stream")
async def stream():
    return StreamingResponse(event_stream(), media_type="text/event-stream")
