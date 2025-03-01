from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import time
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow requests from the frontend (localhost:3000)
    allow_credentials=True,
    allow_methods=["GET"],  # Allow only GET method
    allow_headers=["*"],  # Allow all headers
)
# Streaming function to yield messages continuously
def event_stream():
    while True:
        # Simulate real-time data (e.g., from sensors or live updates)
        real_time_data = random.randint(1, 100)  # Replace with actual real-time data
        yield f"data: {real_time_data}\n\n"
        time.sleep(0.5)  # Adjust the sleep time as per the frequency of updates

@app.get("/stream")
async def stream():
    return StreamingResponse(event_stream(), media_type="text/event-stream")
