from fastapi import FastAPI, File, UploadFile, Request, Form, Response
from aiohttp import ClientSession
from typing import Optional
import sys
import os
import time
from utils_qna_kb import train, ask, semantic_similar

# from twilio.twiml.messaging_response import MessagingResponse


from schemas import (
    TrainFaqResponseBody,
    AskFaqResponseBody
)

# server = Flask(__name__)

app = FastAPI(
    title="ConvAI",
    description="Yet another NLP Pipeline",
    version="0.1",
)

client_session = ClientSession()

CONVAI_HOME = os.environ.get("CONVAI_HOME")
print('ConvAI HOME = ',CONVAI_HOME)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/healthcheck")
async def healthcheck():
    # msg = (
    #     "all a-OK"        
    # )
    return {"message": "all a-OK" }

@app.get('/')
async def home():
    return {"message": 'The epitome of non-sense'}

@app.post('/train_faq', response_model=TrainFaqResponseBody)
async def train_faq(file: UploadFile = File(...)):
    response = {}
    print('file ', file.filename)    
    contents = await file.read()
    fpath = os.path.join(CONVAI_HOME, "uploads", file.filename)
    with open(fpath, 'wb') as f:
        f.write(contents)
    # f.save(os.path.join(FAQ_TRAIN_FOLDER, f.filename))
    print(fpath,"is written to disk.")
    sentence_len = train(data_csv_path = fpath) 
    return {
        "status": "success",
        "sent_count": sentence_len
    }


@app.get('/ask')
async def infer_faq(q: str,corr: Optional[float]=0.5, distance: Optional[int]=10, size: Optional[int] = 3):  
    ans = ask(q,corr_threshold=corr, distance_threshold=distance, num_results=size)
    if(len(ans) > 0):
        return ans
    else:
        return {"No answer found"}
    

@app.get('/similar')
async def similar(sent1: str,sent2: str, corr: Optional[float]=0.5):
    corr = semantic_similar(sent1,sent2)
    return {"corr" : corr}

# @app.post('/whatsapp')
# async def whatapp(Body: str = Form(...)):  
#     # body = b''
#     # async for chunk in request.stream():
#     #     body += chunk    
#     print (Body)
#     # Start our TwiML response
#     resp = MessagingResponse()
#     ans = ask(Body,corr_threshold=.4, distance_threshold=20, num_results=3) 
#     direct_answer = ""
#     try:    
#         direct_answer = ans[0]['answer']    
#     except:    
#         direct_answer = "We are working hard to get you the answer !!"
#     resp.message(direct_answer)
#     return Response(content=str(resp), media_type="application/xml")    
    
        
    
@app.on_event("shutdown")
async def cleanup():
    await client_session.close()        