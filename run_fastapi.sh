export CONVAI_HOME=/home/sandipan/projects/convai
export PYTHONPATH=$PYTHONPATH:/home/sandipan/projects/conversation-ai/src

cd fastapi

nohup uvicorn --workers 2 --host 0.0.0.0 --port 8000 app:app & 
