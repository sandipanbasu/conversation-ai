export CONVAI_HOME=/home/sandipan/projects/convai
export PYTHONPATH=$PYTHONPATH:/home/sandipan/projects/conversation-ai/src

cd fastapi

nohup uvicorn --reload --workers 1 --host 0.0.0.0 --port 8000 app:app & 
