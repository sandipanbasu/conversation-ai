export CONVAI_HOME=/Users/sandipanbasu/projects/convai
export PYTHONPATH=$PYTHONPATH:/Users/sandipanbasu/projects/conversation-ai/src

cd fastapi

nohup uvicorn --workers 2 --host 0.0.0.0 --port 8000 app:app & 
