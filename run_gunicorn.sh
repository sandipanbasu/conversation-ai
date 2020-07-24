export CONVAI_HOME=/home/sandipan/projects/convai
export PYTHONPATH=$PYTHONPATH:/home/sandipan/projects/conversation-ai/src

cd flask_app

nohup gunicorn -w 1 -b 0.0.0.0:8000 --log-file=/home/sandipan/projects/logs/gunicorn.log wsgi:server --timeout 600 &
