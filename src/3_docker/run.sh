export PYTHONPATH="$PYTHONPATH:./engine/"
ai-ocr/bin/python src/app_server.py
# export PYTHONPATH="$PYTHONPATH:./engine/"
# export PYTHONPATH="$PYTHONPATH:./scripts/"
# export PYTHONIOENCODING=utf-8
# export WORKER_COUNT=1
# recapture/bin/gunicorn app_server:app --bind=0.0.0.0:8686 --timeout 500 --workers=${WORKER_COUNT} --threads=5 --worker-connections=1000 --log-level=debug
