# gunicorn.conf.py
bind = "0.0.0.0:80"
worker_class = "uvicorn.workers.UvicornWorker"
workers = 1
timeout = 120