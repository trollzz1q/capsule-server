import os

# Подхватываем порт от хостинга (или 80 по умолчанию)
port = os.getenv("PORT", "80")
bind = f"0.0.0.0:{port}"

worker_class = "uvicorn.workers.UvicornWorker"
workers = 1
timeout = 120
capture_output = True
enable_stdio_inheritance = True