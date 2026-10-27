bind = "0.0.0.0:80"
worker_class = "uvicorn.workers.UvicornWorker"
workers = 1
timeout = 120
capture_output = True              # <-- Перехватывать print() в логи
enable_stdio_inheritance = True    # <-- Наследовать потоки ввода/вывода