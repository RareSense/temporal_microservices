from multiprocessing import cpu_count
workers = max(1, cpu_count())
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 60
