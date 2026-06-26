import os
import json
import logging
import datetime
import shutil
import platform
import uuid
import time
import contextvars
from logging.handlers import RotatingFileHandler

# Context variables for distributed tracing
correlation_id_var = contextvars.ContextVar('correlation_id', default='-')
pipeline_run_id_var = contextvars.ContextVar('pipeline_run_id', default='-')
session_id_var = contextvars.ContextVar('session_id', default='-')

LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'logs'))
ARCHIVE_DIR = os.path.join(LOGS_DIR, 'archive')
RESPONSES_DIR = os.path.join(LOGS_DIR, 'responses')
WEATHER_DIR = os.path.join(LOGS_DIR, 'weather')
GEMINI_DIR = os.path.join(LOGS_DIR, 'gemini')
TEMP_UPLOADS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'temp', 'uploads'))
TEMP_PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'temp', 'processed'))
TEMP_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'temp', 'cache'))

for d in [LOGS_DIR, ARCHIVE_DIR, RESPONSES_DIR, WEATHER_DIR, GEMINI_DIR, TEMP_UPLOADS_DIR, TEMP_PROCESSED_DIR, TEMP_CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

class ContextFormatter(logging.Formatter):
    def format(self, record):
        record.correlation_id = correlation_id_var.get()
        record.pipeline_run_id = pipeline_run_id_var.get()
        record.session_id = session_id_var.get()
        return super().format(record)

def custom_namer(default_name):
    """Names the rotated log file to move it to the archive folder"""
    base_dir, file_name = os.path.split(default_name)
    return os.path.join(ARCHIVE_DIR, file_name)

def get_logger(name: str, log_filename: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Ensure all logs go to errors.log if level is WARNING/ERROR
    error_handler = RotatingFileHandler(
        os.path.join(LOGS_DIR, 'errors.log'), maxBytes=10*1024*1024, backupCount=5
    )
    error_handler.namer = custom_namer
    error_handler.setLevel(logging.WARNING)
    formatter = ContextFormatter('%(asctime)s | %(levelname)s | %(name)s | Session:%(session_id)s | Run:%(pipeline_run_id)s | Corr:%(correlation_id)s | %(message)s')
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    if log_filename:
        file_handler = RotatingFileHandler(
            os.path.join(LOGS_DIR, log_filename), maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.namer = custom_namer
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Optional console handler for Streamlit viewing
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    return logger

def save_raw_payload(directory: str, filename: str, content: str):
    path = os.path.join(directory, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def append_to_manifest(run_id: str, session_id: str, workflow: str, start_time: str, end_time: str, status: str, original_image: str, stored_image: str):
    manifest_path = os.path.join(LOGS_DIR, 'manifest.json')
    entry = {
        "pipeline_run_id": run_id,
        "session_id": session_id,
        "image": {
            "original": original_image,
            "stored": stored_image
        },
        "workflow": workflow,
        "started": start_time,
        "finished": end_time,
        "status": status
    }
    
    # Simple append as JSON lines for massive scale, or read-modify-write if small. We will read-modify-write for strict JSON.
    data = []
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    data = json.loads(content)
        except Exception:
            pass
            
    data.append(entry)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def log_environment_snapshot():
    env_log = os.path.join(LOGS_DIR, 'environment.log')
    if os.path.exists(env_log):
        return  # Only log once per application startup
        
    import tensorflow as tf
    try:
        import streamlit
        st_ver = streamlit.__version__
    except ImportError:
        st_ver = "Unknown"
        
    snapshot = {
        "timestamp": datetime.datetime.now().isoformat(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "streamlit": st_ver,
        "cuda_available": tf.test.is_built_with_cuda(),
        "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0
    }
    with open(env_log, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2)

def log_config_snapshot(workflow_mode: str, adk_enabled: bool):
    config_log = os.path.join(LOGS_DIR, 'config.log')
    snapshot = {
        "timestamp": datetime.datetime.now().isoformat(),
        "workflow_mode": workflow_mode,
        "adk_enabled": adk_enabled,
        "gemini_model": "gemini-2.5-flash",
        "weather_api_timeout": 10,
        "weather_api_retries": 3
    }
    with open(config_log, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2)

def cleanup_temp_files(max_age_hours: int = 24):
    """Removes temporary files older than max_age_hours."""
    temp_dirs = [TEMP_UPLOADS_DIR, TEMP_PROCESSED_DIR, TEMP_CACHE_DIR]
    now = time.time()
    max_age_seconds = max_age_hours * 3600
    cleaned_count = 0
    
    for d in temp_dirs:
        if not os.path.exists(d):
            continue
            
        for filename in os.listdir(d):
            if filename == ".gitkeep":
                continue
                
            filepath = os.path.join(d, filename)
            if os.path.isfile(filepath):
                file_age = now - os.path.getmtime(filepath)
                if file_age > max_age_seconds:
                    try:
                        os.remove(filepath)
                        cleaned_count += 1
                    except Exception as e:
                        print(f"Failed to delete {filepath}: {e}")
                        
    return cleaned_count

