"""Main entry point for the LocalAI Backend application."""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.routers import (
    app_socket,
    downloads,
    generators,
    hardware,
    models,
    styles,
    users,
)
from app.services.logger import StreamToLogger

stdout_logger = logging.getLogger('STDOUT')
stderr_logger = logging.getLogger('STDERR')
sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup event to initialize the database."""
    from app.database import init_db
    from app.services.model_manager import model_manager

    init_db()
    model_manager.unload_model()
    asyncio.create_task(model_manager.monitor_download_queue())
    yield


app = FastAPI(
    title='LocalAI Backend',
    description='Backend for Local AI operations.',
    version='0.1.0',
    lifespan=lifespan,
)
app.mount('/static', StaticFiles(directory='static'), name='static')
app.mount('/ws', app=app_socket)
app.include_router(users)
app.include_router(models)
app.include_router(downloads)
app.include_router(hardware)
app.include_router(generators)
app.include_router(styles)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/favicon.ico')
async def favicon():
    static_folder = 'static'
    favicon_path = os.path.join(static_folder, 'favicon.ico')
    return FileResponse(favicon_path, media_type='image/vnd.microsoft.icon')


@app.get('/')
def health_check():
    """Health check endpoint to verify if the server is running."""
    return {'status': 'healthy', 'message': 'LocalAI Backend is running!'}
