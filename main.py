"""Main entry point for the LocalAI Backend application."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.routers.downloads import downloads
from app.routers.hardware import hardware
from app.routers.models import models
from app.routers.users import users
from app.routers.websocket import socket_app

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup event to initialize the database."""
    from app.database import init_db

    init_db()
    logging.info('Database initialized successfully.')
    yield


app = FastAPI(
    title='LocalAI Backend',
    description='Backend for Local AI operations.',
    version='0.1.0',
    lifespan=lifespan,
)
app.mount('/ws', app=socket_app)
app.include_router(users)
app.include_router(models)
app.include_router(downloads)
app.include_router(hardware)
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
