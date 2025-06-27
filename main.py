"""Main entry point for the LocalAI Backend application."""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.blueprints.downloads import downloads
from app.blueprints.hardware import hardware
from app.blueprints.models import models
from app.blueprints.users import users
from app.blueprints.websocket import socket_app

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

app = FastAPI()
app.mount('/ws', socket_app)
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
