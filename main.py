"""Main entry point for the LocalAI Backend application."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.database import database_service
from app.database.service import SessionLocal
from app.model_manager import model_manager_service
from app.routers import (
    downloads,
    generators,
    hardware,
    models,
    styles,
    users,
)
from app.socket import socket_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup event"""

    database_service.start()
    db = SessionLocal()
    model_manager_service.unload_model()

    yield

    db.close()

app = FastAPI(
    title='LocalAI Backend',
    description='Backend for Local AI operations.',
    version='0.1.0',
    lifespan=lifespan,
)
app.mount('/static', StaticFiles(directory='static'), name='static')
app.mount('/ws', app=socket_service.sio_app)
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