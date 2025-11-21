"""Main entry point for the LocalAI Backend application."""

import os

# Configure PyTorch memory allocator BEFORE importing torch (critical for memory management)
# This reduces GPU memory fragmentation which causes OOM errors
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.cores.model_manager import model_manager
from app.database import database_service
from app.database.service import SessionLocal
from app.features.downloads import downloads
from app.features.generators import generators
from app.features.hardware import hardware
from app.features.histories import histories
from app.features.img2img import img2img
from app.features.loras import loras
from app.features.models import models
from app.features.resizes import resizes
from app.features.styles import styles
from app.features.users import users
from app.services import logger_service, platform_service, storage_service
from app.socket import socket_service
from config import STATIC_FOLDER


@asynccontextmanager
async def lifespan(app: FastAPI):
	"""Startup event"""

	logger_service.init()
	storage_service.init()
	platform_service.init()
	database_service.init()

	# Attach the running ASGI loop to the socket service for thread-safe emits
	socket_service.attach_loop(asyncio.get_running_loop())

	db = SessionLocal()
	await model_manager.unload_model_async()

	yield

	model_manager.loader_service.shutdown()
	db.close()


app = FastAPI(
	description='Backend for Local AI operations.',
	title='LocalAI Backend',
	version='0.1.0',
	lifespan=lifespan,
)
app.mount('/static', StaticFiles(directory='static'), name='static')
app.mount('/socket.io', app=socket_service.sio_app)
app.include_router(users)
app.include_router(models)
app.include_router(loras)
app.include_router(downloads)
app.include_router(hardware)
app.include_router(generators)
app.include_router(img2img)
app.include_router(styles)
app.include_router(histories)
app.include_router(resizes)
app.add_middleware(
	CORSMiddleware,
	allow_origins=['*'],
	allow_credentials=True,
	allow_methods=['*'],
	allow_headers=['*'],
)


@app.get('/favicon.ico')
async def favicon():
	static_folder = STATIC_FOLDER
	favicon_path = os.path.join(static_folder, 'favicon.ico')
	return FileResponse(favicon_path, media_type='image/vnd.microsoft.icon')


@app.get('/')
def health_check():
	"""Health check endpoint to verify if the server is running."""
	return {'status': 'healthy', 'message': 'LocalAI Backend is running!'}
