"""Styles Blueprint"""

from fastapi import APIRouter

users = APIRouter(
    prefix='/styles',
    tags=['styles'],
)
