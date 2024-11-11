from fastapi import APIRouter

from app.api.routes import items, login, users, utils, detection, reporting

api_router = APIRouter()
api_router.include_router(login.router, tags=["login"])
api_router.include_router(detection.router,  prefix="/detect", tags=["detection"])
api_router.include_router(reporting.router,  prefix="/report", tags=["reporting"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(utils.router, prefix="/utils", tags=["utils"])
api_router.include_router(items.router, prefix="/items", tags=["items"])
