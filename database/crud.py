from database.model import chats
from database.db import database

async def create_user(name: str, path: str, index_path: str):
    query = chats.insert().values(name=name, path=path, index_path=index_path)
    return await database.execute(query)

async def get_chats():
    query = chats.select()
    return await database.fetch_all(query)
