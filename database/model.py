from sqlalchemy import Table, Column, Integer, String
from database.db import metadata

chats = Table(
    "docs",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(100)),
    Column("path", String(100)),
    Column("index_path", String(100), unique=True),
)
