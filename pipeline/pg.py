import asyncio
from typing import List
import asyncpg
import os
from datetime import datetime
# Utils & Modelspoetry ad
from utils.col import p
from models.news import Newsletter, NewsletterUpdateEmbedded
from models.rss import ContentUpdate
from dotenv import load_dotenv

load_dotenv()

class PGDatabase:
    """
    A class for interacting with a PostgreSQL database asynchronously.
    This class utilizes the asyncpg library to establish a connection

    Methods:
    - setup_pool: Asynchronously sets up a connection pool.
    - close_pool: Asynchronously closes the connection pool.
    - get_papers: Asynchronously fetches papers from the database
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PGDatabase, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self._DB_CONN_INFO = {
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASS'),
            'host': os.getenv('DB_HOST'),
            'database': os.getenv('DB_NAME'),
            'port': os.getenv('DB_PORT', default=5432),
        }
        self.pool = None
        p.printh(f"PG Instance Created")

    async def setup_pool(self):
        self.pool = await asyncpg.create_pool(**self._DB_CONN_INFO)

    async def close_pool(self):
        if self.pool:
            await self.pool.close()

    async def insert_newsletter(self, newsletter: Newsletter):
        """
        Inserts Newsletter into the database.
        """
        try:
            if self.pool is None:
                await self.setup_pool()
            async with self.pool.acquire() as connection:
                insert_query = """
                INSERT INTO newsletters (
                    uuid,
                    from_email,
                    subject,
                    date,
                    email_body,
                    message_html
                 ) VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT DO NOTHING
                """
                # Execute the INSERT statement for the single Newsletter
                await connection.execute(
                    insert_query,
                    newsletter.uuid,
                    newsletter.from_email,
                    newsletter.subject,
                    newsletter.date,
                    newsletter.email_body,
                    newsletter.message_html
                )
                print(f"Inserted `{newsletter.subject}` Newsletter into the database.")
        except Exception as error:
            print(f"Error inserting newsletter: {newsletter.uuid}\n{error}")

    async def insert_list_newsletter_updates(self, newsletter_updates: List[NewsletterUpdateEmbedded]):
        """
        Inserts a list of newsletter Updates into the database.
        """
        try:
            if self.pool is None:
                await self.setup_pool()
            async with self.pool.acquire() as connection:
                insert_query = """
                INSERT INTO updates (
                    uuid,
                    title,
                    content,
                    date,
                    src,
                    src_uuid,
                    link,
                    lemmatized,
                    resolved_link,
                    embedding
                 ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) ON CONFLICT DO NOTHING
                """
                data = [(
                    newsletter.uuid,
                    newsletter.title,
                    newsletter.content,
                    newsletter.date,
                    newsletter.src,
                    newsletter.src_uuid,
                    newsletter.link,
                    newsletter.lemmatized,
                    newsletter.resolved_link,
                    # Vector list needs to be a String
                    str(newsletter.embedding)
                ) for newsletter in newsletter_updates]
                await connection.executemany(insert_query, data)
                print(f"Inserted `{len(newsletter_updates)}` Newsletter Updates into the database.")
        except Exception as error:
            print(f"Error inserting newsletters: {error}")

    async def insert_list_content_updates(self, content_updates: List[ContentUpdate]):
        """
        Inserts a list of newsletter Updates into the database.
        """
        try:
            if self.pool is None:
                await self.setup_pool()
            async with self.pool.acquire() as connection:
                insert_query = """
                INSERT INTO updates (
                    src_uuid,
                    title,
                    content,
                    src,
                    date,
                    link
                 ) VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT DO NOTHING
                """                

                data = [(
                    content.src_uuid,
                    content.title,
                    content.content,
                    content.src,
                    datetime.strptime(content.date,"%Y-%m-%dT%H:%M:%S%z"),
                    content.link
                ) for content in content_updates]
                await connection.executemany(insert_query, data)
                p.printh(f"Inserted `{len(content_updates)}` Content Updates into the database.")
        except Exception as error:
            p.printf(f"Error inserting newsletters: {error}")

    async def fetch_all_lematized_content(self):
        try:
            if self.pool is None:
                await self.setup_pool()
            async with self.pool.acquire() as connection:
                query = """
                SELECT uuid, lemmatized FROM updates WHERE embedding IS NULL and src = 'newsletter'
                """
                return await connection.fetch(query)
        except Exception as error:
            p.printf(f"Error fetching lemmatized content: {error}")

    async def fetch_all_unembedded_content(self):
        try: 
            if self.pool is None:
                await self.setup_pool()
            async with self.pool.acquire() as connection:
                query = """
                SELECT uuid, title, content FROM updates WHERE embedding IS NULL
                """
                return await connection.fetch(query)
        except Exception as error:
            p.printf(f"Error fetching unembedded content: {error}")

    async def update_embedding(self, uuid, embedding):
        try:
            if self.pool is None:
                await self.setup_pool()
            async with self.pool.acquire() as connection:
                query = """
                UPDATE updates
                SET embedding = $1
                WHERE uuid = $2
                """
                await connection.execute(query, str(embedding), uuid)
        except Exception as error:
            p.printf(f"Error updating embedding: {error}")

    async def update_relevance(self, uuid_boolean: list[(str, bool)]):
        try:
            if self.pool is None:
                await self.setup_pool()
            async with self.pool.acquire() as connection:
                query = """
                UPDATE updates
                SET relevance = $2
                WHERE uuid = $1
                """
                await connection.executemany(query, uuid_boolean)
                print(f"Updated relevance for {len(uuid_boolean)} updates.")
        except Exception as error:
            p.printf(f"Error updating embedding: {error}")

    async def fetch_newsletters(self):
        try:
            if self.pool is None:
                await self.setup_pool()
            async with self.pool.acquire() as connection:

                query = """
                SELECT * FROM newsletters
    
                """
                return await connection.fetch(query)
        except Exception as error:
            p.printf(f"Error fetching newsletters: {error}")

    async def _select(self, sql_query):
        try:
            if self.pool is None:
                await self.setup_pool()
            async with self.pool.acquire() as connection:
                query = sql_query
                return await connection.fetch(query)
        except Exception as error:
            p.printf(f"Error fetching newsletters: {error}")

if __name__ == '__main__':
    from dotenv import load_dotenv
    import pandas as pd
    import time
    # Load .env
    load_dotenv()

    async def collect_csv():
        pg = PGDatabase()
        sql_query = """
        SELECT uuid, title, content, date FROM public.updates
        WHERE src = 'newsletter'
        ORDER BY date DESC LIMIT 50
        """
        res = await pg._select(sql_query)
        # Convert from records to list of Documents
        res = [{
            "id": str(row["uuid"]),
            "title": row["title"],
            "text": f'# {row["title"]} \n {row["content"]}'.strip(),
            # Format the python datetime as a string
            "date": row["date"].strftime("%Y-%m-%d %H:%M:%S"),
        } for row in res]

        # save as csv
        run_id = time.strftime("%Y%m%d-%H%M%S")
        df = pd.DataFrame(res, columns=["id", "title", "text", "date"])
        df.to_csv(f"../bin/data/newsletters-{run_id}.csv", index=False, encoding='utf-8')

    asyncio.run(collect_csv())