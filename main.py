from fastapi import FastAPI, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
import os
import asyncio
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from logic import generate_fact_based_seo_content, extract_topics, fetch_news_links, extract_full_news_content, store_in_vector_db

PORT = int(os.getenv("PORT", 8000))

MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = "test_db"
COLLECTION_NAME = "users"

db_client = AsyncIOMotorClient(MONGO_URI)
db = db_client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

async def connect_to_mongo():
    try:
        await db_client.admin.command("ping") 
        print("Connected to MongoDB")
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mongo()
    yield

app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:3000",  
    "http://127.0.0.1:3000",
    "https://newsify-frontend.vercel.app/",
    "https://newsify-frontend-ayushs-projects-d8e7e17a.vercel.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  
)


class QueryInput(BaseModel):
    query: str

@app.post("/generate-seo-content")
async def generate_seo_content(data: QueryInput):
    try:
        r1 = extract_topics(data.query)
        # print(f"Main Topic: {r1['main_topic']}")
        r2 = fetch_news_links(r1["main_topic"], max_articles=3)
        # print(f"Articles: {r2['articles']}")
        r3 = extract_full_news_content(r2["articles"])
        # print(f"Extracted Content: {r3['articles']}")
        r4 = store_in_vector_db(r3["articles"])
        # print(f"Stored in Vector DB: {r4}")
        response = generate_fact_based_seo_content(data.query)
        # print(f"Generated SEO Content: {response}")
        # asyncio.create_task(generate_fact_based_seo_content(data.query))
        # response = "backend working"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Starting FastAPI on port {PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PORT)