from fastapi import FastAPI, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
import os
import asyncio
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from logic import generate_fact_based_seo_content


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
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  
)


@app.get("/users/{email}")
async def get_user(email: str):
   return {"email": f"{email} from backend"}  

class QueryInput(BaseModel):
    query: str

@app.post("/generate-seo-content")
async def generate_seo_content(data: QueryInput):
    try:
        response = generate_fact_based_seo_content(data.query)
        # asyncio.create_task(generate_fact_based_seo_content(data.query))
        # response = "backend working"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))