import json
from groq import Groq
from pygooglenews import GoogleNews
from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field
import time
import os 
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
grok_api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=grok_api_key)

app = FirecrawlApp(api_key = firecrawl_api_key)  # Replace with actual API key


# ✅ Initialize ChromaDB (Persistent Storage)
chroma_client = chromadb.PersistentClient(path="./vector_store")
collection = chroma_client.get_or_create_collection("news_articles")

# ✅ Use Mistral-compatible embeddings (Ensure consistency)
embedding_model = SentenceTransformer("thenlper/gte-small")

# ✅ Use Llama3-70B-8192 for Content Generation
llm = ChatGroq(model_name="llama3-70b-8192", api_key=grok_api_key)

# ✅ Hallucination Grader Data Model
class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Answer is grounded in the facts, '1' (yes) or '0' (no')")
    explanation: str = Field(description="Explain the reasoning for the score")

def handle_rate_limit(func, *args, **kwargs):
    """Handles rate limit errors by waiting before retrying."""
    retries = 3
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "rate_limit_exceeded" in str(e):
                wait_time = 65
                print(f"⚠️ Rate limit exceeded. Waiting {wait_time}s before retrying...")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("🚨 Max retries exceeded. Could not complete the request.")

# ✅ Step 1: Retrieve Relevant Articles from ChromaDB
def retrieve_relevant_articles(user_query, top_k=5):
    query_embedding = embedding_model.encode(user_query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    retrieved_articles = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            retrieved_articles.append({
                "rank": i + 1,
                "title": results["metadatas"][0][i]["title"],
                "url": results["metadatas"][0][i]["url"],
                "published_at": results["metadatas"][0][i].get("published_at", "Unknown"),
                "author": results["metadatas"][0][i].get("author", "Unknown"),
                "content": doc[:1000]  # ✅ Limit article content to 1000 characters
            })
    else:
        print("⚠️ No relevant articles found!")

    return retrieved_articles

# ✅ Step 2: Extract Location Using Llama3
def extract_location_from_content(news_content):
    location_prompt = f"""
    Identify the geographic location (city, state, country) from the following news article content.
    If no specific location is mentioned, return "Unknown".

    Content: {news_content[:300]}  # ✅ Limit input size

    Provide output in JSON: {{"location": "City/State/Country"}}
    """

    response = handle_rate_limit(llm.invoke, location_prompt)

    try:
        location_data = json.loads(response.content)
        return location_data.get("location", "Unknown")
    except json.JSONDecodeError:
        return "Unknown"

# ✅ Step 3: Generate SEO-Optimized Content with Auto-Retry for Hallucinations
def generate_fact_based_seo_content(user_query, max_retries=3):
    retrieved_articles = retrieve_relevant_articles(user_query)

    if not retrieved_articles:
        return {
            "query": user_query,
            "seo_optimized_article": "No relevant news articles found in stored data.",
            "hallucination_score": 0,
            "explanation": "No articles available for this topic.",
            "source": "vector_database"
        }

    for article in retrieved_articles:
        article["location"] = extract_location_from_content(article["content"])

    retry_count = 0
    while retry_count < max_retries:
        prompt = f"""
        You are an expert SEO content writer. Generate an SEO-optimized news summary.

        **SEO Rules:**
        - Use the main keyword in the title and first 100 words.
        - Create an engaging **H1** title (max 60 characters).
        - Write a **meta description** (150-160 characters).
        - Use **H2 & H3 subheadings** for structure.
        - Include **internal & external links**.
        - Readable paragraphs.

        **Summarized News Articles (with locations extracted):**
        {json.dumps(retrieved_articles, indent=2)}

        **Generate:**
        1. Title (H1)  
        2. SEO-optimized meta description  
        3. Structured article (H2, H3)  
        4. Keyword-rich content  
        5. **Assign location-based subtopics**  

        **Topic:** {user_query}
        """

        response = handle_rate_limit(llm.invoke, prompt)
        generated_content = response.content

        # ✅ Step 4: Hallucination Grading with Auto-Retry
        grading_result = grade_hallucination_with_retry(generated_content, retrieved_articles)

        if grading_result["binary_score"] == "1":
            return {
                "query": user_query,
                "seo_optimized_article": generated_content,
                "hallucination_score": 1,
                "explanation": grading_result["explanation"],
                "source": "vector_database",
                "retrieved_articles": retrieved_articles
            }
        else:
            retry_count += 1
            print(f"⚠️ Hallucination detected! Retrying attempt {retry_count}/{max_retries}...")
            time.sleep(2)

    return {"error": "Failed after retries."}

# ✅ Step 5: Hallucination Grading Function with Retry Mechanism
def grade_hallucination_with_retry(generated_content, retrieved_articles, max_retries=3):
    retry_count = 0
    while retry_count < max_retries:
        result = grade_hallucination(generated_content, retrieved_articles)

        if result["binary_score"] in ["0", "1"]:  # ✅ Valid response
            return result

        retry_count += 1
        print(f"⚠️ Hallucination grading failed. Retrying... {retry_count}/{max_retries}")
        time.sleep(2)

    return {"binary_score": "0", "explanation": "Max retries exceeded. Could not ensure factual correctness."}

# ✅ Step 6: Hallucination Grading LLM Call
def grade_hallucination(generated_content, retrieved_articles):
    hallucination_grader_prompt = f"""
    FACTS: {json.dumps(retrieved_articles, indent=2)[:2000]}  # ✅ Limit input size

    STUDENT ANSWER: {generated_content[:1000]}  # ✅ Limit LLM input size

    Grade this answer:
    - Score 1: Answer is fully grounded in retrieved facts.
    - Score 0: Answer contains hallucinated information.

    Provide output strictly in JSON:
    {{"binary_score": "1" or "0", "explanation": "Step-by-step reasoning"}}
    """

    response = handle_rate_limit(llm.invoke, hallucination_grader_prompt)

    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        print("⚠️ JSON Parsing Failed. Raw Response:", response.content)
        return {"binary_score": "0", "explanation": "Failed to parse JSON response"}

# # ✅ Example Usage:
# user_question = query
# seo_article = generate_fact_based_seo_content(user_question)
# print(json.dumps(seo_article, indent=2))