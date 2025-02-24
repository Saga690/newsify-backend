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

app = FirecrawlApp(api_key=firecrawl_api_key)  # Replace with actual API key

# ✅ Initialize ChromaDB (Persistent Storage)
chroma_client = chromadb.PersistentClient(path="./vector_store")
collection = chroma_client.get_or_create_collection("news_articles")

# ✅ Use a smaller, more memory-efficient embeddings model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Use Llama3-70B-8192 for Content Generation
llm = ChatGroq(model_name="llama3-70b-8192", api_key=grok_api_key)

def extract_topics(user_query):
    prompt = f"""
    You are an AI assistant that extracts the main topic from a news-related query.

    **Rules for Extraction:**
    1. Identify the **main topic**.
    **Examples:**
    **Input:** "Tell me about elections in India"
    **Output:**
    {{
        "main_topic": "India Elections",
    }}

    Now, process the following query and return the result in **valid JSON format**:
    "{user_query}"
    """
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",  # Using Mistral on Groq
        messages=[{"role": "system", "content": "Extract key topics in JSON format."},
                  {"role": "user", "content": prompt}],
        temperature=0.3
    )

    result = response.choices[0].message.content.strip()
    
    try:
        print(result)
        return json.loads(result)
    except json.JSONDecodeError:
        return {"error": "Failed to parse response"}

def fetch_news_links(query, max_articles=3):  # Limit to 3 articles
    gn = GoogleNews()
    search_results = gn.search(query)

    news_links = []
    count = 0

    for entry in search_results["entries"]:
        if count >= max_articles:
            break

        news_links.append({
            "title": entry.title,
            "url": entry.link,
            "published_at": entry.published
        })
        count += 1

    return {"query": query, "articles": news_links}

class NewsArticle(BaseModel):
    title: str = Field(description="The title of the news article")
    author: str = Field(description="The author of the news article")
    publication_date: str = Field(description="The publication date of the news article")
    content: str = Field(description="The full content of the news article")

def extract_full_news_content(articles):
    extracted_news = []

    for article in articles:
        url = article["url"]
        try:
            data = app.scrape_url(
                url,
                params={
                    "formats": ["extract"],
                    "extract": {
                        "schema": NewsArticle.model_json_schema()
                    },
                    "actions": [
                        {"type": "wait", "milliseconds": 2000},
                        {"type": "scroll", "behavior": "smooth"}
                    ]
                }
            )

            extracted_data = data.get("extract", {})
            extracted_news.append({
                "title": extracted_data.get("title", article["title"]),
                "url": url,
                "published_at": extracted_data.get("publication_date", article["published_at"]),
                "author": extracted_data.get("author", "Unknown"),
                "content": extracted_data.get("content", "Content not available.")
            })
        except Exception as e:
            print(f"Error extracting {url}: {e}")

        time.sleep(1.5)

    return {"articles": extracted_news}

def store_in_vector_db(articles):
    for article in articles:
        content = article["content"]
        embedding = embedding_model.encode(content).tolist()

        # Store in vector database
        collection.add(
            documents=[content],
            metadatas=[{"title": article["title"], "url": article["url"], "published_at": article["published_at"], "author": article["author"]}],
            embeddings=[embedding],
            ids=[article["url"]]
        )
    print("✅ News articles stored in vector database!")
    return "✅ News articles stored in vector database!"

# Hallucination Grader Data Model
class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Answer is grounded in the facts, '1' (yes) or '0' (no')")
    explanation: str = Field(description="Explain the reasoning for the score")

def handle_rate_limit(func, *args, **kwargs):
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

# Step 1: Retrieve Relevant Articles from ChromaDB
def retrieve_relevant_articles(user_query, top_k=3):  # Limit to top 3 results
    query_embedding = embedding_model.encode(user_query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    print(results)

    retrieved_articles = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            retrieved_articles.append({
                "rank": i + 1,
                "title": results["metadatas"][0][i]["title"],
                "url": results["metadatas"][0][i]["url"],
                "published_at": results["metadatas"][0][i].get("published_at", "Unknown"),
                "author": results["metadatas"][0][i].get("author", "Unknown"),
                "content": doc[:500]  # Limit content to 500 characters
            })
    else:
        print("⚠️ No relevant articles found!")

    return retrieved_articles

# Step 2: Extract Location Using Llama3
def extract_location_from_content(news_content):
    location_prompt = f"""
    Identify the geographic location (city, state, country) from the following news article content.
    If no specific location is mentioned, return "Unknown".

    Content: {news_content[:200]}  # Limit input size

    Provide output in JSON: {{"location": "City/State/Country"}}
    """
    response = handle_rate_limit(llm.invoke, location_prompt)

    try:
        location_data = json.loads(response.content)
        return location_data.get("location", "Unknown")
    except json.JSONDecodeError:
        return "Unknown"

# Step 3: Generate SEO-Optimized Content with Auto-Retry for Hallucinations
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
        - Create an engaging Title in Bold and larger font 
        - Write a meta description (150-160 characters).
        - Write a readable article with multiple paragraphs if required

        **Summarized News Articles (with locations extracted):**
        {json.dumps(retrieved_articles, indent=2)}

        **Generate:**
        1. Title  
        2. Meta description 
        3. Keyword-rich content  
        4. Location  

        **Topic:** {user_query}
        """

        response = handle_rate_limit(llm.invoke, prompt)
        generated_content = response.content

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

# Step 5: Hallucination Grading with Retry
def grade_hallucination_with_retry(generated_content, retrieved_articles, max_retries=3):
    retry_count = 0
    while retry_count < max_retries:
        result = grade_hallucination(generated_content, retrieved_articles)

        if result["binary_score"] in ["0", "1"]:  
            return result

        retry_count += 1
        print(f"⚠️ Hallucination grading failed. Retrying... {retry_count}/{max_retries}")
        time.sleep(2)

    return {"binary_score": "0", "explanation": "Max retries exceeded. Could not ensure factual correctness."}

# Hallucination Grading LLM Call
def grade_hallucination(generated_content, retrieved_articles):
    hallucination_grader_prompt = f"""
    FACTS: {json.dumps(retrieved_articles, indent=2)[:2000]}  

    STUDENT ANSWER: {generated_content[:1000]}  

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
