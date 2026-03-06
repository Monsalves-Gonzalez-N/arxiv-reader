from fastapi import FastAPI, APIRouter, HTTPException, Query, Depends, Response, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
import feedparser
import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import re

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url, tls=True, tlsAllowInvalidCertificates=True)
db = client[os.environ.get('DB_NAME', 'arxiv_tiktok')]

# Create the main app
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Auth constants
EMERGENT_AUTH_URL = "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data"
SESSION_EXPIRY_DAYS = 7

# arXiv categories
ASTRO_CATEGORIES = {
    "astro-ph": "Astrophysics (all)",
    "astro-ph.GA": "Galaxies",
    "astro-ph.CO": "Cosmology",
    "astro-ph.EP": "Earth & Planetary",
    "astro-ph.HE": "High Energy",
    "astro-ph.IM": "Instrumentation",
    "astro-ph.SR": "Solar & Stellar",
}

# Models
class User(BaseModel):
    user_id: str
    email: str
    name: str
    picture: Optional[str] = None
    created_at: datetime
    last_seen_paper_date: Optional[str] = None

class SessionDataResponse(BaseModel):
    id: str
    email: str
    name: str
    picture: Optional[str] = None
    session_token: str

class Paper(BaseModel):
    id: str
    title: str
    abstract: str
    authors: List[str]
    published: str
    link: str
    category: str
    comment: Optional[str] = None
    is_recommendation: bool = False
    is_new: bool = False

class LikedPaper(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    published: str
    link: str
    category: str
    liked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class LikeRequest(BaseModel):
    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    published: str
    link: str
    category: str

class PapersResponse(BaseModel):
    papers: List[Paper]
    total: int
    has_more: bool
    new_papers_count: int = 0

# Auth helpers
async def get_session_token(request: Request) -> Optional[str]:
    """Extract session token from cookie or header"""
    # Try cookie first
    session_token = request.cookies.get("session_token")
    if session_token:
        return session_token
    
    # Fall back to Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:]
    
    return None

async def get_current_user(request: Request) -> Optional[User]:
    """Get current user from session token"""
    session_token = await get_session_token(request)
    if not session_token:
        return None
    
    session = await db.user_sessions.find_one(
        {"session_token": session_token},
        {"_id": 0}
    )
    if not session:
        return None
    
    # Check expiry with timezone awareness
    expires_at = session["expires_at"]
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    
    if expires_at < datetime.now(timezone.utc):
        return None
    
    user_doc = await db.users.find_one(
        {"user_id": session["user_id"]},
        {"_id": 0}
    )
    if user_doc:
        return User(**user_doc)
    return None

async def require_auth(request: Request) -> User:
    """Require authentication"""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

# Helper functions
async def get_recent_paper_ids(category: str, max_results: int = 50):
    """Scrape new and cross-listed paper IDs from arXiv new submissions page (excludes replacements).
    Returns (paper_ids, announced_date) where announced_date is the arXiv listing date (YYYY-MM-DD)."""
    cat_map = {
        "astro-ph": "astro-ph",
        "astro-ph.GA": "astro-ph.GA",
        "astro-ph.CO": "astro-ph.CO",
        "astro-ph.EP": "astro-ph.EP",
        "astro-ph.HE": "astro-ph.HE",
        "astro-ph.IM": "astro-ph.IM",
        "astro-ph.SR": "astro-ph.SR",
    }
    cat = cat_map.get(category, "astro-ph")

    url = f"https://arxiv.org/list/{cat}/new"
    try:
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            response = await http_client.get(url)
            response.raise_for_status()

        text = response.text

        # Extract the announcement date from the page header
        # e.g. "Showing new listings for Friday, 6 March 2026"
        announced_date = None
        date_match = re.search(r'Showing new listings for \w+, (\d+ \w+ \d+)', text)
        if date_match:
            try:
                from datetime import datetime as dt
                announced_date = dt.strptime(date_match.group(1), "%d %B %Y").strftime("%Y-%m-%d")
            except Exception:
                pass

        # Split by h3 section headers and only keep new + cross submissions
        parts = re.split(r'<h3[^>]*>', text)
        paper_ids = []
        seen = set()
        for part in parts:
            header_match = re.match(r'(.*?)</h3>', part)
            if not header_match:
                continue
            header = header_match.group(1).lower()
            # Skip replacement submissions (old papers that were revised)
            if 'replacement' in header:
                continue
            for pid in re.findall(r'arXiv:(\d+\.\d+)', part):
                if pid not in seen:
                    seen.add(pid)
                    paper_ids.append(pid)

        return paper_ids[:max_results], announced_date
    except Exception as e:
        logging.error(f"Error fetching recent paper IDs: {e}")
        return [], None

async def get_papers_by_ids(paper_ids: List[str], category: str, announced_date: str = None) -> List[Paper]:
    """Fetch paper details by IDs from arXiv API"""
    if not paper_ids:
        return []
    
    # Batch request (max 50 at a time)
    id_list = ','.join(paper_ids[:50])
    url = f"https://export.arxiv.org/api/query?id_list={id_list}&max_results=50"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            response = await http_client.get(url)
            response.raise_for_status()
            
        feed = feedparser.parse(response.text)
        papers = []
        
        for entry in feed.entries:
            paper_id = entry.id.split('/abs/')[-1]
            title = ' '.join(entry.title.split())
            abstract = ' '.join(entry.summary.split())
            authors = [author.name for author in entry.authors]
            published = announced_date if announced_date else entry.published[:10]

            comment = None
            if hasattr(entry, 'arxiv_comment'):
                comment = entry.arxiv_comment
            elif 'arxiv_comment' in entry:
                comment = entry['arxiv_comment']
            
            papers.append(Paper(
                id=paper_id,
                title=title,
                abstract=abstract,
                authors=authors,
                published=published,
                link=entry.link,
                category=category,
                comment=comment,
                is_new=True
            ))
        
        return papers
    except Exception as e:
        logging.error(f"Error fetching papers by IDs: {e}")
        return []

def parse_arxiv_response(feed_data, category: str, last_seen_date: Optional[str] = None) -> List[Paper]:
    """Parse arXiv API response into Paper objects"""
    papers = []
    for entry in feed_data.entries:
        paper_id = entry.id.split('/abs/')[-1]
        title = ' '.join(entry.title.split())
        abstract = ' '.join(entry.summary.split())
        authors = [author.name for author in entry.authors]
        published = entry.published[:10]
        
        # Get comment (contains info like "Accepted in ApJ" or "Submitted to MNRAS")
        comment = None
        if hasattr(entry, 'arxiv_comment'):
            comment = entry.arxiv_comment
        elif 'arxiv_comment' in entry:
            comment = entry['arxiv_comment']
        
        # Check if paper is new
        is_new = False
        if last_seen_date and published > last_seen_date:
            is_new = True
        
        papers.append(Paper(
            id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            published=published,
            link=entry.link,
            category=category,
            comment=comment,
            is_new=is_new
        ))
    
    return papers

async def get_recommendations(liked_papers: List[dict], all_papers: List[Paper], top_n: int = 10) -> List[Paper]:
    """Get paper recommendations based on liked papers using TF-IDF similarity, prioritizing recent papers"""
    if not liked_papers or not all_papers:
        return []
    
    liked_texts = [f"{p['title']} {p['abstract']}" for p in liked_papers]
    liked_ids = set(p['paper_id'] for p in liked_papers)
    
    candidate_papers = [p for p in all_papers if p.id not in liked_ids]
    
    if not candidate_papers:
        return []
    
    candidate_texts = [f"{p.title} {p.abstract}" for p in candidate_papers]
    all_texts = liked_texts + candidate_texts
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    n_liked = len(liked_texts)
    liked_vectors = tfidf_matrix[:n_liked]
    candidate_vectors = tfidf_matrix[n_liked:]
    
    # Calculate content similarity
    content_similarities = cosine_similarity(candidate_vectors, liked_vectors).mean(axis=1)
    
    # Calculate recency score (papers from last 30 days get higher score)
    today = datetime.now(timezone.utc)
    recency_scores = []
    for paper in candidate_papers:
        try:
            pub_date = datetime.strptime(paper.published, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            days_old = (today - pub_date).days
            # Score: 1.0 for today, decays over 365 days
            recency_score = max(0, 1.0 - (days_old / 365))
        except:
            recency_score = 0.5
        recency_scores.append(recency_score)
    
    recency_scores = np.array(recency_scores)
    
    # Combined score: 70% content similarity + 30% recency
    combined_scores = 0.7 * content_similarities + 0.3 * recency_scores
    
    top_indices = np.argsort(combined_scores)[-top_n:][::-1]
    
    recommendations = []
    for idx in top_indices:
        if combined_scores[idx] > 0.05:
            paper = candidate_papers[idx]
            paper.is_recommendation = True
            recommendations.append(paper)
    
    return recommendations

# Auth Routes
@api_router.post("/auth/session")
async def create_session(request: Request, response: Response):
    """Exchange session_id for session_token"""
    body = await request.json()
    session_id = body.get("session_id")
    
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    
    # Call Emergent Auth API
    async with httpx.AsyncClient() as client:
        auth_response = await client.get(
            EMERGENT_AUTH_URL,
            headers={"X-Session-ID": session_id}
        )
        
        if auth_response.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid session")
        
        user_data = auth_response.json()
    
    session_data = SessionDataResponse(**user_data)
    
    # Create or update user
    user_id = f"user_{uuid.uuid4().hex[:12]}"
    existing_user = await db.users.find_one({"email": session_data.email}, {"_id": 0})
    
    if existing_user:
        user_id = existing_user["user_id"]
    else:
        await db.users.insert_one({
            "user_id": user_id,
            "email": session_data.email,
            "name": session_data.name,
            "picture": session_data.picture,
            "created_at": datetime.now(timezone.utc),
            "last_seen_paper_date": None
        })
    
    # Create session
    expires_at = datetime.now(timezone.utc) + timedelta(days=SESSION_EXPIRY_DAYS)
    await db.user_sessions.insert_one({
        "user_id": user_id,
        "session_token": session_data.session_token,
        "expires_at": expires_at,
        "created_at": datetime.now(timezone.utc)
    })
    
    # Set cookie
    response.set_cookie(
        key="session_token",
        value=session_data.session_token,
        httponly=True,
        secure=True,
        samesite="none",
        max_age=SESSION_EXPIRY_DAYS * 24 * 60 * 60,
        path="/"
    )
    
    return {
        "user_id": user_id,
        "email": session_data.email,
        "name": session_data.name,
        "picture": session_data.picture
    }

@api_router.get("/auth/me")
async def get_me(request: Request):
    """Get current user"""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

@api_router.post("/auth/logout")
async def logout(request: Request, response: Response):
    """Logout user"""
    session_token = await get_session_token(request)
    if session_token:
        await db.user_sessions.delete_one({"session_token": session_token})
    
    response.delete_cookie(key="session_token", path="/")
    return {"message": "Logged out"}

# Paper Routes
@api_router.get("/")
async def root():
    return {"message": "ArXiv TikTok API"}

@api_router.get("/categories")
async def get_categories():
    return {"categories": ASTRO_CATEGORIES}

@api_router.get("/papers", response_model=PapersResponse)
async def get_papers(
    request: Request,
    category: str = Query(default="astro-ph"),
    start: int = Query(default=0, ge=0),
    max_results: int = Query(default=20, ge=1, le=50)
):
    """Fetch papers from arXiv API - newest first like TikTok"""
    user = await get_current_user(request)
    last_seen_date = user.last_seen_paper_date if user else None
    
    base_url = "https://export.arxiv.org/api/query"
    query = f"cat:{category}"
    url = f"{base_url}?search_query={query}&start={start}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            response = await http_client.get(url)
            response.raise_for_status()
            
        feed = feedparser.parse(response.text)
        papers = parse_arxiv_response(feed, category, last_seen_date)
        total = int(feed.feed.get('opensearch_totalresults', 0))
        
        # Count new papers
        new_papers_count = sum(1 for p in papers if p.is_new)
        
        return PapersResponse(
            papers=papers,
            total=total,
            has_more=start + len(papers) < total,
            new_papers_count=new_papers_count
        )
    except Exception as e:
        logging.error(f"Error fetching papers: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching papers: {str(e)}")

@api_router.get("/papers/feed", response_model=PapersResponse)
async def get_paper_feed(
    request: Request,
    category: str = Query(default="astro-ph"),
    start: int = Query(default=0, ge=0),
    max_results: int = Query(default=20, ge=1, le=50)
):
    """Get paper feed - newest papers first from arXiv new submissions"""
    user = await get_current_user(request)
    user_id = user.user_id if user else "anonymous"

    # Support comma-separated categories
    categories = [c.strip() for c in category.split(",") if c.strip()]
    if not categories:
        categories = ["astro-ph"]

    try:
        import asyncio
        if len(categories) == 1:
            recent_ids, announced_date = await get_recent_paper_ids(categories[0], max_results=100)
            primary_category = categories[0]
        else:
            # Fetch IDs for each category in parallel and merge
            results = await asyncio.gather(*[get_recent_paper_ids(c, max_results=100) for c in categories])
            seen_set = set()
            recent_ids = []
            announced_date = None
            for cat_ids, cat_date in results:
                if announced_date is None:
                    announced_date = cat_date
                for pid in cat_ids:
                    if pid not in seen_set:
                        seen_set.add(pid)
                        recent_ids.append(pid)
            primary_category = categories[0]

        if not recent_ids:
            # Fallback to arXiv API with OR query
            base_url = "https://export.arxiv.org/api/query"
            or_query = " OR ".join(f"cat:{c}" for c in categories)
            url = f"{base_url}?search_query={or_query}&start={start}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"

            async with httpx.AsyncClient(timeout=30.0) as http_client:
                response = await http_client.get(url)
                response.raise_for_status()

            feed = feedparser.parse(response.text)
            papers = parse_arxiv_response(feed, primary_category, None)
        else:
            # Get seen paper IDs for this user to avoid repetition
            seen_docs = await db.seen_papers.find({"user_id": user_id}).to_list(500)
            seen_ids = set(doc["paper_id"] for doc in seen_docs)

            # Filter out seen papers
            unseen_ids = [pid for pid in recent_ids if pid not in seen_ids]

            # If all papers seen, reset and show from beginning
            if not unseen_ids:
                await db.seen_papers.delete_many({"user_id": user_id})
                unseen_ids = recent_ids

            # Get papers for the current page
            page_ids = unseen_ids[start:start + max_results]

            if page_ids:
                papers = await get_papers_by_ids(page_ids, primary_category, announced_date)
            else:
                papers = []

        # Get liked papers set
        liked_docs = await db.liked_papers.find({"user_id": user_id}, {"paper_id": 1}).to_list(100)
        liked_ids = set(doc["paper_id"] for doc in liked_docs)

        return PapersResponse(
            papers=papers,
            total=len(recent_ids) if recent_ids else 0,
            has_more=start + len(papers) < len(recent_ids) if recent_ids else False,
            new_papers_count=len(papers)
        )
    except Exception as e:
        logging.error(f"Error fetching paper feed: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching papers: {str(e)}")

@api_router.post("/papers/mark-seen")
async def mark_papers_seen(request: Request):
    """Mark current date as last seen - clears new paper badge"""
    user = await require_auth(request)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    await db.users.update_one(
        {"user_id": user.user_id},
        {"$set": {"last_seen_paper_date": today}}
    )
    
    return {"message": "Papers marked as seen", "date": today}

@api_router.post("/papers/mark-viewed/{paper_id}")
async def mark_paper_viewed(request: Request, paper_id: str):
    """Mark a paper as viewed to avoid showing it again"""
    user = await get_current_user(request)
    user_id = user.user_id if user else "anonymous"
    
    # Check if already marked
    existing = await db.seen_papers.find_one({"user_id": user_id, "paper_id": paper_id})
    if not existing:
        await db.seen_papers.insert_one({
            "user_id": user_id,
            "paper_id": paper_id,
            "viewed_at": datetime.now(timezone.utc).isoformat()
        })
    
    return {"message": "Paper marked as viewed"}

# Likes Routes
@api_router.post("/likes")
async def like_paper(request: Request, like_request: LikeRequest):
    """Like a paper"""
    user = await get_current_user(request)
    user_id = user.user_id if user else "anonymous"
    
    existing = await db.liked_papers.find_one({
        "user_id": user_id,
        "paper_id": like_request.paper_id
    })
    if existing:
        return {"message": "Already liked"}
    
    liked_paper = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "paper_id": like_request.paper_id,
        "title": like_request.title,
        "abstract": like_request.abstract,
        "authors": like_request.authors,
        "published": like_request.published,
        "link": like_request.link,
        "category": like_request.category,
        "liked_at": datetime.now(timezone.utc).isoformat()
    }
    
    await db.liked_papers.insert_one(liked_paper)
    # Return without _id field
    result = {k: v for k, v in liked_paper.items() if k != "_id"}
    return result

@api_router.delete("/likes/{paper_id}")
async def unlike_paper(request: Request, paper_id: str):
    """Unlike a paper"""
    user = await get_current_user(request)
    user_id = user.user_id if user else "anonymous"
    
    result = await db.liked_papers.delete_one({
        "user_id": user_id,
        "paper_id": paper_id
    })
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Paper not found in likes")
    return {"message": "Paper unliked"}

@api_router.get("/likes")
async def get_liked_papers(request: Request):
    """Get all liked papers for current user"""
    user = await get_current_user(request)
    user_id = user.user_id if user else "anonymous"
    
    liked_papers = await db.liked_papers.find(
        {"user_id": user_id},
        {"_id": 0}
    ).sort("liked_at", -1).to_list(100)
    
    return liked_papers

@api_router.get("/likes/check/{paper_id}")
async def check_if_liked(request: Request, paper_id: str):
    """Check if a paper is liked"""
    user = await get_current_user(request)
    user_id = user.user_id if user else "anonymous"
    
    existing = await db.liked_papers.find_one({
        "user_id": user_id,
        "paper_id": paper_id
    })
    return {"is_liked": existing is not None}

@api_router.get("/new-papers-count")
async def get_new_papers_count(request: Request, category: str = "astro-ph"):
    """Get count of new papers since last seen"""
    user = await get_current_user(request)
    if not user or not user.last_seen_paper_date:
        return {"count": 0}
    
    base_url = "https://export.arxiv.org/api/query"
    query = f"cat:{category}"
    url = f"{base_url}?search_query={query}&start=0&max_results=50&sortBy=submittedDate&sortOrder=descending"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            response = await http_client.get(url)
            response.raise_for_status()
            
        feed = feedparser.parse(response.text)
        count = 0
        for entry in feed.entries:
            published = entry.published[:10]
            if published > user.last_seen_paper_date:
                count += 1
        
        return {"count": count}
    except Exception as e:
        return {"count": 0}

@api_router.get("/for-you", response_model=PapersResponse)
async def get_for_you_papers(
    request: Request,
    category: str = Query(default="astro-ph"),
    year_from: int = Query(default=2020, ge=1990),
    year_to: int = Query(default=2026, le=2030),
    max_results: int = Query(default=30, ge=1, le=100)
):
    """Get personalized recommendations based on likes, using recent papers"""
    user = await get_current_user(request)
    user_id = user.user_id if user else "anonymous"
    
    # Get liked papers
    liked_papers = await db.liked_papers.find({"user_id": user_id}).to_list(100)
    
    try:
        # Get recent paper IDs from arXiv new submissions page (same as feed)
        recent_ids, announced_date = await get_recent_paper_ids(category, max_results=100)

        if recent_ids:
            # Get full paper details
            all_papers = await get_papers_by_ids(recent_ids, category)
        else:
            # Fallback to API if scraping fails
            base_url = "https://export.arxiv.org/api/query"
            query = f"cat:{category}"
            url = f"{base_url}?search_query={query}&start=0&max_results={max_results * 3}&sortBy=submittedDate&sortOrder=descending"
            
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                response = await http_client.get(url)
                response.raise_for_status()
                
            feed = feedparser.parse(response.text)
            all_papers = parse_arxiv_response(feed, category, None)
        
        # Filter by year range
        filtered_papers = []
        for paper in all_papers:
            try:
                paper_year = int(paper.published[:4])
                if year_from <= paper_year <= year_to:
                    filtered_papers.append(paper)
            except:
                continue
        
        # Get seen paper IDs to avoid repetition
        seen_docs = await db.seen_papers.find({"user_id": user_id}).to_list(500)
        seen_ids = set(doc["paper_id"] for doc in seen_docs)
        
        # Filter out seen papers
        unseen_papers = [p for p in filtered_papers if p.id not in seen_ids]
        
        # If all papers seen, use all papers
        if not unseen_papers:
            unseen_papers = filtered_papers
        
        # If user has likes, get recommendations
        if liked_papers and unseen_papers:
            recommendations = await get_recommendations(liked_papers, unseen_papers, top_n=max_results)
            # Mark recommendations
            for paper in recommendations:
                paper.is_recommendation = True
            
            if recommendations:
                return PapersResponse(
                    papers=recommendations[:max_results],
                    total=len(recommendations),
                    has_more=False,
                    new_papers_count=len([p for p in recommendations if p.is_new])
                )
        
        # No likes or no recommendations found, return recent papers
        for paper in unseen_papers:
            paper.is_recommendation = False
        
        return PapersResponse(
            papers=unseen_papers[:max_results],
            total=len(unseen_papers),
            has_more=len(unseen_papers) > max_results,
            new_papers_count=len([p for p in unseen_papers if p.is_new])
        )
    except Exception as e:
        logging.error(f"Error fetching for-you papers: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching papers: {str(e)}")

# Include router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
