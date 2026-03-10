from fastapi import FastAPI, APIRouter, HTTPException, Query, Depends, Response, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
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
import jwt
from jwt import PyJWKClient

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
_jwks_client = PyJWKClient(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json") if SUPABASE_URL else None

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
use_tls = 'localhost' not in mongo_url and '127.0.0.1' not in mongo_url
if use_tls:
    client = AsyncIOMotorClient(mongo_url, tls=True, tlsAllowInvalidCertificates=True)
else:
    client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'arxiv_tiktok')]

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create the main app
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


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

def get_device_id(request: Request) -> str:
    """Extract user ID from Supabase JWT (JWKS/RS256), fallback to device ID, fallback to anonymous"""
    user_id, _ = _resolve_user(request)
    return user_id

def _resolve_user(request: Request):
    """Returns (user_id, error_detail) for diagnostics"""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        if not _jwks_client:
            msg = "SUPABASE_URL is not set — cannot verify JWT"
            logging.error(msg)
            return "anonymous", msg
        token = auth_header[7:]
        try:
            signing_key = _jwks_client.get_signing_key_from_jwt(token)
            payload = jwt.decode(token, signing_key.key, algorithms=[signing_key.algorithm_name], audience="authenticated")
            user_id = payload.get("sub")
            if user_id:
                return user_id, None
            msg = "JWT decoded but 'sub' claim is missing"
            logging.warning(msg)
            return "anonymous", msg
        except Exception as e:
            msg = f"JWT decode failed: {e}"
            logging.error(msg)
            return "anonymous", msg
    return request.headers.get("X-Device-ID") or "anonymous", "no Bearer token"

# Helper functions
async def get_recent_paper_ids(category: str):
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

        return paper_ids, announced_date
    except Exception as e:
        logging.error(f"Error fetching recent paper IDs: {e}")
        return [], None

async def get_papers_by_ids(paper_ids: List[str], category: str, announced_date: str = None) -> List[Paper]:
    """Fetch paper details by IDs from arXiv API, batching in groups of 50."""
    if not paper_ids:
        return []

    import asyncio

    BATCH_SIZE = 50

    async def fetch_batch(batch: List[str]) -> List[Paper]:
        id_list = ','.join(batch)
        url = f"https://export.arxiv.org/api/query?id_list={id_list}&max_results={len(batch)}"
        try:
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                response = await http_client.get(url)
                response.raise_for_status()
            feed = feedparser.parse(response.text)
            result = []
            for entry in feed.entries:
                paper_id = entry.id.split('/abs/')[-1]
                comment = None
                if hasattr(entry, 'arxiv_comment'):
                    comment = entry.arxiv_comment
                elif 'arxiv_comment' in entry:
                    comment = entry['arxiv_comment']
                result.append(Paper(
                    id=paper_id,
                    title=' '.join(entry.title.split()),
                    abstract=' '.join(entry.summary.split()),
                    authors=[author.name for author in entry.authors],
                    published=announced_date if announced_date else entry.published[:10],
                    link=entry.link,
                    category=category,
                    comment=comment,
                    is_new=True,
                ))
            return result
        except Exception as e:
            logging.error(f"Error fetching batch: {e}")
            return []

    batches = [paper_ids[i:i + BATCH_SIZE] for i in range(0, len(paper_ids), BATCH_SIZE)]
    results = await asyncio.gather(*[fetch_batch(b) for b in batches])
    return [paper for batch in results for paper in batch]

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

async def blend_feed_papers(papers: List[Paper], liked_papers: List[dict]) -> List[Paper]:
    """Re-rank feed papers using 80% recency + 20% TF-IDF similarity to liked papers.
    Falls back to original order if insufficient data."""
    if not papers or not liked_papers or len(liked_papers) < 2:
        return papers

    try:
        liked_texts = [f"{p['title']} {p['abstract']}" for p in liked_papers]
        candidate_texts = [f"{p.title} {p.abstract}" for p in papers]

        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(liked_texts + candidate_texts)

        liked_vectors = tfidf_matrix[:len(liked_texts)]
        candidate_vectors = tfidf_matrix[len(liked_texts):]

        similarity_scores = cosine_similarity(candidate_vectors, liked_vectors).mean(axis=1)

        today = datetime.now(timezone.utc)
        recency_scores = []
        for paper in papers:
            try:
                pub_date = datetime.strptime(paper.published, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                days_old = (today - pub_date).days
                recency_scores.append(max(0.0, 1.0 - (days_old / 365)))
            except Exception:
                recency_scores.append(0.5)

        recency_scores = np.array(recency_scores)
        combined_scores = 0.8 * recency_scores + 0.2 * similarity_scores

        sorted_indices = np.argsort(combined_scores)[::-1]
        return [papers[i] for i in sorted_indices]
    except Exception as e:
        logging.warning(f"blend_feed_papers failed, returning original order: {e}")
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


# Paper Routes
@api_router.get("/")
async def root():
    return {"message": "ArXiv TikTok API"}

@api_router.get("/categories")
async def get_categories():
    return {"categories": ASTRO_CATEGORIES}

@api_router.get("/papers", response_model=PapersResponse)
@limiter.limit("20/minute")
async def get_papers(
    request: Request,
    category: str = Query(default="astro-ph"),
    start: int = Query(default=0, ge=0),
    max_results: int = Query(default=25, ge=1, le=100),
    year: Optional[int] = Query(default=None),
    month: Optional[int] = Query(default=None),
    date_from: Optional[str] = Query(default=None),  # YYYYMMDD
    date_to: Optional[str] = Query(default=None),    # YYYYMMDD
):
    """Fetch papers from arXiv API - newest first, optionally filtered by year/month"""
    import calendar as cal

    base_url = "https://export.arxiv.org/api/query"

    # Build category filter — supports comma-separated list
    cats = [c.strip() for c in category.split(',') if c.strip()]
    if len(cats) > 1:
        cat_filter = '(' + '+OR+'.join(f'cat:{c}' for c in cats) + ')'
    else:
        cat_filter = f'cat:{cats[0]}' if cats else 'cat:astro-ph'

    if date_from and date_to:
        query = f"{cat_filter}+AND+submittedDate:[{date_from}0000+TO+{date_to}2359]"
    elif year and month:
        last_day = cal.monthrange(year, month)[1]
        df = f"{year}{month:02d}010000"
        dt = f"{year}{month:02d}{last_day:02d}2359"
        query = f"{cat_filter}+AND+submittedDate:[{df}+TO+{dt}]"
    elif year:
        query = f"{cat_filter}+AND+submittedDate:[{year}01010000+TO+{year}12312359]"
    else:
        query = cat_filter

    url = f"{base_url}?search_query={query}&start={start}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"

    try:
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            response = await http_client.get(url)
            response.raise_for_status()

        feed = feedparser.parse(response.text)
        papers = parse_arxiv_response(feed, category)
        total = int(feed.feed.get('opensearch_totalresults', 0))

        return PapersResponse(
            papers=papers,
            total=total,
            has_more=start + len(papers) < total,
            new_papers_count=0
        )
    except Exception as e:
        logging.error(f"Error fetching papers: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching papers: {str(e)}")

@api_router.get("/papers/feed", response_model=PapersResponse)
@limiter.limit("30/minute")
async def get_paper_feed(
    request: Request,
    category: str = Query(default="astro-ph"),
    start: int = Query(default=0, ge=0),
    max_results: int = Query(default=20, ge=1, le=50)
):
    """Get paper feed - newest papers first from arXiv new submissions"""
    user_id = get_device_id(request)

    # Support comma-separated categories
    categories = [c.strip() for c in category.split(",") if c.strip()]
    if not categories:
        categories = ["astro-ph"]

    try:
        import asyncio
        if len(categories) == 1:
            recent_ids, announced_date = await get_recent_paper_ids(categories[0])
            primary_category = categories[0]
        else:
            # Fetch IDs for each category in parallel and merge
            results = await asyncio.gather(*[get_recent_paper_ids(c) for c in categories])
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

            # Return all unseen today's papers — the count is finite and known
            if unseen_ids:
                papers = await get_papers_by_ids(unseen_ids, primary_category, announced_date)
            else:
                papers = []

        # Blend feed using liked papers when available (80% recency + 20% similarity)
        liked_docs = await db.liked_papers.find({"user_id": user_id}).to_list(100)
        if liked_docs and len(liked_docs) >= 2:
            papers = await blend_feed_papers(papers, liked_docs)

        return PapersResponse(
            papers=papers,
            total=len(papers),
            has_more=False,
            new_papers_count=len(papers)
        )
    except Exception as e:
        logging.error(f"Error fetching paper feed: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching papers: {str(e)}")

@api_router.post("/papers/mark-seen")
async def mark_papers_seen(request: Request):
    return {"message": "ok"}

@api_router.post("/papers/mark-viewed/{paper_id}")
@limiter.limit("60/minute")
async def mark_paper_viewed(request: Request, paper_id: str):
    """Mark a paper as viewed to avoid showing it again"""
    user_id = get_device_id(request)
    
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
@limiter.limit("30/minute")
async def like_paper(request: Request, like_request: LikeRequest):
    """Like a paper"""
    user_id = get_device_id(request)
    
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
@limiter.limit("30/minute")
async def unlike_paper(request: Request, paper_id: str):
    """Unlike a paper"""
    user_id = get_device_id(request)
    
    result = await db.liked_papers.delete_one({
        "user_id": user_id,
        "paper_id": paper_id
    })
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Paper not found in likes")
    return {"message": "Paper unliked"}

@api_router.get("/likes")
@limiter.limit("30/minute")
async def get_liked_papers(request: Request):
    """Get all liked papers for current user"""
    user_id = get_device_id(request)
    
    liked_papers = await db.liked_papers.find(
        {"user_id": user_id},
        {"_id": 0}
    ).sort("liked_at", -1).to_list(100)
    
    return liked_papers

@api_router.get("/likes/check/{paper_id}")
async def check_if_liked(request: Request, paper_id: str):
    """Check if a paper is liked"""
    user_id = get_device_id(request)
    
    existing = await db.liked_papers.find_one({
        "user_id": user_id,
        "paper_id": paper_id
    })
    return {"is_liked": existing is not None}

@api_router.get("/new-papers-count")
async def get_new_papers_count(request: Request, category: str = "astro-ph"):
    """Get count of new papers since last seen"""
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
@limiter.limit("10/minute")
async def get_for_you_papers(
    request: Request,
    category: str = Query(default="astro-ph"),
    year_from: int = Query(default=2020, ge=1990),
    year_to: int = Query(default=2026, le=2030),
    max_results: int = Query(default=30, ge=1, le=100)
):
    """Get personalized recommendations based on likes, using recent papers"""
    user_id = get_device_id(request)
    
    # Get liked papers
    liked_papers = await db.liked_papers.find({"user_id": user_id}).to_list(100)
    
    try:
        # Get recent paper IDs from arXiv new submissions page (same as feed)
        recent_ids, announced_date = await get_recent_paper_ids(category)

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

@api_router.get("/me")
async def get_me(request: Request):
    """Returns the resolved user_id and their like count — for verifying auth is working correctly"""
    user_id, auth_error = _resolve_user(request)
    like_count = await db.liked_papers.count_documents({"user_id": user_id})
    return {"user_id": user_id, "like_count": like_count, "auth_error": auth_error}

# Include router
app.include_router(api_router)

ALLOWED_ORIGINS = [
    "https://arxiv-reader-psi.vercel.app",
    "http://localhost:8081",
    "http://localhost:19006",
]

app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=ALLOWED_ORIGINS,
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
