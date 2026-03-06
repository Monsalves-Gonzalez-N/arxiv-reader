#!/usr/bin/env python3
"""
Backend API Testing for arXiv TikTok-style App
Tests all backend endpoints comprehensively
"""

import asyncio
import httpx
import json
from datetime import datetime
import sys
import os

# Backend URL from frontend environment
BACKEND_URL = "https://astro-reader-4.preview.emergentagent.com"
API_BASE = f"{BACKEND_URL}/api"

class BackendTester:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.test_results = []
        self.test_paper_id = None
        
    async def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test results"""
        status = "✅ PASS" if success else "❌ FAIL"
        message = f"{status} {test_name}"
        if details:
            message += f" - {details}"
        print(message)
        self.test_results.append({
            "test": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
        
    async def test_get_categories(self):
        """Test GET /api/categories"""
        try:
            response = await self.client.get(f"{API_BASE}/categories")
            if response.status_code == 200:
                data = response.json()
                if "categories" in data and isinstance(data["categories"], dict):
                    categories = data["categories"]
                    expected_cats = ["astro-ph", "astro-ph.GA", "astro-ph.CO"]
                    has_expected = all(cat in categories for cat in expected_cats)
                    await self.log_test("GET /api/categories", has_expected, 
                                      f"Found {len(categories)} categories")
                    return has_expected
                else:
                    await self.log_test("GET /api/categories", False, "Invalid response format")
                    return False
            else:
                await self.log_test("GET /api/categories", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            await self.log_test("GET /api/categories", False, f"Error: {str(e)}")
            return False
            
    async def test_get_papers(self):
        """Test GET /api/papers"""
        try:
            response = await self.client.get(f"{API_BASE}/papers?category=astro-ph&max_results=5")
            if response.status_code == 200:
                data = response.json()
                if "papers" in data and isinstance(data["papers"], list):
                    papers = data["papers"]
                    if len(papers) > 0:
                        # Store first paper for later tests
                        self.test_paper_id = papers[0]["id"]
                        await self.log_test("GET /api/papers", True, 
                                          f"Retrieved {len(papers)} papers")
                        return True, papers[0]  # Return first paper for testing
                    else:
                        await self.log_test("GET /api/papers", False, "No papers returned")
                        return False, None
                else:
                    await self.log_test("GET /api/papers", False, "Invalid response format")
                    return False, None
            else:
                await self.log_test("GET /api/papers", False, f"Status: {response.status_code}")
                return False, None
        except Exception as e:
            await self.log_test("GET /api/papers", False, f"Error: {str(e)}")
            return False, None
            
    async def test_get_paper_feed(self):
        """Test GET /api/papers/feed"""
        try:
            response = await self.client.get(f"{API_BASE}/papers/feed?category=astro-ph&start=0&max_results=5")
            if response.status_code == 200:
                data = response.json()
                if "papers" in data and isinstance(data["papers"], list):
                    papers = data["papers"]
                    await self.log_test("GET /api/papers/feed", True, 
                                      f"Retrieved {len(papers)} papers in feed")
                    return True
                else:
                    await self.log_test("GET /api/papers/feed", False, "Invalid response format")
                    return False
            else:
                await self.log_test("GET /api/papers/feed", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            await self.log_test("GET /api/papers/feed", False, f"Error: {str(e)}")
            return False
            
    async def test_post_like(self, paper_data):
        """Test POST /api/likes"""
        if not paper_data:
            await self.log_test("POST /api/likes", False, "No paper data available")
            return False
            
        try:
            like_request = {
                "paper_id": paper_data["id"],
                "title": paper_data["title"],
                "abstract": paper_data["abstract"],
                "authors": paper_data["authors"],
                "published": paper_data["published"],
                "link": paper_data["link"],
                "category": paper_data["category"]
            }
            
            response = await self.client.post(f"{API_BASE}/likes", json=like_request)
            if response.status_code == 200:
                data = response.json()
                if "id" in data and "paper_id" in data:
                    await self.log_test("POST /api/likes", True, 
                                      f"Liked paper: {paper_data['id']}")
                    return True
                else:
                    await self.log_test("POST /api/likes", False, "Invalid response format")
                    return False
            else:
                await self.log_test("POST /api/likes", False, 
                                  f"Status: {response.status_code}, Response: {response.text}")
                return False
        except Exception as e:
            await self.log_test("POST /api/likes", False, f"Error: {str(e)}")
            return False
            
    async def test_get_likes(self):
        """Test GET /api/likes"""
        try:
            response = await self.client.get(f"{API_BASE}/likes")
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    await self.log_test("GET /api/likes", True, 
                                      f"Retrieved {len(data)} liked papers")
                    return True, len(data)
                else:
                    await self.log_test("GET /api/likes", False, "Invalid response format")
                    return False, 0
            else:
                await self.log_test("GET /api/likes", False, f"Status: {response.status_code}")
                return False, 0
        except Exception as e:
            await self.log_test("GET /api/likes", False, f"Error: {str(e)}")
            return False, 0
            
    async def test_check_like(self, paper_id):
        """Test GET /api/likes/check/{paper_id}"""
        if not paper_id:
            await self.log_test("GET /api/likes/check/{paper_id}", False, "No paper ID available")
            return False
            
        try:
            response = await self.client.get(f"{API_BASE}/likes/check/{paper_id}")
            if response.status_code == 200:
                data = response.json()
                if "is_liked" in data:
                    is_liked = data["is_liked"]
                    await self.log_test("GET /api/likes/check/{paper_id}", True, 
                                      f"Paper {paper_id} liked status: {is_liked}")
                    return True, is_liked
                else:
                    await self.log_test("GET /api/likes/check/{paper_id}", False, "Invalid response format")
                    return False, False
            else:
                await self.log_test("GET /api/likes/check/{paper_id}", False, f"Status: {response.status_code}")
                return False, False
        except Exception as e:
            await self.log_test("GET /api/likes/check/{paper_id}", False, f"Error: {str(e)}")
            return False, False
            
    async def test_delete_like(self, paper_id):
        """Test DELETE /api/likes/{paper_id}"""
        if not paper_id:
            await self.log_test("DELETE /api/likes/{paper_id}", False, "No paper ID available")
            return False
            
        try:
            response = await self.client.delete(f"{API_BASE}/likes/{paper_id}")
            if response.status_code == 200:
                data = response.json()
                if "message" in data:
                    await self.log_test("DELETE /api/likes/{paper_id}", True, 
                                      f"Unliked paper: {paper_id}")
                    return True
                else:
                    await self.log_test("DELETE /api/likes/{paper_id}", False, "Invalid response format")
                    return False
            else:
                await self.log_test("DELETE /api/likes/{paper_id}", False, 
                                  f"Status: {response.status_code}, Response: {response.text}")
                return False
        except Exception as e:
            await self.log_test("DELETE /api/likes/{paper_id}", False, f"Error: {str(e)}")
            return False
            
    async def test_get_recommendations(self):
        """Test GET /api/recommendations"""
        try:
            response = await self.client.get(f"{API_BASE}/recommendations?category=astro-ph&max_results=5")
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    await self.log_test("GET /api/recommendations", True, 
                                      f"Retrieved {len(data)} recommendations")
                    return True
                else:
                    await self.log_test("GET /api/recommendations", False, "Invalid response format")
                    return False
            else:
                await self.log_test("GET /api/recommendations", False, f"Status: {response.status_code}")
                return False
        except Exception as e:
            await self.log_test("GET /api/recommendations", False, f"Error: {str(e)}")
            return False
            
    async def run_comprehensive_test(self):
        """Run all tests in sequence"""
        print(f"🚀 Starting Backend API Tests for {API_BASE}")
        print("=" * 60)
        
        # Test 1: Get categories
        await self.test_get_categories()
        
        # Test 2: Get papers (and store one for testing)
        papers_success, test_paper = await self.test_get_papers()
        
        # Test 3: Get paper feed
        await self.test_get_paper_feed()
        
        # Test 4: Like a paper (if we have one)
        like_success = False
        if test_paper:
            like_success = await self.test_post_like(test_paper)
        
        # Test 5: Get liked papers
        likes_success, likes_count = await self.test_get_likes()
        
        # Test 6: Check if paper is liked
        if self.test_paper_id:
            await self.test_check_like(self.test_paper_id)
        
        # Test 7: Get recommendations (should work if we have likes)
        await self.test_get_recommendations()
        
        # Test 8: Unlike the paper
        if self.test_paper_id and like_success:
            unlike_success = await self.test_delete_like(self.test_paper_id)
            
            # Test 9: Verify paper is no longer liked
            if unlike_success:
                await self.test_check_like(self.test_paper_id)
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results if result["success"])
        total = len(self.test_results)
        
        for result in self.test_results:
            status = "✅" if result["success"] else "❌"
            print(f"{status} {result['test']}")
            if result["details"]:
                print(f"   └─ {result['details']}")
        
        print(f"\n🎯 Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed!")
        else:
            print("⚠️  Some tests failed - check details above")
            
        return passed == total
        
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

async def main():
    """Main test runner"""
    tester = BackendTester()
    try:
        success = await tester.run_comprehensive_test()
        return 0 if success else 1
    finally:
        await tester.close()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)