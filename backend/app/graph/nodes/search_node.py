"""
Web Search Node - Fallback Knowledge Retrieval

Performs web search when RAG confidence is insufficient.
Ensures current information availability.
"""

import time
from typing import Dict

from langchain_community.tools.tavily_search import TavilySearchResults

from backend.app.graph.state import VetAgentState
from backend.app.config import settings


class SearchNode:
    """
    Web search fallback for low RAG confidence.

    Uses Tavily for AI-optimized search results.
    """

    def __init__(self):
        self.search_tool = TavilySearchResults(
            max_results=settings.TAVILY_MAX_RESULTS,
            search_depth=settings.TAVILY_SEARCH_DEPTH,
            api_key=settings.TAVILY_API_KEY
        )

    async def __call__(self, state: VetAgentState) -> Dict:
        """
        Perform web search if RAG insufficient.

        Returns updated state with:
        - web_search_results
        - web_search_performed
        """

        # Check if web search needed
        if not state.get("requires_web_search", False):
            print(f"[Search] Skipping (RAG confidence sufficient)")
            return {
                "web_search_performed": False,
                "graph_path": state.get("graph_path", []) + ["search_skipped"]
            }

        start_time = time.time()

        query = state["query"]
        intent = state.get("query_intent", "general")
        species = state.get("species", "")

        # Optimize query for veterinary search
        search_query = self._optimize_query(query, intent, species)

        print(f"\n[Search] Performing web search...")
        print(f"  Query: {search_query}")

        try:
            # Perform search
            results = await self.search_tool.ainvoke(search_query)

            latency_ms = int((time.time() - start_time) * 1000)

            print(f"  Found: {len(results)} results")
            print(f"[Search] Complete ({latency_ms}ms)")

            return {
                "web_search_query": search_query,
                "web_search_results": results,
                "web_search_performed": True,
                "graph_path": state.get("graph_path", []) + ["search"]
            }

        except Exception as e:
            print(f"[Search] Error: {e}")

            return {
                "web_search_query": search_query,
                "web_search_results": [],
                "web_search_performed": False,
                "errors": state.get("errors", []) + [f"Search error: {str(e)}"],
                "graph_path": state.get("graph_path", []) + ["search"]
            }

    def _optimize_query(self, query: str, intent: str, species: str = "") -> str:
        """
        Optimize query for veterinary web search.

        Args:
            query: Original query
            intent: Query intent
            species: Patient species

        Returns:
            Optimized search query
        """

        # Add veterinary context
        search_query = f"{query} veterinary medicine"

        # Add species if available
        if species:
            search_query = f"{search_query} {species}"

        # Add intent-specific modifiers
        if intent == "treatment":
            search_query = f"{search_query} treatment protocol FDA approved"
        elif intent == "diagnosis":
            search_query = f"{search_query} differential diagnosis"
        elif intent == "triage":
            search_query = f"{search_query} emergency assessment"

        return search_query


# Singleton instance
search_node = SearchNode()