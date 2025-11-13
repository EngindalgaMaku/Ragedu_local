# src/recommendation/recommender.py

from typing import List, Dict, Any
from duckduckgo_search import DDGS

def generate_recommendations(query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Analyzes the user's query and retrieved document chunks to generate recommendations.

    Args:
        query (str): The user's query.
        retrieved_chunks (List[Dict[str, Any]]): A list of document chunks retrieved from the vector store.

    Returns:
        Dict[str, List[str]]: A dictionary of recommended resources, categorized by type.
    """
    recommendations = {
        "internal_links": [],
        "external_links": [],
        "practice_problems": []
    }

    # 1. Identify key topics from the query.
    # A simple approach is to use the query itself as the topic.
    topic = query

    # 2. Find other relevant sections in the existing materials.
    # This is a placeholder. A real implementation could involve finding chunks with high 
    # similarity to the query but not included in the top results.
    if retrieved_chunks:
        # Suggesting the source of the first chunk as an internal link for demonstration.
        source = retrieved_chunks.get('metadata', {}).get('source', 'N/A')
        if source != 'N/A':
            recommendations["internal_links"].append(f"For more context, you can review the document: {source}")

    # 3. Generate external web links using DuckDuckGo Search.
    try:
        with DDGS() as ddgs:
            search_results = list(ddgs.text(f"tutorial or documentation on {topic}", max_results=2))
            if search_results:
                for result in search_results:
                    recommendations["external_links"].append(f"{result['title']}: {result['href']}")
    except Exception as e:
        print(f"Error during DuckDuckGo search: {e}")

    # 4. Suggest practice problems based on keywords.
    if "algorithm" in query.lower() or "data structure" in query.lower():
        recommendations["practice_problems"].append("Try solving related problems on platforms like LeetCode or HackerRank.")
    if "python" in query.lower():
        recommendations["practice_problems"].append("Check out Python exercises on sites like Exercism or Codewars.")
    if "machine learning" in query.lower():
        recommendations["practice_problems"].append("Explore datasets on Kaggle and try to build a predictive model.")

    return recommendations