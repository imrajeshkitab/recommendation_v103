from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def _execute_hybrid_search(
    supabase_client: Any,
    function_name: str,
    query_embedding: Optional[List[float]],
    query_text: str,
    match_count: int = 10,
) -> List[Dict[str, Any]]:
    """Execute a Supabase RPC for hybrid search and return the results as a list."""
    if query_embedding is None:
        logger.warning("Hybrid search skipped: missing query embedding for %s", function_name)
        return []

    try:
        logger.info("Calling RPC '%s' with count=%d", function_name, match_count)
        response = (
            supabase_client.rpc(
                function_name,
                {
                    "query_text": query_text,
                    "query_embedding": query_embedding,
                    "match_count": match_count,
                },
            ).execute()
        )

        data = response.data if getattr(response, "data", None) else []
        count = len(data) if isinstance(data, list) else (1 if data else 0)

        if count == 0:
            logger.warning("RPC '%s' returned no results", function_name)
        else:
            logger.info("RPC '%s' returned %d results", function_name, count)

        return data if data else []
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Error executing RPC '%s': %s", function_name, exc)
        return []


def search_summaries(
    supabase_client: Any,
    query_embedding: Optional[List[float]],
    query_text: str,
    match_count: int = 10,
) -> List[Dict[str, Any]]:
    """Search summaries using the hybrid search RPC."""
    return _execute_hybrid_search(
        supabase_client,
        "hybrid_search_summaries",
        query_embedding,
        query_text,
        match_count,
    )


def search_bytes(
    supabase_client: Any,
    query_embedding: Optional[List[float]],
    query_text: str,
    match_count: int = 10,
) -> List[Dict[str, Any]]:
    """Search bytes using the hybrid search RPC."""
    return _execute_hybrid_search(
        supabase_client,
        "hybrid_search_bytes",
        query_embedding,
        query_text,
        match_count,
    )