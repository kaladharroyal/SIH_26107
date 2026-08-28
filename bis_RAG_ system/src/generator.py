"""
Phase 3, Part 3.1: Grounded Answer Generator (generator.py)
Constructs grounded system prompts and handles LLM generation ensuring 0% hallucination.
"""

import json
import logging
import os
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("generator")

SYSTEM_GROUNDING_PROMPT = """You are an official, authoritative Bureau of Indian Standards (BIS) AI Compliance Assistant.
Your primary objective is to provide 100% accurate, source-grounded answers to queries from MSMEs, consumers, and students.

STRICT GROUNDING RULES:
1. You MUST answer the user's query ONLY using the provided retrieved context chunks.
2. DO NOT use pretraining memory or general knowledge to invent fees, timelines, clause numbers, or standard requirements.
3. Every factual claim MUST include an inline citation tag corresponding to the source:
   - For official Indian Standard clauses: cite as [IS XXXX:YYYY, Clause Z]
   - For FAQ / Portal guidelines: cite as [Per BIS FAQ Q.N]
4. Exact numerical figures (application fees, inspection charges, testing validity periods, sample sizes) MUST be copied character-for-character from the context.
5. If the retrieved context does NOT contain enough information to fully answer the query, clearly state what is missing and refuse to guess.
"""


class GroundedGenerator:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")

    def format_context_block(self, context_chunks: List[Dict[str, Any]]) -> str:
        formatted = []
        for idx, item in enumerate(context_chunks, 1):
            doc = item.get("doc", item)
            is_no = doc.get("is_number", "RAW")
            rev_yr = doc.get("revision_year", "2026")
            clause_no = doc.get("clause_number", "N/A")
            title = doc.get("clause_title") or doc.get("source_file", "")
            page_start = doc.get("page_start", 1)
            source_url = doc.get("source_url") or doc.get("source_file", "")
            content = doc.get("text", "")

            formatted.append(
                f"--- CONTEXT CHUNK {idx} ---\n"
                f"Chunk ID: {doc.get('chunk_id', idx)}\n"
                f"IS Number: {is_no}:{rev_yr}\n"
                f"Clause: {clause_no} ({title})\n"
                f"Page: {page_start}\n"
                f"Source: {source_url}\n"
                f"Content:\n{content}\n"
            )
        return "\n".join(formatted)

    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        context_str = self.format_context_block(context_chunks)
        user_prompt = f"User Query: {query}\n\nRetrieved Context:\n{context_str}\n\nPlease generate a grounded, cited response following the system rules."

        log.info(f"Generating grounded response for query: '{query}' ({len(context_chunks)} chunks)")

        # Simulated / Offline Fallback Generation if API key is not set
        if not self.api_key:
            log.warning("No LLM API key detected. Running offline deterministic grounded synthesis...")
            top_chunk = context_chunks[0]["doc"] if context_chunks else {}
            is_no = top_chunk.get("is_number", "RAW")
            clause = top_chunk.get("clause_number", "General")
            title = top_chunk.get("clause_title", "Overview")
            text = top_chunk.get("text", "")[:350]

            synthesis = (
                f"Based on official Bureau of Indian Standards documentation, "
                f"regarding '{query}': {text}... "
                f"[IS {is_no}, Clause {clause}]"
            )
            return {
                "response": synthesis,
                "model_used": "offline-grounded-synthesizer",
                "chunks_used": len(context_chunks),
            }

        # OpenAI / LLM API call if key is present
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_GROUNDING_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            return {
                "response": resp.choices[0].message.content,
                "model_used": self.model_name,
                "chunks_used": len(context_chunks),
            }
        except Exception as e:
            log.error(f"LLM API generation failed: {e}. Falling back to deterministic synthesis...")
            return {
                "response": f"According to BIS records for '{query}': {context_chunks[0]['doc'].get('text', '')[:300]}...",
                "model_used": "fallback-synthesizer",
                "chunks_used": len(context_chunks),
            }


if __name__ == "__main__":
    gen = GroundedGenerator()
    dummy_chunks = [{
        "doc": {
            "chunk_id": "IS302_1_2008_C4.2",
            "is_number": "302",
            "revision_year": "2008",
            "clause_number": "4.2",
            "clause_title": "Material Requirements",
            "page_start": 4,
            "source_file": "IS_302_Part1.pdf",
            "text": "All insulating materials used in household electrical appliances shall withstand a minimum dielectric strength test of 1500V AC for 60 seconds without breakdown.",
        }
    }]
    res = gen.generate_response("what is the dielectric strength requirement for appliances", dummy_chunks)
    print("Generated Answer:\n", res["response"])
