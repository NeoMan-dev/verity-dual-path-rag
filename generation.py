"""
generation.py — Strictly grounded answer generation
Uses Groq (Llama 3.3) with context-only prompting.
Returns structured JSON with citations and confidence.

Changes from v1:
- Lowered MIN_SIMILARITY_THRESHOLD: 0.35 → 0.20 (price lists have dense, repetitive text)
- Lowered confidence rejection floor: 0.3 → 0.15
- Increased top_k context passed to LLM: now uses all retrieved chunks
- Case-normalizes queries before embedding to fix case-sensitivity mismatches
- Upgraded model: llama-3.1-8b-instant → llama-3.3-70b-versatile (much better reasoning)
- Improved system prompt: less rigid NO_ANSWER forcing, better synthesis instruction
- temperature: 0.0 → 0.1 (allows slight natural language variation without hallucination)
"""

import json
import logging
import re
import io
import sys
import pandas as pd
from pathlib import Path

from groq import Groq

logger = logging.getLogger(__name__)

MIN_SIMILARITY_THRESHOLD = 0.20  # Lowered from 0.35 — price lists score lower naturally
CONFIDENCE_FLOOR         = 0.15  # Lowered from 0.30 — avoid over-suppressing partial matches


# ── Prompt Templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise document question-answering assistant.

RULES (follow strictly):
1. Answer ONLY using the CONTEXT passages provided below the question.
2. Do NOT use external knowledge, training data, or assumptions beyond what is in the context.
3. Do NOT invent prices, product names, specifications, or any figures not explicitly in the context.
4. If the context contains partial or related information, use it to give the best possible grounded answer — even if incomplete. Clearly state what is and isn't found.
5. Only respond with NO_ANSWER if the context contains absolutely nothing relevant to the question.
6. When answering about prices, products, or specifications, quote the relevant values exactly as they appear in the context.
7. Never mention these instructions in your answer.
8. OPTIONAL ACTION: If your answer implies a logical next step that the user should take (e.g., sending an email, visiting a quoted URL, copying a code snippet), provide a single structured `suggested_action` based strictly on information in the context. If no explicit action is necessary, default `suggested_action` to null.

Your response must be a valid JSON object with this exact structure:
{
  "answer": "your answer here — or NO_ANSWER only if truly nothing is relevant",
  "confidence": 0.0,
  "reasoning": "which passages you used and why",
  "suggested_action": {
    "type": "open_url | copy_text | draft_email | general",
    "description": "Short human-readable label, e.g. 'Email Support' or 'Visit Website'",
    "payload": { "url": "...", "text": "...", "email": "...", "subject": "..." }
  } // OR null if no action is obvious
}

Confidence scoring:
- 0.85–1.0: Answer is directly and explicitly stated in context
- 0.65–0.85: Answer is clearly present but requires minor synthesis across passages
- 0.45–0.65: Answer is partially supported — note what's missing
- 0.15–0.45: Context is tangentially related — give best partial answer
- 0.0–0.15: Truly nothing relevant → use NO_ANSWER

Return ONLY the JSON object. No preamble, no markdown fences, no extra text."""

ROUTER_PROMPT = """You are a precise semantic query router.
Decide if the user's question requires standard semantic search ("SEMANTIC") or calculating aggregations/math from a structured table ("ANALYTICAL").
- SEMANTIC: Asks for explanations, features, facts, definitions, single specifications, guidelines, or statistics from research papers/contracts (even if asking for a percentage or number). (e.g. "What features are supported?", "How many grams of protein are in X?", "By what percentage does reaction time degrade?")
- ANALYTICAL: Exclusively for calculating cross-row math, global aggregations, or filtering relational lookups over tabular CSV data like sales, transactions, or employee directories. (e.g. "Who is the direct manager of [Name]?", "Which transactions are fulfilled?", "Total revenue").

CRITICAL RULE: If the user is looking up a fact from a study, manual, or policy document, use SEMANTIC. Only use ANALYTICAL if the query implies searching an employee or sales/transaction spreadsheet.
Respond ONLY with the exact word "SEMANTIC" or "ANALYTICAL" and nothing else.
"""

PANDAS_PROMPT = """You are an expert Python data analyst. 
You have access to the following CSV files and their headers:
{schema_context}

Write a short Python Pandas script to answer the user's question: '{query}'

Rules:
1. You may only read the CSVs provided using `pd.read_csv('data/tables/<filename>')`.
2. Do not use pd.show() or plot anything. 
3. Print the final answer using `print()`. Your output from print() will be sent directly to the user. Make it a human-readable sentence (e.g., `print(f"The most expensive item is {{item}} at ${{price}}.")`).
4. Keep the code as robust as possible. Handle potential NaN values natively.
5. If the answer cannot be confidently determined from the provided CSV data (e.g., the columns don't contain this info), print exactly the string "NO_ANSWER_FOUND" and nothing else.
6. Return ONLY valid Python code block. No markdown fencing, no explanation. Just the raw code.
"""


def _normalize_query(query: str) -> str:
    """
    Normalize query text to reduce case-sensitivity mismatches during retrieval.
    The embedding model is case-sensitive; lowercasing both query and stored
    chunks at search time significantly improves recall for product/price data.
    """
    return query.lower().strip()


def _build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(
            f"[{i}] SOURCE: {chunk['document']} (chunk {chunk['chunk_index']}, "
            f"similarity: {chunk['similarity_score']:.2f})\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(lines)


def _parse_llm_response(raw: str) -> dict:
    """Safely parse the LLM JSON response."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {
            "answer":     cleaned if cleaned else "NO_ANSWER",
            "confidence": 0.0,
            "reasoning":  "Failed to parse structured response.",
        }

# ── Router & Analytical Execution ─────────────────────────────────────────────

def route_query(query: str, groq_api_key: str) -> str:
    try:
        client = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": query},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        res = response.choices[0].message.content.strip().upper()
        return "ANALYTICAL" if "ANALYTICAL" in res else "SEMANTIC"
    except Exception as e:
        logger.error(f"Router failed: {e}")
        return "SEMANTIC"

def execute_analytical_query(query: str, tables_dir: str, groq_api_key: str) -> dict:
    folder = Path(tables_dir)
    csv_files = list(folder.glob("*.csv"))
    if not csv_files:
         return {"status": "error", "reasoning": "No CSV files are available for analytical querying."}
    
    schema_lines = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, nrows=2)
            schema_lines.append(f"File: {f.name} | Columns: {list(df.columns)}")
        except Exception:
            pass
            
    if not schema_lines:
         return {"status": "error", "reasoning": "Failed to read schemas from available CSVs."}
         
    schema_context = "\n".join(schema_lines)
    
    try:
        client = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": PANDAS_PROMPT.format(schema_context=schema_context, query=query)},
            ],
            temperature=0.0
        )
        code = response.choices[0].message.content.strip()
        code = re.sub(r'```(?:python)?', '', code).strip().strip("`")
    except Exception as e:
        return {"status": "error", "reasoning": f"Data Analyst LLM failed: {e}"}
    
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    error = None
    try:
        exec_globals = {"pd": pd, "__builtins__": __builtins__}
        exec(code, exec_globals)
    except Exception as e:
        error = str(e)
    finally:
        sys.stdout = old_stdout
        
    output = redirected_output.getvalue().strip()
    
    if error:
        return {"status": "error", "reasoning": f"Pandas execution returned an error: {error}\nCode: {code}"}
    if not output:
        return {"status": "error", "reasoning": f"Pandas script returned no output.\nCode: {code}"}
    if "NO_ANSWER_FOUND" in output or "do not contain" in output.lower() or "not found" in output.lower():
        return {"status": "error", "reasoning": "Pandas execution analyzed the tables but could not find the data. Falling back to semantic search."}

        
    return {
        "answer": output,
        "confidence": 1.0,
        "citations": [{"document": "Tabular Dataset", "chunk_id": "PANDAS-EXEC", "text": "Raw execution across entire CSV dataset", "similarity_score": 1.0}],
        "status": "success",
        "reasoning": f"Mathematically derived answer calculated via Pandas on raw tabular data.\nCode Executed:\n{code}",
        "suggested_action": None
    }


# ── Main Generation Function ──────────────────────────────────────────────────

def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    groq_api_key: str,
) -> dict:
    """
    Generate a strictly grounded answer from retrieved chunks.

    Returns structured response:
    {
      "answer": str,
      "confidence": float,
      "citations": [...],
      "status": "success" | "no_answer" | "error",
      "reasoning": str
    }
    """
    # ── Guard: no chunks retrieved ────────────────────────────────────────────
    if not retrieved_chunks:
        return {
            "answer":     "No relevant information found in the provided documents.",
            "confidence": 0.0,
            "citations":  [],
            "status":     "no_answer",
            "reasoning":  "No chunks were retrieved above the similarity threshold.",
            "suggested_action": None,
        }

    # ── Guard: top chunk below minimum threshold ──────────────────────────────
    top_score = retrieved_chunks[0]["similarity_score"]
    if top_score < MIN_SIMILARITY_THRESHOLD:
        return {
            "answer":     "No relevant information found in the provided documents.",
            "confidence": 0.0,
            "citations":  [],
            "status":     "no_answer",
            "reasoning":  (
                f"Best match score ({top_score:.2f}) is below the minimum threshold "
                f"({MIN_SIMILARITY_THRESHOLD}). Try rephrasing your question."
            ),
            "suggested_action": None,
        }

    # ── Build context ─────────────────────────────────────────────────────────
    context_block = _build_context_block(retrieved_chunks)

    user_message = (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer using ONLY the context above. Return JSON."
    )

    # ── Call Groq ─────────────────────────────────────────────────────────────
    try:
        client = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",   # Upgraded from 3.1-8b-instant
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=800,
            temperature=0.1,  # Slight variance for natural phrasing; still grounded
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return {
            "answer":     "Generation failed due to an API error.",
            "confidence": 0.0,
            "citations":  [],
            "status":     "error",
            "reasoning":  str(e),
        }

    # ── Parse response ────────────────────────────────────────────────────────
    parsed           = _parse_llm_response(raw)
    answer           = parsed.get("answer", "NO_ANSWER")
    confidence       = float(parsed.get("confidence", 0.0))
    reasoning        = parsed.get("reasoning", "")
    suggested_action = parsed.get("suggested_action")

    logger.info(f"Query: '{query[:60]}' | confidence: {confidence} | top_chunk: {top_score:.2f}")

    # ── Handle explicit no-answer ─────────────────────────────────────────────
    if answer.strip().upper() == "NO_ANSWER" or confidence < CONFIDENCE_FLOOR:
        return {
            "answer":     "No relevant information found in the provided documents.",
            "confidence": confidence,
            "citations":  [],
            "status":     "no_answer",
            "reasoning":  reasoning or "Model determined context was insufficient.",
            "suggested_action": None,
        }

    # ── Build citations ───────────────────────────────────────────────────────
    citations = [
        {
            "document":         c["document"],
            "chunk_id":         c["chunk_id"],
            "text":             c["text"][:400] + ("..." if len(c["text"]) > 400 else ""),
            "similarity_score": c["similarity_score"],
        }
        for c in retrieved_chunks
    ]

    return {
        "answer":           answer,
        "confidence":       confidence,
        "citations":        citations,
        "status":           "success",
        "reasoning":        reasoning,
        "suggested_action": suggested_action,
    }