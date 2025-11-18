#!/usr/bin/env python3
"""
Unit tests for RAG query system with 5 example prompts.

Tests both the basic retrieval (rag_query.py) and the Gemini-based approach (rag_query_gemini.py)
to ensure accurate answers are returned.

Usage:
    py -3 scripts/test_rag.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rag_query import retrieve, compose_answer

# Test cases with expected answers covering all three schemes
TEST_CASES = [
    # JM Value Fund tests
    {
        "question": "What is the expense ratio of JM Value Fund?",
        "expected_keywords": ["0.98%", "expense ratio"],
        "expected_url": "https://groww.in/mutual-funds/jm-basic-fund-direct-growth",
    },
    {
        "question": "What is the exit load for JM Value Fund?",
        "expected_keywords": ["1%", "30 days", "exit load"],
        "expected_url": "https://groww.in/mutual-funds/jm-basic-fund-direct-growth",
    },
    # JM Aggressive Hybrid Fund tests
    {
        "question": "What is the expense ratio of JM Aggressive Hybrid Fund?",
        "expected_keywords": ["0.67%", "expense ratio"],
        "expected_url": "https://groww.in/mutual-funds/jm-aggressive-hybrid-fund-direct-growth",
    },
    {
        "question": "What is the exit load for JM Aggressive Hybrid Fund?",
        "expected_keywords": ["1%", "60 days", "exit load"],
        "expected_url": "https://groww.in/mutual-funds/jm-aggressive-hybrid-fund-direct-growth",
    },
    # JM Flexicap Fund tests
    {
        "question": "What is the expense ratio of JM Flexicap Fund?",
        "expected_keywords": ["0.54%", "expense ratio"],
        "expected_url": "https://groww.in/mutual-funds/jm-multi-strategy-fund-direct-growth",
    },
    {
        "question": "What is the minimum SIP investment for JM Flexicap Fund?",
        "expected_keywords": ["₹100", "minimum sip"],
        "expected_url": "https://groww.in/mutual-funds/jm-multi-strategy-fund-direct-growth",
    },
    # Guide test
    {
        "question": "How to download mutual fund statement from Groww?",
        "expected_keywords": ["Groww app", "You", "SIP & Reports", "Capital Gain"],
        "expected_url": "https://groww.in/blog/how-to-get-capital-gains-statement-for-mutual-fund-investments",
    },
]


def check_keywords_in_text(text: str, keywords: List[str]) -> tuple[bool, List[str]]:
    """Check if all expected keywords are present in the text (case-insensitive)."""
    text_lower = text.lower()
    missing_keywords = []
    
    for keyword in keywords:
        if keyword.lower() not in text_lower:
            missing_keywords.append(keyword)
    
    return len(missing_keywords) == 0, missing_keywords


def check_url_in_text(text: str, expected_url: str) -> bool:
    """Check if the expected URL is present in the text."""
    return expected_url in text


def run_test_case(test_num: int, test_case: Dict) -> bool:
    """Run a single test case and return whether it passed."""
    print(f"\n{'='*80}")
    print(f"Test Case #{test_num}: {test_case['question']}")
    print(f"{'='*80}")
    
    # Retrieve relevant documents
    hits = retrieve(test_case['question'], top_k=3)
    
    # Compose answer
    answer = compose_answer(test_case['question'], hits)
    
    print(f"\nRetrieved Answer:")
    print(f"{answer}")
    
    # Check if expected keywords are present
    keywords_found, missing_keywords = check_keywords_in_text(
        answer, test_case['expected_keywords']
    )
    
    # Check if expected URL is present
    url_found = check_url_in_text(answer, test_case['expected_url'])
    
    # Determine if test passed
    test_passed = keywords_found and url_found
    
    print(f"\n{'─'*80}")
    print(f"Test Results:")
    print(f"  ✓ Keywords found: {keywords_found}")
    if not keywords_found:
        print(f"    Missing keywords: {missing_keywords}")
    print(f"  ✓ URL found: {url_found}")
    if not url_found:
        print(f"    Expected URL: {test_case['expected_url']}")
    
    print(f"\n{'─'*80}")
    if test_passed:
        print(f"✅ TEST PASSED")
    else:
        print(f"❌ TEST FAILED")
    
    return test_passed


def main() -> None:
    """Run all test cases and report results."""
    print("="*80)
    print("RAG QUERY SYSTEM - UNIT TESTS")
    print("="*80)
    print(f"\nRunning {len(TEST_CASES)} test cases...\n")
    
    results = []
    for idx, test_case in enumerate(TEST_CASES, start=1):
        passed = run_test_case(idx, test_case)
        results.append((idx, test_case['question'], passed))
    
    # Summary
    print(f"\n\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    passed_count = sum(1 for _, _, passed in results if passed)
    total_count = len(results)
    
    for idx, question, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - Test #{idx}: {question[:60]}...")
    
    print(f"\n{'─'*80}")
    print(f"Total: {passed_count}/{total_count} tests passed ({passed_count/total_count*100:.1f}%)")
    print(f"{'='*80}")
    
    # Exit with appropriate code
    sys.exit(0 if passed_count == total_count else 1)


if __name__ == "__main__":
    main()
