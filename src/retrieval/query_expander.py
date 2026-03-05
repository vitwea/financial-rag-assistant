"""
query_expander.py
-----------------
Expands user queries with domain-specific synonyms before FAISS retrieval.

Solves vocabulary mismatch between natural language queries and the actual
terminology used in each company's 10-K filing.

Key example:
    Users ask about "cloud" but Apple's 10-K calls it "Services".
    Without expansion, Apple chunks never appear in cross-company queries.

Design:
    - Expansion is applied only at the FAISS stage (broad recall).
    - The original query is kept intact for Cohere re-ranking (precise relevance).
"""

# ---------------------------------------------------------------------------
# Synonym map — keys are query terms, values are the 10-K equivalents
# ---------------------------------------------------------------------------
SYNONYM_MAP: dict[str, dict[str, str]] = {
    "apple": {
        "cloud":            "Services iCloud",
        "cloud revenue":    "Services segment revenue App Store iCloud",
        "cloud growth":     "Services revenue growth",
        "cloud margin":     "Services gross margin",
        "saas":             "Services software subscription",
        "subscription":     "Services Apple One",
        "streaming":        "Apple TV+ Apple Music Services",
        "enterprise":       "Mac iPad business enterprise",
        "hardware":         "iPhone Mac iPad product revenue",
        "wearables":        "Apple Watch AirPods accessories",
    },
    "tesla": {
        "cloud":            "software connectivity services",
        "profit":           "gross margin net income operating income",
        "factory":          "Gigafactory manufacturing plant",
        "supply chain":     "manufacturing suppliers components raw materials",
        "competition":      "competitors automotive EV market",
        "energy":           "Powerwall Megapack solar energy storage",
        "autonomous":       "Full Self-Driving Autopilot FSD",
        "delivery":         "vehicle deliveries production units",
    },
    "microsoft": {
        "cloud":            "Azure intelligent cloud",
        "saas":             "Microsoft 365 commercial cloud subscription",
        "office":           "Microsoft 365 productivity",
        "gaming":           "Xbox gaming Activision",
        "search":           "Bing search advertising",
        "enterprise":       "commercial cloud enterprise agreements",
        "ai":               "Azure OpenAI Copilot artificial intelligence",
        "linkedin":         "LinkedIn professional network",
    },
}


def expand_query(query: str, company: str | None = None) -> str:
    """
    Append relevant synonym terms to the query for better FAISS recall.

    Args:
        query   : original user question
        company : if set, apply only that company's synonym map;
                  if None, apply all maps (for cross-company queries)

    Returns:
        Expanded query string with appended domain terms.
        Returns the original query unchanged if no synonyms match.

    Example:
        expand_query("Compare cloud revenue", "apple")
        → "Compare cloud revenue Services iCloud App Store"

        expand_query("Compare cloud revenue")   # cross-company
        → "Compare cloud revenue Services iCloud Azure intelligent cloud ..."
    """
    companies_to_check = (
        [company] if company and company in SYNONYM_MAP
        else list(SYNONYM_MAP.keys())
    )

    extra_terms: set[str] = set()
    query_lower = query.lower()

    for comp in companies_to_check:
        for term, expansion in SYNONYM_MAP[comp].items():
            if term in query_lower:
                extra_terms.update(expansion.split())

    if not extra_terms:
        return query  # no expansion needed

    expanded = query + " " + " ".join(sorted(extra_terms))
    print(f"  Query expanded  : \"{expanded[:120]}{'…' if len(expanded) > 120 else ''}\"")
    return expanded
