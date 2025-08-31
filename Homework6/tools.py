from typing import Optional

def search_arxiv(query: str) -> str:
    """
    Simulate an arXiv search or return a placeholder passage for the given query.
    In a real system, you can query the arXiv API and return a concise summary.
    """
    if not query or not isinstance(query, str):
        return "Error: empty query."
    # Placeholder snippet
    return f"[arXiv] Top result for '{query}': A recent paper discusses core concepts and findings. Summary: ..."

def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression using sympy for safety and return the result.
    Supports algebraic expressions like '2+2', 'sqrt(2)', 'sin(pi/2)', 'integrate(x, (x,0,1))' etc.
    """
    try:
        from sympy import sympify
        result = sympify(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"
