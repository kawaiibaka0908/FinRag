"""
Stock prediction scoring system for FinRAG.
"""
import importlib

__all__ = [
    'EnsembleScorer',
    'ScoringConfig', 
    'ScoringResult',
    'IndianFinancialNewsFetcher',
    'FinancialDataFetcher'
]

def __getattr__(name: str):
    if name in ('EnsembleScorer', 'ScoringConfig', 'ScoringResult'):
        mod = importlib.import_module('.ensemble_scorer', __name__)
        return getattr(mod, name)
    elif name in ('IndianFinancialNewsFetcher',):
        mod = importlib.import_module('.news_fetcher', __name__)
        return getattr(mod, name)
    elif name in ('FinancialDataFetcher',):
        mod = importlib.import_module('.financial_data_fetcher', __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
