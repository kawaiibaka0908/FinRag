"""
FinRAG Orchestrator package.
"""
import importlib

__all__ = [
    "FinRAGOrchestrator",
    "OrchestratorConfig", 
    "ToolRegistry",
    "Tool",
    "ToolResult",
    "IndiaMultiAgentOrchestrator"
]

def __getattr__(name: str):
    if name in ("FinRAGOrchestrator", "OrchestratorConfig"):
        mod = importlib.import_module(".orchestrator", __name__)
        return getattr(mod, name)
    elif name in ("ToolRegistry", "Tool", "ToolResult"):
        mod = importlib.import_module(".tools", __name__)
        return getattr(mod, name)
    elif name in ("IndiaMultiAgentOrchestrator",):
        mod = importlib.import_module(".india_orchestrator", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
