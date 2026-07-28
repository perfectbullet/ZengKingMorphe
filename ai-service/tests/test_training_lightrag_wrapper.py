"""LightRAG 包装器的基础参数测试。"""

import sys
from types import ModuleType


def test_query_param_does_not_inject_keywords(monkeypatch):
    """关键词应由 LightRAG 自行提取，包装器不注入 hl/ll keywords。"""
    from app.services import training_lightrag_wrapper as wrapper

    class FakeQueryParam:
        def __init__(self, mode, stream, top_k, **kwargs):
            self.mode = mode
            self.stream = stream
            self.top_k = top_k
            self.kwargs = kwargs

    fake_base = ModuleType("lightrag.base")
    fake_base.QueryParam = FakeQueryParam
    monkeypatch.setitem(sys.modules, "lightrag.base", fake_base)

    _, kwargs, debug = wrapper._build_query_param("hybrid", True)

    assert "hl_keywords" not in kwargs
    assert "ll_keywords" not in kwargs
    assert "hl_keywords" not in debug["candidate_kwargs_keys"]
    assert "ll_keywords" not in debug["candidate_kwargs_keys"]
