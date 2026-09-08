import pytest
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_pipeline import BISRAGPipeline
from consumer_complaint import ConsumerComplaintHandler
from scheme_walkthrough import SchemeWalkthroughGuide
from lab_locator import LabLocator
from product_recommender import ProductRecommender
from router import QueryIntentRouter
from guardrails import GuardrailGate, CONFIDENCE_THRESHOLD


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live: mark test as requiring active internet and live LLM API keys (run with 'pytest -m live')"
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "_live.py" in str(item.fspath):
            item.add_marker(pytest.mark.live)


@pytest.fixture(scope="session")
def pipeline():
    return BISRAGPipeline(llm_provider="mock", use_mock_retrieval=False)


@pytest.fixture(scope="session")
def live_pipeline():
    return BISRAGPipeline(llm_provider=None, use_mock_retrieval=False)


@pytest.fixture(scope="session")
def handler(pipeline):
    return ConsumerComplaintHandler(
        retrieval_pipeline=pipeline.retrieval,
        generator=pipeline.generator,
        citation_engine=pipeline.citation_engine,
    )


@pytest.fixture(scope="session")
def guide(pipeline):
    return SchemeWalkthroughGuide(
        retrieval_pipeline=pipeline.retrieval,
        generator=pipeline.generator,
        citation_engine=pipeline.citation_engine,
    )


@pytest.fixture(scope="session")
def locator(pipeline):
    return LabLocator(
        retrieval_pipeline=pipeline.retrieval,
        generator=pipeline.generator,
        citation_engine=pipeline.citation_engine,
    )


@pytest.fixture(scope="session")
def recommender(pipeline):
    return ProductRecommender(
        retrieval_pipeline=pipeline.retrieval,
        generator=pipeline.generator,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )


@pytest.fixture(scope="session")
def router(pipeline):
    return QueryIntentRouter(encoder=pipeline.retrieval.encoder)


@pytest.fixture(scope="session")
def gate():
    return GuardrailGate(threshold=CONFIDENCE_THRESHOLD)
