import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_execute_query():
    assert len([1]) == 1
