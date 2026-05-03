import requests


def test_requests_installed() -> None:
    assert requests.__version__


def test_example_com_reachable() -> None:
    response = requests.get("https://example.com", timeout=10)
    assert response.status_code == 200
