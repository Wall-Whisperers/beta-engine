import requests


def main() -> None:
    response = requests.get("https://example.com", timeout=10)
    print(f"status={response.status_code}")


if __name__ == "__main__":
    main()
