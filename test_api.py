import requests
import json

url = "http://127.0.0.1:8000/image-search"

products = [
    {
        "title": "gaming laptop",
        "description": "RTX4060 gaming laptop"
    },
    {
        "title": "school notebook",
        "description": "lightweight notebook"
    },
    {
        "title": "desktop computer",
        "description": "high performance gaming pc"
    }
]

with open("D:/true/ecommerce_ai_project/scripts/scripts/test.jpg", "rb") as f:

    response = requests.post(
        url,
        files={
            "file": f
        },
        data={
            "products": json.dumps(products)
        }
    )

print(response.json())