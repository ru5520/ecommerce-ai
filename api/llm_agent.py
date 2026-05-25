import os

from openai import OpenAI

_api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
client = OpenAI(api_key=_api_key, base_url="https://api.deepseek.com") if _api_key else None


def is_available() -> bool:
    return client is not None


def rewrite_query(user_query: str) -> str:
    if client is None:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY，请在终端执行: $env:DEEPSEEK_API_KEY=\"你的密钥\"")
    prompt = f"""你是电商AI购物助手。商品库为英文服装标题（如 blue shirt, dress, watch）。

用户需求：
{user_query}

请转成适合检索的英文关键词（含颜色、品类），一行输出，不要解释。
例：蓝色衬衫 -> blue shirt
例：男士手表 -> men watch"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def chat(prompt: str) -> str:
    """通用导购对话（完整 prompt 由调用方构造）。"""
    if client is None:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    print(rewrite_query("蓝色衬衫"))
