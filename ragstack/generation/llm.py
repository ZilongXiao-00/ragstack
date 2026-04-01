"""生成模块"""

from typing import List
import requests


class LLMGenerator:
    """LLM生成器"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int = 500,
        temperature: float = 0.1
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # 构建完整的API URL
        if not base_url.endswith("/v1/chat/completions"):
            self.api_url = base_url.rstrip("/") + "/v1/chat/completions"
        else:
            self.api_url = base_url

    def generate(self, query: str, chunks: List[str]) -> str:
        """
        生成回答

        Args:
            query: 用户问题
            chunks: 相关文本片段

        Returns:
            生成的回答
        """
        context = "\n\n".join(chunks)

        prompt = f"""### 角色
你是一位专业的知识问答助手，基于给定的文本片段回答用户问题。

### 指示
1. 严格依据「相关片段」中的信息回答，不得使用外部知识
2. 回答需简洁、准确，贴合问题核心
3. 若片段中无足够信息，回复："未找到与该问题相关的有效信息"
4. 回答通俗易懂，控制在300字以内

### 相关片段
{context}

### 用户问题
{query}

### 输出"""

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

        response = requests.post(self.api_url, headers=headers, json=data)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API请求失败: {response.status_code}, {response.text}")
