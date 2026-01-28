#!/usr/bin/env python3
from dotenv import load_dotenv

# Load default environment variables (.env)
load_dotenv()

import os
import time
import logging
from collections import deque
from typing import Dict, List
import importlib
import openai
import chromadb
import tiktoken as tiktoken
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import re
import json
from tools.taobao import search_taobao
import requests
from typing import Tuple

# default opt out of chromadb telemetry.
from chromadb.config import Settings

LAST_TAOBAO_CANDIDATES_PATH = "runs/taobao_candidates.json"
LAST_TAOBAO_FILE_ID = None  # 上传到 OpenAI Files 后拿到的 file_id

client = chromadb.Client(Settings(anonymized_telemetry=False))

# Engine configuration

# Model: GPT, LLAMA, HUMAN, etc.
LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")).lower()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not (LLM_MODEL.startswith("llama") or LLM_MODEL.startswith("human")):
    assert OPENAI_API_KEY, "\033[91m\033[1m" + "OPENAI_API_KEY environment variable is missing from .env" + "\033[0m\033[0m"

# Table config
RESULTS_STORE_NAME = os.getenv("RESULTS_STORE_NAME", os.getenv("TABLE_NAME", ""))
assert RESULTS_STORE_NAME, "\033[91m\033[1m" + "RESULTS_STORE_NAME environment variable is missing from .env" + "\033[0m\033[0m"

# Run configuration
INSTANCE_NAME = os.getenv("INSTANCE_NAME", os.getenv("BABY_NAME", "BabyAGI"))
COOPERATIVE_MODE = "none"
JOIN_EXISTING_OBJECTIVE = False

# Goal configuration
OBJECTIVE = os.getenv("OBJECTIVE", "")
INITIAL_TASK = os.getenv("INITIAL_TASK", os.getenv("FIRST_TASK", ""))

# Model configuration
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))


# Extensions support begin

def can_import(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


DOTENV_EXTENSIONS = os.getenv("DOTENV_EXTENSIONS", "").split(" ")

# Command line arguments extension
# Can override any of the above environment variables
ENABLE_COMMAND_LINE_ARGS = (
        os.getenv("ENABLE_COMMAND_LINE_ARGS", "false").lower() == "true"
)
if ENABLE_COMMAND_LINE_ARGS:
    if can_import("extensions.argparseext"):
        from extensions.argparseext import parse_arguments

        OBJECTIVE, INITIAL_TASK, LLM_MODEL, DOTENV_EXTENSIONS, INSTANCE_NAME, COOPERATIVE_MODE, JOIN_EXISTING_OBJECTIVE = parse_arguments()

# Human mode extension
# Gives human input to babyagi
if LLM_MODEL.startswith("human"):
    if can_import("extensions.human_mode"):
        from extensions.human_mode import user_input_await

# Load additional environment variables for enabled extensions
# TODO: This might override the following command line arguments as well:
#    OBJECTIVE, INITIAL_TASK, LLM_MODEL, INSTANCE_NAME, COOPERATIVE_MODE, JOIN_EXISTING_OBJECTIVE
if DOTENV_EXTENSIONS:
    if can_import("extensions.dotenvext"):
        from extensions.dotenvext import load_dotenv_extensions

        load_dotenv_extensions(DOTENV_EXTENSIONS)

# TODO: There's still work to be done here to enable people to get
# defaults from dotenv extensions, but also provide command line
# arguments to override them

# Extensions support end

print("\033[95m\033[1m" + "\n*****CONFIGURATION*****\n" + "\033[0m\033[0m")
print(f"Name  : {INSTANCE_NAME}")
print(
    f"Mode  : {'alone' if COOPERATIVE_MODE in ['n', 'none'] else 'local' if COOPERATIVE_MODE in ['l', 'local'] else 'distributed' if COOPERATIVE_MODE in ['d', 'distributed'] else 'undefined'}")
print(f"LLM   : {LLM_MODEL}")

# Check if we know what we are doing
assert OBJECTIVE, "\033[91m\033[1m" + "OBJECTIVE environment variable is missing from .env" + "\033[0m\033[0m"
assert INITIAL_TASK, "\033[91m\033[1m" + "INITIAL_TASK environment variable is missing from .env" + "\033[0m\033[0m"

LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "models/llama-13B/ggml-model.bin")
if LLM_MODEL.startswith("llama"):
    if can_import("llama_cpp"):
        from llama_cpp import Llama

        print(f"LLAMA : {LLAMA_MODEL_PATH}" + "\n")
        assert os.path.exists(LLAMA_MODEL_PATH), "\033[91m\033[1m" + f"Model can't be found." + "\033[0m\033[0m"

        CTX_MAX = 1024
        LLAMA_THREADS_NUM = int(os.getenv("LLAMA_THREADS_NUM", 8))

        print('Initialize model for evaluation')
        llm = Llama(
            model_path=LLAMA_MODEL_PATH,
            n_ctx=CTX_MAX,
            n_threads=LLAMA_THREADS_NUM,
            n_batch=512,
            use_mlock=False,
        )

        print('\nInitialize model for embedding')
        llm_embed = Llama(
            model_path=LLAMA_MODEL_PATH,
            n_ctx=CTX_MAX,
            n_threads=LLAMA_THREADS_NUM,
            n_batch=512,
            embedding=True,
            use_mlock=False,
        )

        print(
            "\033[91m\033[1m"
            + "\n*****USING LLAMA.CPP. POTENTIALLY SLOW.*****"
            + "\033[0m\033[0m"
        )
    else:
        print(
            "\033[91m\033[1m"
            + "\nLlama LLM requires package llama-cpp. Falling back to GPT-3.5-turbo."
            + "\033[0m\033[0m"
        )
        LLM_MODEL = "gpt-3.5-turbo"

if LLM_MODEL.startswith("gpt-4"):
    print(
        "\033[91m\033[1m"
        + "\n*****USING GPT-4. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"
        + "\033[0m\033[0m"
    )

if LLM_MODEL.startswith("human"):
    print(
        "\033[91m\033[1m"
        + "\n*****USING HUMAN INPUT*****"
        + "\033[0m\033[0m"
    )

print("\033[94m\033[1m" + "\n*****OBJECTIVE*****\n" + "\033[0m\033[0m")
print(f"{OBJECTIVE}")

if not JOIN_EXISTING_OBJECTIVE:
    print("\033[93m\033[1m" + "\nInitial task:" + "\033[0m\033[0m" + f" {INITIAL_TASK}")
else:
    print("\033[93m\033[1m" + f"\nJoining to help the objective" + "\033[0m\033[0m")

# Configure OpenAI
openai.api_key = OPENAI_API_KEY


# Llama embedding function
class LlamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        return

    def __call__(self, texts: Documents) -> Embeddings:
        embeddings = []
        for t in texts:
            e = llm_embed.embed(t)
            embeddings.append(e)
        return embeddings


# Results storage using local ChromaDB
class DefaultResultsStorage:
    def __init__(self):
        logging.getLogger('chromadb').setLevel(logging.ERROR)
        # Create Chroma collection
        chroma_persist_dir = "chroma"
        chroma_client = chromadb.PersistentClient(
            settings=chromadb.config.Settings(
                persist_directory=chroma_persist_dir,
            )
        )

        metric = "cosine"
        if LLM_MODEL.startswith("llama"):
            embedding_function = LlamaEmbeddingFunction()
        else:
            embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)
        self.collection = chroma_client.get_or_create_collection(
            name=RESULTS_STORE_NAME,
            metadata={"hnsw:space": metric},
            embedding_function=embedding_function,
        )

    def add(self, task: Dict, result: str, result_id: str):

        # Break the function if LLM_MODEL starts with "human" (case-insensitive)
        if LLM_MODEL.startswith("human"):
            return
        # Continue with the rest of the function

        embeddings = llm_embed.embed(result) if LLM_MODEL.startswith("llama") else None
        if (
                len(self.collection.get(ids=[result_id], include=[])["ids"]) > 0
        ):  # Check if the result already exists
            self.collection.update(
                ids=result_id,
                embeddings=embeddings,
                documents=result,
                metadatas={"task": task["task_name"], "result": result},
            )
        else:
            self.collection.add(
                ids=result_id,
                embeddings=embeddings,
                documents=result,
                metadatas={"task": task["task_name"], "result": result},
            )

    def query(self, query: str, top_results_num: int) -> List[dict]:
        count: int = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(
            query_texts=query,
            n_results=min(top_results_num, count),
            include=["metadatas"]
        )
        return [item["task"] for item in results["metadatas"][0]]


# Initialize results storage
def try_weaviate():
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
    WEAVIATE_USE_EMBEDDED = os.getenv("WEAVIATE_USE_EMBEDDED", "False").lower() == "true"
    if (WEAVIATE_URL or WEAVIATE_USE_EMBEDDED) and can_import("extensions.weaviate_storage"):
        WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
        from extensions.weaviate_storage import WeaviateResultsStorage
        print("\nUsing results storage: " + "\033[93m\033[1m" + "Weaviate" + "\033[0m\033[0m")
        return WeaviateResultsStorage(OPENAI_API_KEY, WEAVIATE_URL, WEAVIATE_API_KEY, WEAVIATE_USE_EMBEDDED, LLM_MODEL,
                                      LLAMA_MODEL_PATH, RESULTS_STORE_NAME, OBJECTIVE)
    return None


def try_pinecone():
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    if PINECONE_API_KEY and can_import("extensions.pinecone_storage"):
        PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
        assert (
            PINECONE_ENVIRONMENT
        ), "\033[91m\033[1m" + "PINECONE_ENVIRONMENT environment variable is missing from .env" + "\033[0m\033[0m"
        from extensions.pinecone_storage import PineconeResultsStorage
        print("\nUsing results storage: " + "\033[93m\033[1m" + "Pinecone" + "\033[0m\033[0m")
        return PineconeResultsStorage(OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, LLM_MODEL,
                                      LLAMA_MODEL_PATH, RESULTS_STORE_NAME, OBJECTIVE)
    return None


def use_chroma():
    print("\nUsing results storage: " + "\033[93m\033[1m" + "Chroma (Default)" + "\033[0m\033[0m")
    return DefaultResultsStorage()


results_storage = try_weaviate() or try_pinecone() or use_chroma()


# Task storage supporting only a single instance of BabyAGI
class SingleTaskListStorage:
    def __init__(self):
        self.tasks = deque([])
        self.task_id_counter = 0

    def append(self, task: Dict):
        self.tasks.append(task)

    def replace(self, tasks: List[Dict]):
        self.tasks = deque(tasks)

    def popleft(self):
        return self.tasks.popleft()

    def is_empty(self):
        return False if self.tasks else True

    def next_task_id(self):
        self.task_id_counter += 1
        return self.task_id_counter

    def get_task_names(self):
        return [t["task_name"] for t in self.tasks]


# Initialize tasks storage
tasks_storage = SingleTaskListStorage()
if COOPERATIVE_MODE in ['l', 'local']:
    if can_import("extensions.ray_tasks"):
        import sys
        from pathlib import Path

        sys.path.append(str(Path(__file__).resolve().parent))
        from extensions.ray_tasks import CooperativeTaskListStorage

        tasks_storage = CooperativeTaskListStorage(OBJECTIVE)
        print("\nReplacing tasks storage: " + "\033[93m\033[1m" + "Ray" + "\033[0m\033[0m")
elif COOPERATIVE_MODE in ['d', 'distributed']:
    pass


def limit_tokens_from_string(string: str, model: str, limit: int) -> str:
    """Limits the string to a number of tokens (estimated)."""

    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        encoding = tiktoken.encoding_for_model('gpt2')  # Fallback for others.

    encoded = encoding.encode(string)

    return encoding.decode(encoded[:limit])


# 新增的函数schema
'''
OpenAI 的 function calling 需要你给模型一个“可调用函数定义”。你当前的 openai_call() 用的是 openai.ChatCompletion.create(...)，但没有 functions 参数，所以模型根本不知道有工具可用。
babyagi
改哪里
在 openai_call 函数定义附近（紧跟在 openai_call 上方/下方都可以），新增一个全局常量。
'''
SEARCH_TAOBAO_FUNCTION = {
    "name": "search_taobao",
    "description": "Search Taobao by keyword (mock crawler) and save candidate items to a local JSON file.",
    "parameters": {
        "type": "object",
        "properties": {
            "keyword": {
                "type": "string",
                "description": "Search keyword, short Chinese phrase, e.g., '2024 秋季 女装 衬衫'."
            },
            "n": {
                "type": "integer",
                "description": "How many candidate items to generate/save",
                "default": 300
            },
            "save_path": {
                "type": "string",
                "description": "Where to save JSON candidates",
                "default": "runs/taobao_candidates.json"
            },
            "merge": {
                "type": "boolean",
                "description": "Whether to merge with existing cached candidates",
                "default": False
            },
            "preview_k": {
                "type": "integer",
                "description": "How many items to include in tool preview",
                "default": 5
            }
        },
        "required": ["keyword"]
    }
}

'''
你现有 openai_call() 不支持 functions / function_call，因此无法让模型触发工具调用。
babyagi

做法：复制 openai_call() 的重试框架，新增一个 openai_call_with_functions()。

'''


def openai_call_with_functions(
        messages: List[Dict],
        functions: List[Dict],
        model: str = LLM_MODEL,
        temperature: float = OPENAI_TEMPERATURE,
        max_tokens: int = 800,
):
    while True:
        try:
            # 仅对 GPT-chat 模型启用 functions
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                functions=functions,
                function_call="auto",
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                stop=None,
            )
            return response.choices[0].message  # dict: {"role","content", optional "function_call"}
        except openai.error.RateLimitError:
            print("   *** The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again. ***")
            time.sleep(10)
        except openai.error.Timeout:
            print("   *** OpenAI API timeout occurred. Waiting 10 seconds and trying again. ***")
            time.sleep(10)
        except openai.error.APIError:
            print("   *** OpenAI API error occurred. Waiting 10 seconds and trying again. ***")
            time.sleep(10)
        except openai.error.APIConnectionError:
            print("   *** OpenAI API connection error occurred. Waiting 10 seconds and trying again. ***")
            time.sleep(10)
        except openai.error.InvalidRequestError:
            print("   *** OpenAI API invalid request. Waiting 10 seconds and trying again. ***")
            time.sleep(10)
        except openai.error.ServiceUnavailableError:
            print("   *** OpenAI API service unavailable. Waiting 10 seconds and trying again. ***")
            time.sleep(10)
        else:
            break


def upload_file_user_data(path: str, purpose: str = "user_data") -> str:
    """
    上传本地文件到 OpenAI Files，返回 file_id
    参考：Files API + purpose=user_data（推荐用于当作模型输入）:contentReference[oaicite:2]{index=2}
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    url = "https://api.openai.com/v1/files"
    headers = {"Authorization": f"Bearer {api_key}"}

    with open(path, "rb") as f:
        files = {"file": (os.path.basename(path), f)}
        data = {"purpose": purpose}
        r = requests.post(url, headers=headers, data=data, files=files, timeout=120)

    r.raise_for_status()
    j = r.json()
    file_id = j.get("id")
    if not file_id:
        raise RuntimeError(f"Upload succeeded but no file id returned: {j}")
    return file_id


def _extract_output_text(resp_json: dict) -> str:
    """
    从 Responses API 返回结构里提取最终文本（message.content[].type=output_text）
    """
    # 有些 SDK 会直接给 output_text；我们这里走通用 JSON 解析
    chunks = []
    for item in resp_json.get("output", []) or []:
        if item.get("type") == "message":
            for c in item.get("content", []) or []:
                if c.get("type") in ("output_text", "text"):
                    t = c.get("text")
                    if isinstance(t, str) and t.strip():
                        chunks.append(t)
    if chunks:
        return "\n".join(chunks).strip()

    # fallback
    ot = resp_json.get("output_text")
    if isinstance(ot, str):
        return ot.strip()
    return str(resp_json)[:2000]



def analyze_with_file(objective: str, task: str, file_id: str, model: str = "gpt-4.1") -> str:
    """
    用 Responses API + Code Interpreter 读取 container 里的 taobao_candidates.json 并按 objective 产出答案。
    这个版本会返回“可验证的审计信息（AUDIT）”和 python tool 执行日志，证明模型确实读了文件并计算。
    """
    # 优先用环境变量（建议你 .env 里设置 ANALYSIS_MODEL="gpt-4.1"）
    model = model or os.getenv("ANALYSIS_MODEL") or "gpt-4.1"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "ERROR: OPENAI_API_KEY not set."

    if not file_id:
        return "ERROR: file_id is empty. Did you upload the candidates json to OpenAI Files?"

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 让模型必须用 python tool，并输出“不可伪造”的 AUDIT 信息
    instructions = (
        "You are a Taobao data analyst.\n"
        "You MUST use the python tool to read the uploaded JSON file and compute the answer.\n"
        "You MUST print a single-line audit JSON in python logs exactly like:\n"
        "__AUDIT__={...}\n"
        "The audit JSON must include sha256, total_rows, filtered_rows, sort_key, topk, and top_sales_list.\n"
        "If you cannot run python or cannot find the file, you must say FAILED.\n"
    )

    # 注意：这里强制它输出“审计证据”，用于验证它确实读文件算过
    user_input = f"""
OBJECTIVE（用户目标）：
{objective}

CURRENT TASK（当前任务）：
{task}

你将获得一个已上传到 Code Interpreter 容器的 JSON 文件（通常名为 taobao_candidates.json，内容是 list[dict]）。
请你必须在 python tool 中完成以下步骤（不可省略）：

【A. 证明你读了文件（审计证据）】
1) 列出当前目录文件（如 ls / 递归查找），定位 JSON 文件（优先找名包含 'taobao' 和 '.json' 的文件）。
2) 读取该 JSON 为 items（list[dict]）。
3) 计算文件内容 SHA256（对原始文件 bytes 计算）。
4) 立刻打印一行审计信息（必须严格一行）：
   __AUDIT__={{"filename": "...", "sha256": "...", "total_rows": N}}

【B. 解析目标并计算答案（必须用代码算）】
5) 从 OBJECTIVE 解析约束（例如：year=2025, gender=女装, category=短裤, monthly_sales>2000, sale_price>100, 以及“销量最高/最低”、TopK=几款；若未写 TopK，默认 5）。
6) 用代码过滤 items，并计算 filtered_rows。
7) 根据目标排序（销量最高 => monthly_sales 降序；销量最低 => 升序），取 TopK。
8) 再打印第二行审计信息（必须严格一行）：
   __AUDIT2__={{"filtered_rows": M, "sort_key": "monthly_sales", "order": "desc/asc", "topk": K,
               "top_sales_list": [..], "top_item_ids": [..]}}

【C. 输出最终答案】
9) 以表格输出 TopK（列：item_id,title,year,season,category,gender,monthly_sales,sale_price,profit_margin,competitor_shops,shop_name,link）。
10) 再用 3~6 行中文总结（引用关键数值）。

注意：
- 最终答案必须与审计信息一致。
- 不允许凭空编造。若找不到文件或无法运行 python，输出 FAILED 并说明原因。
"""

    payload = {
        "model": model,
        "tools": [{
            "type": "code_interpreter",
            "container": {
                "type": "auto",
                "memory_limit": "1g",
                "file_ids": [file_id],  # ✅ 通过 container 挂载文件
            }
        }],
        "tool_choice": "required",  # ✅ 强制用 python tool
        "instructions": instructions,
        "input": user_input,
        # ✅ 关键：把 python 执行日志带回来，便于你验证是否真的读文件算过
        "include": ["code_interpreter_call.outputs"],  # 官方支持 :contentReference[oaicite:3]{index=3}
    }

    def _extract_output_text(resp_json: dict) -> str:
        # 兼容：优先 output_text；其次从 output items 里拼 message 文本
        t = (resp_json.get("output_text") or "").strip()
        if t:
            return t
        out_items = resp_json.get("output") or []
        texts = []
        for it in out_items:
            if isinstance(it, dict) and it.get("type") == "message":
                content = it.get("content") or []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        texts.append(c.get("text", ""))
        return ("\n".join(texts)).strip()

    def _extract_ci_logs(resp_json: dict) -> str:
        # 需要 include=["code_interpreter_call.outputs"] 才更稳拿到 logs :contentReference[oaicite:4]{index=4}
        out_items = resp_json.get("output") or []
        logs_chunks = []
        for it in out_items:
            if not isinstance(it, dict):
                continue
            if it.get("type") == "code_interpreter_call":
                outputs = it.get("outputs") or []
                for o in outputs:
                    if isinstance(o, dict) and o.get("type") == "logs":
                        logs_chunks.append(o.get("logs", ""))
        return ("\n".join(logs_chunks)).strip()

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=300)
        if r.status_code >= 400:
            return f"Error from /v1/responses: {r.status_code}\n{r.text}"

        data = r.json()

        answer_text = _extract_output_text(data)
        ci_logs = _extract_ci_logs(data)

        # 在 logs 里找审计行，作为“确实读文件”的证据
        audit_lines = []
        if ci_logs:
            for line in ci_logs.splitlines():
                if line.startswith("__AUDIT__=") or line.startswith("__AUDIT2__="):
                    audit_lines.append(line.strip())

        # 拼接“可验证证据块”
        proof = ""
        if audit_lines:
            proof = (
                    "\n\n===== FILE ANALYSIS AUDIT (proof of file read + computation) =====\n"
                    + "\n".join(audit_lines)
                    + "\n"
            )
        else:
            proof = (
                "\n\n===== FILE ANALYSIS AUDIT =====\n"
                "WARNING: No __AUDIT__ lines found in Code Interpreter logs.\n"
                "This likely means python tool did not run as required, or logs were not produced.\n"
            )

        # 可选：附上部分 logs，便于你排查（太长就截断）
        if ci_logs:
            max_chars = 4000
            tail = ci_logs[-max_chars:]
            proof += "\n----- Code Interpreter logs (tail) -----\n" + tail + "\n"

        return (answer_text.strip() + proof).strip()

    except Exception as e:
        return f"ERROR calling /v1/responses: {e}"


def openai_call(
        prompt: str,
        model: str = LLM_MODEL,
        temperature: float = OPENAI_TEMPERATURE,
        max_tokens: int = 100,
):
    while True:
        try:
            if model.lower().startswith("llama"):
                result = llm(prompt[:CTX_MAX],
                             stop=["### Human"],
                             echo=False,
                             temperature=0.2,
                             top_k=40,
                             top_p=0.95,
                             repeat_penalty=1.05,
                             max_tokens=200)
                # print('\n*****RESULT JSON DUMP*****\n')
                # print(json.dumps(result))
                # print('\n')
                return result['choices'][0]['text'].strip()
            elif model.lower().startswith("human"):
                return user_input_await(prompt)
            elif not model.lower().startswith("gpt-"):
                # Use completion API
                response = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return response.choices[0].text.strip()
            else:
                # Use 4000 instead of the real limit (4097) to give a bit of wiggle room for the encoding of roles.
                # TODO: different limits for different models.

                trimmed_prompt = limit_tokens_from_string(prompt, model, 4000 - max_tokens)

                # Use chat completion API
                messages = [{"role": "system", "content": trimmed_prompt}]
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=1,
                    stop=None,
                )
                return response.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            print(
                "   *** The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.Timeout:
            print(
                "   *** OpenAI API timeout occurred. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIError:
            print(
                "   *** OpenAI API error occurred. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIConnectionError:
            print(
                "   *** OpenAI API connection error occurred. Check your network settings, proxy configuration, SSL certificates, or firewall rules. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.InvalidRequestError:
            print(
                "   *** OpenAI API invalid request. Check the documentation for the specific API method you are calling and make sure you are sending valid and complete parameters. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.ServiceUnavailableError:
            print(
                "   *** OpenAI API service unavailable. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        else:
            break


def task_creation_agent(
        objective: str, result: Dict, task_description: str, task_list: List[str]
):
    prompt = f"""
    Never add a task that just repeats or paraphrases the objective.
    If the last result indicates candidate data has been saved (e.g., contains "saved_path", "candidates", "file_id", or tool == "search_taobao"), you MUST add exactly one next task that says: "读取并分析候选商品文件并按 OBJECTIVE 输出最终答案" (or equivalent). Do NOT add another search task unless the saved data is missing.
You are to use the result from an execution agent to create new tasks with the following objective: {objective}.
The last completed task has the result: \n{result["data"]}
This result was based on this task description: {task_description}.\n"""
    prompt += """
    Important planning rules:
    - If the last result looks like tool output from search_taobao (e.g., contains '\"tool\": \"search_taobao\"' and a 'saved_path' or 'file_id'),
      you MUST add exactly ONE next task that analyzes the candidates file and directly answers the OBJECTIVE (e.g., "读取并分析候选商品文件并按目标输出最终结果").
    - If the last result already contains the final answer (a ranked list that answers the objective), return: "There are no tasks to add at this time."
    """

    if task_list:
        prompt += f"These are incomplete tasks: {', '.join(task_list)}\n"
    prompt += "Based on the result, return a list of tasks to be completed in order to meet the objective. "
    if task_list:
        prompt += "These new tasks must not overlap with incomplete tasks. "

    prompt += """
Return one task per line in your response. The result must be a numbered list in the format:

#. First task
#. Second task

The number of each entry must be followed by a period. If your list is empty, write "There are no tasks to add at this time."
Unless your list is empty, do not include any headers before your numbered list or follow your numbered list with any other output."""

    print(f'\n*****TASK CREATION AGENT PROMPT****\n{prompt}\n')
    response = openai_call(prompt, max_tokens=2000)
    print(f'\n****TASK CREATION AGENT RESPONSE****\n{response}\n')
    new_tasks = response.split('\n')
    new_tasks_list = []
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = ''.join(s for s in task_parts[0] if s.isnumeric())
            task_name = re.sub(r'[^\w\s_]+', '', task_parts[1]).strip()
            if task_name.strip() and task_id.isnumeric():
                # 丢掉把 objective 当任务的情况
                if task_name.strip() == objective.strip():
                    continue
                # 去重（防止同一任务重复出现）
                if task_name not in new_tasks_list:
                    new_tasks_list.append(task_name)

            # print('New task created: ' + task_name)

    out = [{"task_name": task_name} for task_name in new_tasks_list]
    return out


def prioritization_agent():
    task_names = tasks_storage.get_task_names()
    bullet_string = '\n'

    prompt = f"""
You are tasked with prioritizing the following tasks: {bullet_string + bullet_string.join(task_names)}
Consider the ultimate objective of your team: {OBJECTIVE}.
Tasks should be sorted from highest to lowest priority, where higher-priority tasks are those that act as pre-requisites or are more essential for meeting the objective.
Do not remove any tasks. Return the ranked tasks as a numbered list in the format:

#. First task
#. Second task

The entries must be consecutively numbered, starting with 1. The number of each entry must be followed by a period.
Do not include any headers before your ranked list or follow your list with any other output."""

    print(f'\n****TASK PRIORITIZATION AGENT PROMPT****\n{prompt}\n')
    response = openai_call(prompt, max_tokens=2000)
    print(f'\n****TASK PRIORITIZATION AGENT RESPONSE****\n{response}\n')
    if not response:
        print('Received empty response from priotritization agent. Keeping task list unchanged.')
        return
    new_tasks = response.split("\n") if "\n" in response else [response]
    new_tasks_list = []
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = ''.join(s for s in task_parts[0] if s.isnumeric())
            task_name = re.sub(r'[^\w\s_]+', '', task_parts[1]).strip()
            if task_name.strip():
                new_tasks_list.append({"task_id": task_id, "task_name": task_name})

    return new_tasks_list


def is_analysis_task(task_text: str) -> bool:
    t = (task_text or "").lower()
    analysis_hints = [
        "读取", "分析", "按目标", "输出最终", "生成最终答案",
        "销量最高", "销量最低", "top", "bottom", "排序", "筛选", "统计"
    ]
    return any(k in t for k in analysis_hints)


def is_search_task(task_text: str) -> bool:
    t = (task_text or "").lower()
    search_hints = [
        "search_taobao", "淘宝搜索", "搜索", "检索", "生成关键词", "爬取", "抓取"
    ]
    # 只要像“读取并分析候选商品文件”这种分析任务，就绝不能走搜索
    if is_analysis_task(t):
        return False
    return any(k in t for k in search_hints)


# 替换之后的execution_agent
def execution_agent(objective: str, task: str) -> str:
    """
    Step2 version:
    - Let the LLM decide an appropriate keyword based on OBJECTIVE + TASK
    - Let the LLM trigger function calling to search_taobao
    - Execute the tool locally and return a compact JSON summary
    """
    global LAST_TAOBAO_FILE_ID, LAST_TAOBAO_CANDIDATES_PATH

    task_text = (task or "").strip()
    # ✅ 1) 只要是分析/读取/输出最终结果类任务，优先走 analyze_with_file
    if any(k in task_text for k in ["读取", "分析", "输出最终", "销量最高", "销量最低", "Top", "top", "排序", "筛选"]):
        if not LAST_TAOBAO_FILE_ID:
            return "ERROR: No cached file_id. Please run search_taobao first."
        return analyze_with_file(objective, task_text, LAST_TAOBAO_FILE_ID)
    # 0) 分析任务：直接读文件分析（不要再触发 search）
    if is_analysis_task(task_text):
        if not LAST_TAOBAO_FILE_ID:
            return "ERROR: 没有可分析的数据文件 file_id。请先执行一次 search_taobao 生成候选文件。"
        return analyze_with_file(objective=objective, task=task_text, file_id=LAST_TAOBAO_FILE_ID)

    context = context_agent(query=objective, top_results_num=5)

    # 给模型的“行为规范”：什么时候必须调用 search_taobao，keyword 怎么写
    system_msg = (
        "You are an LLM-powered Taobao product selection agent.\n"
        "You can call a tool named search_taobao to fetch candidate items.\n\n"
        "When the task is about searching Taobao / generating keywords / retrieving candidate products, "
        "you MUST call search_taobao.\n"
        "Decide the keyword by extracting constraints from OBJECTIVE and TASK, such as:\n"
        "- year (e.g., 2024)\n"
        "- season (春季/夏季/秋季/冬季)\n"
        "- gender (女装/男装)\n"
        "- category (e.g., 衬衫/连衣裙)\n"
        "Keep keyword short (3-8 Chinese words), e.g., '2024 秋季 女装 衬衫'.\n"
        "Do NOT fabricate product lists in plain text when you can call the tool."
    )

    user_msg = f"OBJECTIVE:\n{objective}\n\nTASK:\n{task}\n"
    if context:
        user_msg += "\nPREVIOUS COMPLETED TASK CONTEXT:\n" + "\n".join(context) + "\n"

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # 如果不是 GPT-chat 模型，直接走原 openai_call（functions 不可用）
    if not LLM_MODEL.lower().startswith("gpt-"):
        prompt = f'Perform one task based on the following objective: {objective}.\n'
        if context:
            prompt += 'Take into account these previously completed tasks:' + '\n'.join(context)
        prompt += f'\nYour task: {task}\nResponse:'
        return openai_call(prompt, max_tokens=2000)

    # ===== Step3: 如果已经有候选文件 file_id 且当前不是搜索任务，则直接读文件分析 =====
    if LAST_TAOBAO_FILE_ID and (not is_search_task(task)):
        return analyze_with_file(objective, task, LAST_TAOBAO_FILE_ID)

    msg = openai_call_with_functions(
        messages=messages,
        functions=[SEARCH_TAOBAO_FUNCTION],
        max_tokens=800
    )

    # 模型触发了工具调用
    fc = msg.get("function_call")
    if fc and fc.get("name") == "search_taobao":
        raw_args = fc.get("arguments", "{}")

        # 解析 arguments（模型有时会给非严格 JSON，做容错）
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except Exception:
            args = {"keyword": str(raw_args)}

        keyword = (args.get("keyword") or "").strip()
        n = int(args.get("n") or 300)
        save_path = (args.get("save_path") or "runs/taobao_candidates.json").strip()
        merge = bool(args.get("merge") or False)
        preview_k = int(args.get("preview_k") or 5)

        tool_result = search_taobao(
            keyword=keyword,
            n=n,
            save_path=save_path,
            merge=merge,
            preview_k=preview_k,
        )
        # ===== Step3: 立刻上传文件并缓存 file_id =====

        saved_path = tool_result.get("saved_path") or save_path

        prev_path = LAST_TAOBAO_CANDIDATES_PATH
        # 避免每次都重复上传：只有当 file_id 为空或路径变化时才上传
        if (LAST_TAOBAO_FILE_ID is None) or (prev_path != saved_path):
            LAST_TAOBAO_FILE_ID = upload_file_user_data(saved_path)

        # 上传后再更新全局路径
        LAST_TAOBAO_CANDIDATES_PATH = saved_path
        tool_result["file_id"] = LAST_TAOBAO_FILE_ID

        # 返回紧凑 JSON 摘要（包含 file_id，方便 task_creation_agent 规划下一步）
        return json.dumps(tool_result, ensure_ascii=False)

    # 模型没调用工具：就返回它的文本结果
    return (msg.get("content") or "").strip()


def context_agent(query: str, top_results_num: int):
    """
    Retrieves context for a given query from an index of tasks.

    Args:
        query (str): The query or objective for retrieving context.
        top_results_num (int): The number of top results to retrieve.

    Returns:
        list: A list of tasks as context for the given query, sorted by relevance.

    """
    results = results_storage.query(query=query, top_results_num=top_results_num)
    # print("****RESULTS****")
    # print(results)
    return results


# Add the initial task if starting new objective
if not JOIN_EXISTING_OBJECTIVE:
    initial_task = {
        "task_id": tasks_storage.next_task_id(),
        "task_name": INITIAL_TASK
    }
    tasks_storage.append(initial_task)


def main():
    loop = True
    max_iterations = int(os.getenv("MAX_ITERATIONS", "2"))
    iteration = 0
    while loop:
        iteration += 1
        if iteration > max_iterations:
            print("Reached MAX_ITERATIONS, stopping.")
            break

        # As long as there are tasks in the storage...
        if not tasks_storage.is_empty():
            # Print the task list
            print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
            for t in tasks_storage.get_task_names():
                print(" • " + str(t))

            # Step 1: Pull the first incomplete task
            task = tasks_storage.popleft()
            print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
            print(str(task["task_name"]))

            # Send to execution function to complete the task based on the context
            result = execution_agent(OBJECTIVE, str(task["task_name"]))
            print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
            print(result)

            # Step 2: Enrich result and store in the results storage
            # This is where you should enrich the result if needed
            enriched_result = {
                "data": result
            }
            # extract the actual result from the dictionary
            # since we don't do enrichment currently
            # vector = enriched_result["data"]

            result_id = f"result_{task['task_id']}"

            results_storage.add(task, result, result_id)

            # Step 3: Create new tasks and re-prioritize task list
            # only the main instance in cooperative mode does that
            new_tasks = task_creation_agent(
                OBJECTIVE,
                enriched_result,
                task["task_name"],
                tasks_storage.get_task_names(),
            )

            print('Adding new tasks to task_storage')
            for new_task in new_tasks:
                new_task.update({"task_id": tasks_storage.next_task_id()})
                print(str(new_task))
                tasks_storage.append(new_task)

            if not JOIN_EXISTING_OBJECTIVE:
                prioritized_tasks = prioritization_agent()
                if prioritized_tasks:
                    tasks_storage.replace(prioritized_tasks)

            # Sleep a bit before checking the task list again
            time.sleep(5)
        else:
            print('Done.')
            loop = False


if __name__ == "__main__":
    main()
