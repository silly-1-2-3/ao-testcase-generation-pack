import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List

# ---------------------- README: local config you may update ---------------------- #
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "qwen2.5-local"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_QUESTION = "你好，请介绍一下你自己。"

DEFAULT_DEBUG_ARGS = [
    "--base-url", DEFAULT_BASE_URL,
    "--model", DEFAULT_MODEL,
    "--interactive",
    "--question", DEFAULT_QUESTION,
    "--system-prompt", DEFAULT_SYSTEM_PROMPT,
]
# -------------------------------------------------------------------------------- #


# Inject default args when no CLI args are provided.
def inject_default_debug_args_if_needed() -> None:
    if len(sys.argv) == 1:
        sys.argv.extend(DEFAULT_DEBUG_ARGS)
        print("[INFO] No CLI args provided, fallback to DEFAULT_DEBUG_ARGS")
        print(f"[INFO] Injected args: {' '.join(DEFAULT_DEBUG_ARGS)}")


# Parse CLI args for single-turn or interactive Q&A.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a deployed local vLLM model via OpenAI-compatible API.")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="vLLM API base URL, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="served model name in vLLM")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="Any non-empty value is accepted by local vLLM by default")
    parser.add_argument("--question", type=str, default=None, help="Single-turn user question")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--interactive", action="store_true", help="Run interactive chat mode")
    return parser.parse_args()


# Build message list in OpenAI chat format.
def build_messages(system_prompt: str, question: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    return messages


# Send one chat completion request to local vLLM server.
def ask_once(args: argparse.Namespace, messages: List[Dict[str, str]]) -> str:
    url = args.base_url.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
        raise RuntimeError(f"HTTPError {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"URLError: {e}") from e

    result = json.loads(body)
    choices = result.get("choices", [])
    if not choices:
        raise RuntimeError(f"No choices in response: {result}")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    return content.strip()


# Run terminal interactive loop against deployed model.
def run_interactive(args: argparse.Namespace) -> None:
    print("Interactive mode started. Type /exit to quit.")
    while True:
        question = input("User> ").strip()
        if not question:
            continue
        if question.lower() == "/exit":
            break

        messages = build_messages(args.system_prompt, question)
        answer = ask_once(args, messages)
        print(f"Assistant> {answer}\n")


# Program entry point.
def main() -> None:
    args = parse_args()

    if args.interactive:
        run_interactive(args)
        return

    if not args.question:
        raise ValueError("Please provide --question or use --interactive.")

    messages = build_messages(args.system_prompt, args.question)
    answer = ask_once(args, messages)
    print(answer)


if __name__ == "__main__":
    inject_default_debug_args_if_needed()
    main()

