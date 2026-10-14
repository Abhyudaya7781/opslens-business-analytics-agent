from __future__ import annotations

import json
import os
from typing import Any, Callable

import pandas as pd
from groq import Groq

from opslens.analytics import compare_metric, delivery_risk, metric_trend, profile_data


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "profile_dataset",
            "description": "Summarize validated dataset coverage, quality, revenue, profit, and delivery metrics.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_metric",
            "description": "Compare a supported metric across region, category, product, or status. Use for rankings and group comparisons.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": ["revenue", "profit", "units", "orders", "avg_delivery_days", "cancellation_rate", "profit_margin"]},
                    "group_by": {"type": "string", "enum": ["region", "category", "product", "status"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["metric", "group_by"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "metric_trend",
            "description": "Calculate monthly or weekly trend and percentage change for a supported metric.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": ["revenue", "profit", "units", "orders", "avg_delivery_days", "cancellation_rate", "profit_margin"]},
                    "granularity": {"type": "string", "enum": ["month", "week"]},
                },
                "required": ["metric"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delivery_risk",
            "description": "Rank regions by cancellation rate and average delivery time to identify operational areas for investigation.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 20}},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]


class AgentError(RuntimeError):
    pass


def _tool_functions(frame: pd.DataFrame) -> dict[str, Callable[..., dict[str, Any]]]:
    return {
        "profile_dataset": lambda: profile_data(frame),
        "compare_metric": lambda metric, group_by, limit=10: compare_metric(frame, metric, group_by, limit),
        "metric_trend": lambda metric, granularity="month": metric_trend(frame, metric, granularity),
        "delivery_risk": lambda limit=10: delivery_risk(frame, limit),
    }


def answer_question(frame: pd.DataFrame, question: str) -> dict[str, Any]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise AgentError("GROQ_API_KEY is not set. Add it to your .env file to enable natural-language analysis.")
    if not question.strip():
        raise AgentError("Enter a question about the dataset.")

    client = Groq(api_key=api_key)
    functions = _tool_functions(frame)
    messages = [
        {
            "role": "system",
            "content": (
                "You are OpsLens, an operations analytics assistant. Use the provided tools to calculate evidence before making data claims. "
                "Never invent figures, columns, causes, or business context. If the available tools cannot answer a question, explain the limitation. "
                "Separate measured results from possible explanations. Cite the tool name and the relevant values in concise prose. "
                "Do not request or execute code. You may make at most four tool calls in this turn."
            ),
        },
        {"role": "user", "content": question.strip()[:2000]},
    ]
    evidence = []

    for _ in range(4):
        try:
            response = client.chat.completions.create(
                model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0,
            )
        except Exception as exc:
            raise AgentError(f"The language model request failed: {exc.__class__.__name__}. Check your API key, model name, and connection.") from exc

        message = response.choices[0].message
        if not message.tool_calls:
            return {"answer": message.content or "The model returned no answer.", "evidence": evidence}

        messages.append(message)
        for call in message.tool_calls:
            name = call.function.name
            if name not in functions:
                raise AgentError(f"The model requested an unsupported tool: {name}.")
            try:
                arguments = json.loads(call.function.arguments or "{}")
                result = functions[name](**arguments)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                result = {"error": str(exc)}
            evidence.append({"tool": name, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=str)[:12000]})

    messages.append({"role": "user", "content": "Summarize only the evidence collected so far. Do not make additional tool calls."})
    try:
        final = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=messages,
            temperature=0,
        )
    except Exception as exc:
        raise AgentError(f"The final response failed: {exc.__class__.__name__}.") from exc
    return {"answer": final.choices[0].message.content or "No summary was returned.", "evidence": evidence}
