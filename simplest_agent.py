"""
Simple Quote Agent — core agent definition and tool functions.

Defines a pydantic-ai Agent that responds to questions with random philosophy
or science quotes. Reads all configuration (model provider, base_url, system
prompt, retries) from ai_system_config.yaml. When the model provider is "vllm"
(i.e. a Domino-hosted LLM), the agent fetches a short-lived access token from
the Domino runtime for authentication.

This file is imported by chat_app.py (production) and dev_eval_simplest_agent.py
(batch evaluation). Call create_agent() to get a fresh agent instance.
"""

import asyncio
import os
import yaml
import string
import requests
import random
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from typing import Dict, Any, Annotated
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIModel, ModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from domino.agents.tracing import add_tracing, search_traces
from domino.agents.logging import DominoRun,log_evaluation


script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'ai_system_config.yaml')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

oai_model = config['model']['full_name']
retries = config['agent']['retries']
system_prompt = config['prompts']['simple_agent_system']
BASE_URL = config['model'].get('base_url', '')



# Tool functions defined outside the agent so they can be registered on new agents
def science_quote(ctx: RunContext[str], question: str) -> str:
    """
    Use this function to answer any question with a random science quote.
    
    Args:
        ctx: The context of the run.
        question: The question
    
    Returns:
        A science quote
    """
    print("************** CALLED TOOL: science_quote")
    quotes = [
        "Imagination is more important than knowledge. — Albert Einstein",
        "If I have seen further it is by standing on the shoulders of giants. — Isaac Newton",
        "Research is what I'm doing when I don't know what I'm doing. — Wernher von Braun",
        "The important thing is to never stop questioning. — Albert Einstein",
        "Somewhere, something incredible is waiting to be known. — Carl Sagan",
        "Nothing in life is to be feared, it is only to be understood. — Marie Curie",
        "The first principle is that you must not fool yourself—and you are the easiest person to fool. — Richard Feynman",
        "However difficult life may seem, there is always something you can do and succeed at. — Stephen Hawking",
        "Science is a way of thinking much more than it is a body of knowledge. — Carl Sagan",
        "Equipped with his five senses, man explores the universe around him and calls the adventure Science. — Edwin Hubble"
    ]
    
    return random.choice(quotes)


def philosophy_quote(ctx: RunContext[str], question: str) -> str:
    """
    Use this function to answer any question with a random philosophy quote.
    
    Args:
        ctx: The context of the run.
        question: The question
    
    Returns:
        A philosophy quote
    """
    print("************** CALLED TOOL: philosophy_quote")
    quotes = [
        "The unexamined life is not worth living. — Socrates",
        "He who thinks great thoughts, often makes great errors. — Martin Heidegger",
        "Happiness is not an ideal of reason but of imagination. — Immanuel Kant",
        "We are what we repeatedly do. Excellence, then, is not an act, but a habit. — Aristotle",
        "Man is condemned to be free; because once thrown into the world, he is responsible for everything he does. — Jean-Paul Sartre"
    ]
    return random.choice(quotes)


def create_agent():
    """
    Factory function to create a fresh agent with a new API key.
    Call this before each chat request since the VLLM_API_KEY expires every 5 minutes.
    """
    VLLM_API_KEY = requests.get("http://localhost:8899/access-token").text
    
    provider = OpenAIProvider(
        base_url=BASE_URL,
        api_key=VLLM_API_KEY,
    )
    
    vllm_model = OpenAIModel("", provider=provider)  # have to leave model name blank?
    
    selected_model = oai_model
    if config['model']['provider'] == 'vllm':
        selected_model = vllm_model

    the_agent = Agent(
        selected_model,
        retries=retries,
        system_prompt=system_prompt,
        instrument=True
    )
    
    # Register tools on the fresh agent
    the_agent.tool(science_quote)
    the_agent.tool(philosophy_quote)
    
    return the_agent


# Create a default agent for backwards compatibility (e.g., direct script usage)
simplest_agent = create_agent()

#user_query = "what is the limit for 401k contribution 2023?"

