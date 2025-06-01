# %%
from dotenv import load_dotenv

load_dotenv()
import aiohttp
import openai
import os
from pathlib import Path
import re
import sqlalchemy as sa
import yaml

if '__file__' in globals():
    HERE = Path(__file__).resolve().parent  # running from a file
else:
    HERE = Path.cwd()  # running in a console/notebook
_RULES_PATH = HERE.with_name("canon_rules.yaml")


def load_rules() -> dict:
    """
    Return the YAML canon-rules as a Python dict.
    The path is resolved *relative to this file* so
    the import still works when CanonFodder is installed as a package.
    """
    if not _RULES_PATH.exists():
        raise FileNotFoundError(f"Expected canon rules at {_RULES_PATH}")
    with _RULES_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class CanonFodderLLM:
    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.AsyncOpenAI(api_key=openai_api_key)
        self.model = model

    async def country_from_context(self, artist_name: str) -> dict:
        """
        Return {
            "artist_name": "<unchanged input>",
            "country_iso2": "HU" | None,
            "confidence": 0.00-1.00
        }
        using (1) MusicBrainz, then (2) an LLM fallback.
        """
        result = {
            "artist_name": artist_name,
            "country_iso2": None,
            "confidence": 0.0,
        }
        # 1️⃣  MusicBrainz first
        mb = await self.mb_search(f'artist:"{artist_name}"')
        if mb and mb.get("country"):
            result["country_iso2"] = mb["country"]
            result["confidence"] = 0.99
            return result
        # 2️⃣  LLM fallback
        prompt = (
            "Give the most probable ISO-2 country code for the artist below "
            "and a confidence 0–1; respond as JSON {country_iso2, confidence}.\n\n"
            f"Artist name: {artist_name}"
        )
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        # pull out the first {...} block in the reply
        m = re.search(r"\{.*?}", resp.choices[0].message.content, flags=re.S)
        if m:
            try:
                parsed = yaml.safe_load(m.group(0))
                # guard against partial / malformed replies
                result["country_iso2"] = parsed.get("country_iso2")
                result["confidence"] = float(parsed.get("confidence", 0))
            except Exception:
                pass  # leave defaults (None, 0.0)
        return result

    async def decide_merge(self, variants: list[str]) -> dict:
        system = ("You are an expert music data curator with a vast knowledge on artists, bands, composers and alike."
                  "Pay attention to detail, follow canonization rules exactly, especially when it comes to punctuation.")
        user = f"Rules:\n{yaml.dump(load_rules())}\nVariants:\n" + "\n".join(variants)
        tools = [
            {  # MusicBrainz search primitive
                "type": "function",
                "function": {
                    "name": "mb_search",
                    "description": "Search MusicBrainz artist index",
                    "parameters": {"query": {"type": "string"}}
                }
            }
        ]
        # openai spec-style function calling
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            tools=tools,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}]
        )
        return resp.choices[0].message  # json w/ decision, canonical, reasons

    @staticmethod
    async def mb_search(query: str) -> dict | None:
        """Async wrapper around the MusicBrainz `/ws/2/artist` endpoint."""
        url = "https://musicbrainz.org/ws/2/artist/"
        params = {"query": query, "fmt": "json", "limit": 1}
        async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)) as sess:
            async with sess.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return data["artists"][0] if data["artists"] else None
