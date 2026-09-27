"""The providers package: one class per backend, all behind the same Provider seam.

Three backends, three files, one interface:

    OllamaProvider            a local model through the Ollama daemon (the benchmark's home)
    OpenAICompatibleProvider  a local model through vLLM / llama-server / LM Studio, or OpenAI
    AnthropicProvider         Claude, through the official SDK

`PROVIDERS` maps the short names the CLI's --provider flag uses to the classes, so
picking a backend by name is a dict lookup rather than an if/elif chain that every
new backend would have to be threaded into. A fourth provider is one file plus one
line here; nothing else in cornac changes — that is the whole point of the seam.

The imports here are cheap: the Anthropic SDK is imported lazily inside its provider's
__init__, so importing this package never requires it to be installed.
"""

from cornac.providers.anthropic import AnthropicProvider
from cornac.providers.base import Provider
from cornac.providers.ollama import OllamaProvider
from cornac.providers.openai_compat import OpenAICompatibleProvider

PROVIDERS: dict[str, type[Provider]] = {
    "ollama": OllamaProvider,
    "openai": OpenAICompatibleProvider,
    "anthropic": AnthropicProvider,
}

__all__ = [
    "Provider",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "AnthropicProvider",
    "PROVIDERS",
]
