from __future__ import annotations

import gc
import hashlib
import json
import logging
import random
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from functools import lru_cache
from pprint import pformat
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import torch
from hydra import compose

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:
    pass  # transformers not installed; HFBackend will not work
import platform


def supports_bitsandbytes() -> bool:
    """Check if bitsandbytes 4-bit quantization is supported on this platform.

    BitsAndBytes requires Linux and CUDA to be available.

    Returns:
        True if bitsandbytes is supported, False otherwise.
    """
    if platform.system().lower() != "linux":
        return False
    if not torch.cuda.is_available():
        return False
    try:
        import bitsandbytes as bnb  # noqa: F401

        return True
    except Exception:
        return False


@lru_cache(maxsize=None)
def load_model_cfg(name: str) -> Any:
    """Load model configuration from Hydra config files.

    This function is cached to avoid reloading the same configuration multiple times.

    Args:
        name: Name of the model to load configuration for.

    Returns:
        Hydra configuration object for the specified model.
    """
    return compose(config_name=f"model/{name}")


log = logging.getLogger("Scheduler")

TOKENIZER_CACHE: Dict[tuple[str, str], AutoTokenizer] = {}
LEGAL_QUANTIZATIONS = ["none", "4bit"]


def _normalize_device_str(device: str | None) -> str:
    """Normalize user-specified device strings into PyTorch-compatible format.

    Examples:
        - "cuda", "cuda:0" -> "cuda:0" (if CUDA available)
        - "mps" -> "mps" (if MPS available on macOS)
        - anything else -> "cpu"

    Args:
        device: Device string to normalize, or None for CPU.

    Returns:
        Normalized device string compatible with PyTorch.
    """
    if device is None:
        return "cpu"

    dev = str(device).lower()

    # CUDA handling
    if "cuda" in dev and torch.cuda.is_available():
        # default to cuda:0 if no index given
        if ":" not in dev:
            return "cuda:0"
        return dev

    # MPS handling (mac)
    if (
        dev == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    # Fallback
    return "cpu"


def hash_dict(d: Dict) -> str:
    """Generate a stable MD5 hash for a dictionary.

    Keys are sorted to ensure consistent serialization across runs.

    Args:
        d: Dictionary to hash.

    Returns:
        MD5 hash as hexadecimal string.
    """
    encoded = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.md5(encoded).hexdigest()


class GenerationBackend(Protocol):
    """Protocol defining the interface for model generation backends.

    This protocol specifies the required methods and properties that any backend
    must implement to be used with the InferenceScheduler.
    """

    def __init__(self, model_name: str, device: str):
        """Initialize the backend.

        Args:
            model_name: Name of the model to load.
            device: Device to load the model on (e.g., 'cuda:0', 'cpu', 'mps').
        """
        self.model_name = model_name
        self.device = device
        self._last_latency = 0.0
        self._tokens = 0
        self._last_prompt: str = ""

    def embed(self, context) -> List[Any]:
        """Generate embeddings for the given context.

        Args:
            context: Context to embed (list of chat messages or string).

        Returns:
            List of embeddings (hidden states) from the model.

        Raises:
            NotImplementedError: Must be implemented by concrete backends.
        """
        raise NotImplementedError

    def unload(self) -> None:
        """Unload model from memory to free resources.

        Called when the backend is evicted from the LRU cache.
        """
        raise NotImplementedError

    @property
    def tokens_used(self) -> int:
        """Get the number of tokens used in the last operation.

        Returns:
            Number of tokens processed.
        """
        raise NotImplementedError

    @property
    def last_latency(self) -> float:
        """Get the latency of the last operation.

        Returns:
            Latency in seconds.
        """
        raise NotImplementedError

    @property
    def last_prompt(self) -> str:
        """Get the last prompt that was processed.

        Returns:
            The last prompt string.
        """
        return self._last_prompt


class DummyBackend(GenerationBackend):
    """Dummy backend for testing without real models.

    This backend provides deterministic fake responses for testing and development
    without requiring actual model loading. Useful for testing experiment logic.
    """

    def generate(self, prompt, seed: Optional[int] = None) -> str:
        """Generate a deterministic fake response.

        Args:
            prompt: The prompt (list of chat messages or string).
            seed: Random seed for deterministic output.

        Returns:
            A fake acknowledgment string with model name and random number.
        """
        t0 = time.time()
        # prompt may be a list of {"role","content"} dicts or a str
        if isinstance(prompt, list):
            user = next((m["content"] for m in prompt if m.get("role") == "user"), "")
            sys = next((m["content"] for m in prompt if m.get("role") == "system"), "")
            s = sys + "\n" + user
        else:
            s = str(prompt)
        # deterministic-ish short reply
        rnd = random.Random((seed or 0) ^ (hash(s) & 0xFFFFFFFF))
        reply = f"[{self.model_name}] ack:{rnd.randint(0, 9999)}"
        self._tokens = len(s.split()) + len(reply.split())
        self._last_latency = max(0.0, time.time() - t0)
        return reply

    def get_logits(self, context: List[Dict[str, str]]) -> np.ndarray:
        """Generate fake logits from hashed context.

        Args:
            context: List of chat message dictionaries.

        Returns:
            Fake logits array of shape (1, vocab_size).
        """
        # Create fake logits for a small vocabulary (1000 tokens)
        vocab_size = 1000
        hashed_dicts = tuple(hash_dict(d) for d in context)
        h = hashlib.sha256(str(hashed_dicts).encode()).digest()

        # Use hash to create deterministic logits
        rnd = random.Random(int.from_bytes(h[:8], "big"))
        logits = np.array(
            [[rnd.gauss(0, 1) for _ in range(vocab_size)]], dtype=np.float32
        )

        # Make tokens 1-6 have higher logits for testing
        for i in range(1, 7):
            if i < vocab_size:
                logits[0, i] += 5.0

        return logits

    def embed(self, context: List[Dict[str, str]]) -> List[float]:
        """Generate fake embeddings from hashed context.

        Args:
            context: List of chat message dictionaries.

        Returns:
            A 16-dimensional fake embedding vector.
        """
        hashed_dicts = tuple(hash_dict(d) for d in context)
        h = hashlib.sha256(str(hashed_dicts).encode()).digest()
        # 16-d float vector in [0,1)
        vec = [b / 255.0 for b in h[:16]]
        return vec

    def unload(self) -> None:
        """No-op unload for DummyBackend.

        No resources to free since this is a fake backend.
        """
        pass

    def acquire_device(self) -> None:
        """No-op device acquisition for DummyBackend."""
        pass

    def release_device(self) -> None:
        """No-op device release for DummyBackend."""
        pass

    @property
    def tokens_used(self) -> int:
        """Get the number of tokens used in the last operation.

        Returns:
            Number of tokens processed.
        """
        return self._tokens

    @property
    def last_latency(self) -> float:
        """Get the latency of the last operation.

        Returns:
            Latency in seconds.
        """
        return self._last_latency


class HFBackend(GenerationBackend):
    """HuggingFace transformer backend for real model inference.

    Supports multiple configurations:
        - CUDA fp16: Models loaded on CPU (warm pool), moved to CUDA during inference
        - MPS/CPU fp16: Models loaded on CPU, moved to MPS/CPU during inference

    The backend manages model lifecycle including loading, device placement,
    and memory cleanup.
    """

    def __init__(self, model_name: str, device: str):
        """Initialize HuggingFace backend.

        Args:
            model_name: Name of the model to load.
            device: Target device for inference ('cuda:0', 'cpu', 'mps').
                    For fp16 models, they are loaded on CPU and moved to device during inference.

        Raises:
            AssertionError: If quantization mode is not recognized.
        """
        self.model_name = model_name
        self.device = _normalize_device_str(device)

        self._model = None
        self._tokenizer = None
        self._loaded = False

        self._last_latency = 0.0
        self._tokens = 0
        self._last_prompt: str = ""

        self._device_lock = threading.Lock()
        self._last_used_ts: float = 0.0

    def _load_tokenizer(
        self, hf_path: str, token: Optional[str] = None
    ) -> AutoTokenizer:
        """Load tokenizer with global caching.

        Tokenizers are cached globally to avoid reloading the same tokenizer
        multiple times across different backend instances.

        Args:
            hf_path: HuggingFace model path.
            token: Optional HuggingFace access token for private models.

        Returns:
            Loaded AutoTokenizer instance.
        """
        key = (hf_path, token or "")
        if key in TOKENIZER_CACHE:
            return TOKENIZER_CACHE[key]

        tokenizer = AutoTokenizer.from_pretrained(hf_path, token=token or "")
        # reasonable default: pad with EOS if pad_token missing
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        TOKENIZER_CACHE[key] = tokenizer
        return tokenizer

    def load(self) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model and tokenizer.

        For fp16 models, loads to CPU (warm pool) and moves to target device during inference.

        This is typically called once per model_name per process, thanks to LRU caching.

        Returns:
            Tuple of (model, tokenizer).
        """
        if self._loaded:
            return self._model, self._tokenizer

        cfg = load_model_cfg(self.model_name)
        model_cfg = cfg.get("model")
        hf_path = model_cfg.get("hf_path")
        token = model_cfg.get("access", {}).get("token", "")

        # --- Load tokenizer (cached globally) ---
        self._tokenizer = self._load_tokenizer(hf_path, token)

        t0 = time.time()
        # choose dtype: prefer bf16 where supported, else fp16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

        use_4bit = supports_bitsandbytes()

        if use_4bit:
            # 4-bit quantized model, lives on GPU; no CPU warm pool here
            log.info(
                f"Loading HF model '{self.model_name}' in 4-bit on device '{self.device}'"
            )

            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            # Let HF/accelerate place it on the GPU
            device_map = "auto" if self.device.startswith("cuda") else self.device

            self._model = AutoModelForCausalLM.from_pretrained(
                hf_path,
                token=token,
                quantization_config=quant_cfg,
                device_map=device_map,
            )
            self._quantized = True
            log.info(
                f"Loaded 4-bit model '{self.model_name}' in {time.time() - t0:.2f}s "
                f"on device_map={device_map}"
            )
        else:

            log.info(
                f"Loading HF model '{self.model_name}' on CPU (warm pool); "
                f"target device during inference: {self.device}"
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                hf_path,
                token=token,
                torch_dtype=dtype,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
            self._quantized = False
            log.info(
                f"Loaded model '{self.model_name}' on CPU in {time.time() - t0:.2f}s"
            )

        self._loaded = True

        return self._model, self._tokenizer

    def embed(self, context: List[Dict[str, str]]) -> List:
        """Generate embeddings from the model's hidden states.

        Ensures model is loaded, tokenizes the input, runs inference,
        and extracts hidden states from all layers.

        Args:
            context: List of chat messages with 'role' and 'content' keys,
                    or a string prompt.

        Returns:
            List of hidden state tensors (as numpy arrays) from each layer.
        """
        if not self._loaded:
            self.load()

        self._last_prompt: str = self._tokenizer.apply_chat_template(
            context, tokenize=False
        )
        model_device = next(self._model.parameters()).device

        tokens = self.tokenize(context).to(model_device)

        # Ensure all transfers are done
        if model_device.type == "cuda":
            torch.cuda.synchronize(model_device.index)

        t0 = time.time()
        with torch.inference_mode():
            out = self._model(tokens, output_hidden_states=True, return_dict=True)
        self._last_latency = time.time() - t0

        return [h.detach().cpu().to(torch.float32).numpy() for h in out.hidden_states]

    def get_logits(self, context: List[Dict[str, str]]) -> torch.Tensor:
        """Generate logits from the model's output layer.

        Ensures model is loaded, tokenizes the input, runs inference,
        and extracts logits from the final layer.

        Args:
            context: List of chat messages with 'role' and 'content' keys,
                    or a string prompt.

        Returns:
            Logits tensor of shape (batch_size, sequence_length, vocab_size).
        """
        if not self._loaded:
            self.load()

        self._last_prompt = self._tokenizer.apply_chat_template(
            context, tokenize=False, continue_final_message=True
        )
        log.debug(f"Prompt:{self._last_prompt}")
        model_device = next(self._model.parameters()).device
        tokenized = self.tokenize(context, continue_final_message=True).to(model_device)
        log.debug("Tokenized ends with:", tokenized[0, -5:])

        with torch.no_grad():
            outputs = self._model(tokenized, return_dict=True)
            logits = outputs.logits

        # Return logits for the last token
        return logits[:, -1, :].detach().cpu()

    def tokenize(
        self, chat: List[Dict[str, str]], continue_final_message: bool = False
    ) -> torch.Tensor:
        """Tokenize chat messages using the model's tokenizer.

        Args:
            chat: List of dictionaries with 'role' and 'content' keys.
            continue_final_message: Whether to continue the final message.

        Returns:
            Tokenized input as a PyTorch tensor.

        Raises:
            AssertionError: If tokenizer is not loaded.
        """
        assert self._tokenizer is not None, "Tokenizer not loaded."
        tokens = self._tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=False,
            continue_final_message=continue_final_message,
        )
        return tokens

    def acquire_device(self) -> None:
        """Move model to target device for inference.

        For fp16 models: moves from CPU to self.device (cuda/mps/cpu).

        This is called by the scheduler when the backend becomes active.
        """
        if not self._loaded:
            self.load()

        model_device = next(self._model.parameters()).device
        if str(model_device) == self.device:
            # Already on correct device -> skip
            return
        t0 = time.time()

        with self._device_lock:
            if self._model is None:
                raise RuntimeError(
                    "HFBackend.acquire_device called with no model loaded"
                )

            target = self.device

            if target.startswith("cuda") and torch.cuda.is_available():
                log.debug(f"[{self.model_name}] moving model to {target}")
                self._model.to(target, non_blocking=True)
                torch.cuda.synchronize()
            elif (
                target == "mps"
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                log.debug(f"[{self.model_name}] moving model to mps")
                self._model.to("mps")
                # no direct mps.synchronize; operations are async but that's usually fine
            else:
                # already on CPU or fallback to CPU
                log.debug(f"[{self.model_name}] using CPU device")
                self._model.to("cpu")
        log.info(
            f"[{self.model_name}] acquire_device completed in {time.time() - t0:.3f}s "
            f"(device={next(self._model.parameters()).device})"
        )

    def release_device(self) -> None:
        """Move model back to CPU to free GPU/MPS memory.

        For fp16 models: moves back to CPU to free GPU/MPS memory.

        Called by the scheduler when inference is complete.
        """
        if not self._loaded or self._model is None:
            return

        with self._device_lock:
            log.debug(f"[{self.model_name}] moving model back to CPU")
            try:
                self._model.to("cpu")
            except Exception:
                pass

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    def unload(self) -> None:
        """Unload model from memory and free all resources.

        Called by LRU cache eviction. Frees GPU VRAM in both fp16 and 4-bit cases.
        Performs garbage collection and empties GPU caches.
        """
        if not self._loaded:
            return

        log.info(f"Unloading model '{self.model_name}' from memory")

        try:
            if self._model is not None and hasattr(self._model, "to"):
                try:
                    self._model.to("cpu")
                except Exception:
                    pass
        except Exception:
            pass

        try:
            del self._model
        except Exception:
            pass

        self._model = None
        self._tokenizer = None
        self._loaded = False

        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    @property
    def tokens_used(self) -> int:
        """Get the number of tokens used in the last operation.

        Returns:
            Number of tokens processed.
        """
        return self._tokens

    @property
    def last_latency(self) -> float:
        """Get the latency of the last operation.

        Returns:
            Latency in seconds.
        """
        return self._last_latency

    @property
    def last_prompt(self) -> str:
        """Get the last prompt that was processed.

        Returns:
            The last prompt string.
        """
        return self._last_prompt


class LRU:
    """Least Recently Used (LRU) cache for GenerationBackend instances.

    Manages a fixed-capacity cache of model backends. When capacity is reached,
    the least recently used backend is evicted and properly cleaned up.
    Thread-safe for concurrent access.

    Attributes:
        capacity: Maximum number of backends to cache.
        store: Ordered dictionary storing backends.
        lock: Threading lock for thread-safe operations.
    """

    def __init__(self, capacity: int):
        """Initialize LRU cache.

        Args:
            capacity: Maximum number of backends to cache.
        """
        self.capacity = capacity
        self.store: OrderedDict[str, GenerationBackend] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[GenerationBackend]:
        """Retrieve a backend from cache and mark it as recently used.

        Args:
            key: Model name to retrieve.

        Returns:
            The cached backend if found, None otherwise.
        """
        with self.lock:
            if key in self.store:
                self.store.move_to_end(key)
                return self.store[key]
            return None

    def put(self, key: str, value: GenerationBackend) -> None:
        """Add or update a backend in the cache.

        If at capacity, evicts the least recently used backend first.

        Args:
            key: Model name.
            value: Backend instance to cache.
        """
        with self.lock:
            # If we're at capacity, evict LRU
            if key not in self.store and len(self.store) >= self.capacity:
                evict_key, evicted = self.store.popitem(last=False)
                if hasattr(evicted, "unload"):
                    evicted.unload()
                log.info(f"Evicted model {evict_key} from cache")

            self.store[key] = value
            self.store.move_to_end(key)


class InferenceScheduler:
    """Manages model loading, caching, and inference across multiple devices.

    The scheduler uses an LRU cache to manage a pool of loaded models,
    automatically evicting least-recently-used models when capacity is reached.
    Supports multi-GPU round-robin device selection and both HuggingFace and
    dummy backends.

    Attributes:
        cfg: Configuration dictionary.
        seed: Random seed for reproducibility.
        pool_capacity: Maximum number of models to keep in memory.
        devices: List of available devices for inference.
        model_pool: LRU cache of loaded model backends.
        device_index: Current device index for round-robin selection.
        device_lock: Threading lock for device selection.
    """

    def __init__(self, cfg):
        """Initialize the inference scheduler.

        Args:
            cfg: Configuration dictionary containing:
                - seed: Random seed
                - model_pool_capacity: Max models to cache (default: 1)
                - device_pool: List of devices (default: ['cpu'])
                - backend: Backend type ('hf' or 'dummy')
        """
        self.cfg = cfg
        self.seed: int = cfg.get("seed")
        self.pool_capacity = int(cfg.get("model_pool_capacity", 1))
        self.devices = list(cfg.get("device_pool", ["cpu"]))
        self.model_pool = LRU(capacity=self.pool_capacity)
        self.device_index = 0
        self.device_lock = threading.Lock()

    def embed(
        self,
        agent,
        t: int,
        neighbor_view: Optional[Dict[int, float]] = None,
        verbose: bool = False,
    ) -> List:
        """Generate embeddings for an agent's current context.

        Prepares the agent for the current round, constructs the message,
        and generates embeddings using the appropriate model backend.

        Args:
            agent: Agent instance to generate embeddings for.
            t: Current round number.
            neighbor_view: Dictionary mapping neighbor IDs to their belief values.
            verbose: If True, logs the message being embedded.

        Returns:
            List of hidden state activations from all model layers.
        """
        agent.prepare_round(t, neighbor_view or {})
        text_to_embed: List[Dict[str, str]] = agent.current_message

        if verbose:
            log.debug(
                "Message for agent %s at round %s:\n%s",
                agent.id,
                t,
                pformat(text_to_embed, indent=2),
            )

        with self.ensure_loaded(agent.model_name) as mdl:
            return mdl.embed(text_to_embed)

    def get_logits(
        self,
        agent,
        t: int,
        neighbor_view: Optional[Dict[int, float]] = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Generate logits for an agent's current context.

        Prepares the agent for the current round, constructs the message,
        and generates logits using the appropriate model backend.

        Args:
            agent: Agent instance to generate logits for.
            t: Current round number.
            neighbor_view: Dictionary mapping neighbor IDs to their belief values.
            verbose: If True, logs the message being processed.

        Returns:
            Logits tensor from the model's output layer.
        """
        agent.prepare_round(t, neighbor_view or {})
        text_to_embed: List[Dict[str, str]] = agent.current_message

        if verbose:
            log.debug(
                "Message for agent %s at round %s:\n%s",
                agent.id,
                t,
                pformat(text_to_embed, indent=2),
            )

        with self.ensure_loaded(agent.model_name) as mdl:
            if hasattr(mdl, "get_logits"):
                return mdl.get_logits(text_to_embed)
            else:
                raise NotImplementedError(
                    f"Backend {type(mdl).__name__} does not support logit extraction"
                )

    @contextmanager
    def ensure_loaded(self, model_name: str):
        """Context manager ensuring model backend is loaded and on correct device.

        Retrieves backend from cache or loads it if not cached. Acquires device
        before yielding and releases it afterward.

        Args:
            model_name: Name of the model to ensure is loaded.

        Yields:
            GenerationBackend instance ready for inference.
        """
        backend = self.model_pool.get(model_name)
        if backend is None:
            device = self.pick_device()
            backend = self.load_model(model_name, device)
            self.model_pool.put(model_name, backend)

        if hasattr(backend, "acquire_device"):
            backend.acquire_device()
            log.debug(f"Acquired device for model {model_name}")

        try:
            yield backend
        finally:
            if hasattr(backend, "release_device"):
                backend.release_device()

    def pick_device(self) -> str:
        """Select next device using round-robin strategy.

        Thread-safe device selection across the configured device pool.

        Returns:
            Device string (e.g., 'cuda:0', 'cpu', 'mps').
        """
        with self.device_lock:
            device = self.devices[self.device_index]
            self.device_index = (self.device_index + 1) % len(self.devices)
            return device

    def load_model(self, model_name: str, device: str) -> GenerationBackend:
        """Load a model backend based on configuration.

        Creates either a HuggingFace backend for real models or a dummy backend
        for testing, based on the configured backend type.

        Args:
            model_name: Name of the model to load.
            device: Device to load the model on.

        Returns:
            Initialized GenerationBackend instance.

        Raises:
            NotImplementedError: If backend type is not recognized.
        """
        backend = (self.cfg.get("backend") or "hf").lower()

        if backend in ("dummy", "dry", "test") or model_name == "expert":
            return DummyBackend(model_name, device)
        elif backend == "hf":
            return HFBackend(model_name, device)

        raise NotImplementedError(f'Unknown backend "{backend}"')
