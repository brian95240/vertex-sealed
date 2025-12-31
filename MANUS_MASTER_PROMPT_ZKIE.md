# MANUS MASTER PROMPT: ZENITH KERNEL INFERENCE ENGINE (ZKIE)
## Version: 1.0 | Date: 2025-12-31 | Vertex Engineering Standard: 0.01%

---

## 🎯 PROJECT OVERVIEW

**Mission:** Build the world's first privacy-first, universally adaptive AI inference orchestrator that runs on any hardware from Celeron to datacenter clusters, with native MCP support, 15% cost-optimization threshold, and hermetic offline mode.

**Name:** Zenith Kernel Inference Engine (ZKIE)  
**Architecture:** Modular plugin system with vertex-sealed core  
**Differentiation:** Only local inference tool with native MCP + API cost routing + offline air-gap  

---

## 📊 CREDIT BUDGET & OPTIMIZATION

**Total Estimated Credits:** 2,500-3,500  
**Stages:** 7 major stages (3 parallel tracks in stages 2-5)  
**Optimization Strategy:**
- Batch file creation (5-10 files per stage)
- Parallel execution where dependencies allow
- Template-driven code generation
- Incremental testing with early failure detection

**Credit Breakdown:**
- Stage 1 (Bug Fixes): 200 credits
- Stage 2-3 (Core + Plugins): 800 credits (parallel)
- Stage 4-5 (UI + Features): 600 credits (parallel)
- Stage 6 (Integration + Testing): 400 credits
- Stage 7 (Docs + Package): 300 credits
- Buffer (15%): 500 credits

---

## 🔄 EXECUTION WORKFLOW

```
STAGE 1: Fix Critical Bugs (Serial)
    ├─ Fix vertex_core.py (14 bugs)
    ├─ Fix vertex_trinity.py (18 bugs)
    └─ Fix vertex_hyper.py (15 bugs)
         │
         ├──────────────────┬───────────────────┐
         ▼                  ▼                   ▼
    STAGE 2:           STAGE 3:            STAGE 4:
    Core Engine        Plugin System       UI Layer
    (Parallel)         (Parallel)          (Parallel)
         │                  │                   │
         └──────────────────┴───────────────────┘
                            ▼
                     STAGE 5: Privacy & Offline
                            ▼
                     STAGE 6: Integration + Testing
                            ▼
                     STAGE 7: Package + Documentation
```

---

## 🚀 STAGE 1: CRITICAL BUG FIXES (Serial — 200 Credits)

### Objective
Fix all 47 critical bugs in the 3 existing vertex files to create a stable foundation.

### Files to Fix
1. `vertex_core_FIXED.py` → `zkie/core/kernel.py`
2. `vertex_trinity_FIXED.py` → `zkie/core/trinity.py`
3. `vertex_hyper_FIXED.py` → `zkie/core/hyper.py`

### Task 1.1: Fix vertex_core (14 bugs)

**Critical Fixes:**
1. **GPU Benchmark Async Race** (Line 164-173)
   - Add `torch.cuda.synchronize()` before and after benchmark loop
   - Prevents measuring queue time instead of execution time
   
2. **CPU Features Reparsed** (Line 203-207)
   - Add `@functools.cache` decorator to `_get_cpu_features()`
   - Cache /proc/cpuinfo parse (5ms → 0ms after first call)

3. **Non-Atomic File Writes** (Line 243)
   - Use `tempfile.NamedTemporaryFile` + `os.replace()`
   - Prevents corruption on crash during write

4. **Delta Check Timeout** (Line 119)
   - Reduce timeout from 5s → 1s
   - Add silent failure on timeout (don't block startup)

5. **Disk Speed Detection** (Line 92-94)
   - Read `/sys/block/*/queue/max_sectors_kb` for actual bandwidth
   - Distinguish virtio-blk (900 MB/s) vs virtio-scsi (200 MB/s)

6. **Thread-Safe Lazy Pulse** (Line 20-21)
   - Add `threading.Lock()` for pulse property
   - Double-check locking pattern

7-14. **Additional Validations:**
   - VRAM overflow check (max 1TB)
   - RAM overflow check
   - FLOPS calculation max check (10 PFLOPS)
   - Empty disk_tiers list check
   - Model path existence validation
   - Exception handling in all property getters

**Output:** `zkie/core/kernel.py` (fully fixed, all tests pass)

### Task 1.2: Fix vertex_trinity (18 bugs)

**CATASTROPHIC Fixes:**

1. **Broken Lock-Free Atomics** (Line 44-54)
   ```python
   # WRONG: PyThread_acquire_lock is GIL, not memory fence
   def memory_barrier():
       ctypes.pythonapi.PyThread_acquire_lock(...)
   
   # CORRECT: Use multiprocessing.shared_memory + atomic CAS
   import multiprocessing as mp
   RING = mp.shared_memory.SharedMemory(create=True, size=RING_SIZE)
   
   # For true atomics, use C extension or ctypes to libc:
   libc = ctypes.CDLL('libc.so.6')
   libc.atomic_compare_exchange_strong_explicit.argtypes = [...]
   
   def atomic_update_head(new_head, expected_gen):
       # 128-bit CAS on (head, generation) pair
       pass
   ```

2. **id() != Memory Address** (Line 137-140)
   ```python
   # WRONG: id() is undefined on PyPy, returns random integer
   dst_ptr = ctypes.c_void_p(id(RING) + write_pos)
   
   # CORRECT: Use ctypes.addressof() with from_buffer
   ring_buf = (ctypes.c_char * RING_SIZE).from_buffer(RING)
   dst = ctypes.addressof(ring_buf) + write_pos
   
   src_buf = (ctypes.c_char * size_bytes).from_buffer(blob)
   ctypes.memmove(dst, ctypes.addressof(src_buf), size_bytes)
   ```

3. **GGUF Bounds Validation Missing** (Line 105-122)
   ```python
   offset = struct.unpack_from('<Q', src, pos)[0]
   
   # Add validation:
   if offset + size_bytes > len(src):
       raise ValueError(f"Tensor exceeds file: {offset}+{size_bytes} > {len(src)}")
   if offset < pos:
       raise ValueError(f"Tensor offset {offset} before KV section {pos}")
   
   blob = memoryview(src)[offset : offset + size_bytes]
   ```

4. **Tensor Size Overflow** (Line 120)
   ```python
   # Add overflow protection:
   MAX_TENSOR_SIZE = 16 << 30  # 16 GB
   product = math.prod(dims)
   if product > MAX_TENSOR_SIZE // dtype_size:
       raise ValueError(f"Tensor too large: {product * dtype_size / 1e9:.1f} GB")
   size_bytes = product * dtype_size
   ```

5-18. **Additional Fixes:**
   - GGUF version validation (only v2/v3)
   - Complete KV skip logic for all 13 types
   - Recursive string array handling
   - Magic number validation (0x46554747)
   - Generation counter wraparound handling
   - Ring full condition (wait loop with timeout)
   - Alignment based on actual cache line size (detect via sysconf)
   - cuBLAS handle caching
   - Proper type map for all GGUF dtypes
   - Error messages with context
   - Memory cleanup on exception
   - Validate n_dims < 8 (GGUF spec limit)

**Output:** `zkie/core/trinity.py` (lock-free atomics sealed, all bounds checks)

### Task 1.3: Fix vertex_hyper (15 bugs)

**CATASTROPHIC Fixes:**

1. **Unsigned Code Execution** (Line 79-104)
   ```python
   from cryptography.hazmat.primitives.asymmetric import ed25519
   
   # Add public key for signature verification
   ZKIE_PUBKEY = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(
       'YOUR_PUBLIC_KEY_HERE'  # Generate with ed25519.Ed25519PrivateKey.generate()
   ))
   
   def mutate(self):
       if not self.auto_update:  # OPT-IN only
           return
       
       # Download code + signature
       new_code = urlopen('https://.../zkie_hyper.py').read()
       signature = urlopen('https://.../zkie_hyper.sig').read()
       
       # Verify signature
       try:
           ZKIE_PUBKEY.verify(signature, new_code)
       except InvalidSignature:
           print('⚠ INVALID SIGNATURE — rejecting update')
           return
       
       # ... rest of atomic file replacement
   ```

2. **L1 Size Wrong** (Line 29)
   ```python
   def get_l1_size():
       try:
           # Read L1 DATA cache only (index0), not instruction cache
           l1d_path = Path('/sys/devices/system/cpu/cpu0/cache/index0/size')
           size_str = l1d_path.read_text().strip()
           if size_str.endswith('K'):
               return int(size_str[:-1]) * 1024
       except (FileNotFoundError, PermissionError, ValueError):
           return 32 * 1024  # 32KB typical L1 data cache
   ```

3. **inotify Path TOCTOU** (Line 45)
   ```python
   # WRONG: exists() check creates TOCTOU race
   if Path(watch_path).exists():
       wd = inotify.add_watch(watch_path, flags)
   
   # CORRECT: Just catch the exception
   try:
       wd = inotify.add_watch(watch_path, flags)
   except (FileNotFoundError, OSError):
       inotify = None
   ```

4. **Compression Oracle Sequential** (Line 183-186)
   ```python
   # Parallelize compression ratio checks:
   from concurrent.futures import ThreadPoolExecutor
   
   with ThreadPoolExecutor(max_workers=3) as ex:
       ratios = list(ex.map(
           lambda s: len(zlib.compress(s)) / len(s),
           samples
       ))
   ```

5-15. **Additional Fixes:**
   - L1 bounds check before write
   - Shutdown flag async-signal-safe
   - MOVED_TO/MOVED_FROM inotify flags
   - mmap for large files (>1MB)
   - Multi-point entropy sampling (first, mid, last)
   - Auto-update disabled by default
   - Atomic file replacement with backup
   - Shadow racer honest limitations
   - Clean resource cleanup
   - Signal handler thread-safety
   - Timeout on inotify.read()
   - Disk speed fallback heuristics

**Output:** `zkie/core/hyper.py` (signed updates, proper atomics, parallel compression)

### Stage 1 Testing Protocol

After all fixes applied:

```bash
# Test 1: Syntax validation
python -m py_compile zkie/core/kernel.py
python -m py_compile zkie/core/trinity.py
python -m py_compile zkie/core/hyper.py

# Test 2: Import without errors
python -c "from zkie.core import kernel, trinity, hyper"

# Test 3: Basic functionality
python zkie/core/kernel.py  # Should print hardware detection
python zkie/core/trinity.py  # Should initialize ring
python zkie/core/hyper.py  # Should start hyper loop

# Test 4: GPU benchmark accuracy (if CUDA available)
# Measure actual FLOPS vs claimed, verify within 20% tolerance

# Test 5: Crash resilience
# Kill -9 during rules.json write, verify file intact or missing (not corrupted)
```

**Deliverables:**
- ✅ 3 fixed core files in `zkie/core/`
- ✅ All 47 bugs resolved
- ✅ Test results documented
- ✅ Changelog with before/after for each fix

---

## 🔧 STAGE 2: CORE INFERENCE ENGINE (Parallel Track A — 400 Credits)

### Objective
Build actual LLM inference capability using llama.cpp with streaming, context management, and model loading.

### Architecture
```
zkie/
├── core/
│   ├── kernel.py        (from Stage 1)
│   ├── trinity.py       (from Stage 1)
│   ├── hyper.py         (from Stage 1)
│   └── inference/       (NEW)
│       ├── __init__.py
│       ├── engine.py         # Main inference orchestrator
│       ├── loader.py         # Model download + cache
│       ├── context.py        # KV cache + conversation state
│       ├── streamer.py       # Token streaming
│       └── backends/
│           ├── __init__.py
│           ├── llama_cpp.py  # llama.cpp backend
│           ├── vllm.py       # vLLM backend (optional)
│           └── base.py       # Abstract backend interface
```

### Task 2.1: Model Loader (`loader.py`)

```python
from huggingface_hub import hf_hub_download
from pathlib import Path
import hashlib

class ModelLoader:
    """Download and cache models from HuggingFace with resume support"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)
    
    def download(self, repo_id: str, filename: str, show_progress: bool = True) -> Path:
        """
        Download model with automatic resume and verification.
        
        Args:
            repo_id: HuggingFace repo (e.g., "TheBloke/Llama-2-7B-GGUF")
            filename: Model file (e.g., "llama-2-7b.Q4_K_M.gguf")
            show_progress: Show download progress bar
        
        Returns:
            Path to downloaded model file
        """
        # Check cache first
        cached = self.cache_dir / repo_id.replace('/', '_') / filename
        if cached.exists():
            if self._verify_integrity(cached, repo_id, filename):
                return cached
        
        # Download with resume
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=self.cache_dir,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        
        return Path(model_path)
    
    def _verify_integrity(self, path: Path, repo_id: str, filename: str) -> bool:
        """Verify file integrity via checksum (optional)"""
        # TODO: Implement checksum verification from HF metadata
        return path.exists() and path.stat().st_size > 0
    
    def list_local_models(self) -> list:
        """List all cached models"""
        models = []
        for model_dir in self.cache_dir.glob('*'):
            if model_dir.is_dir():
                for gguf in model_dir.glob('*.gguf'):
                    models.append({
                        'path': gguf,
                        'name': gguf.stem,
                        'size_gb': gguf.stat().st_size / 1e9
                    })
        return models
```

### Task 2.2: Context Manager (`context.py`)

```python
class ConversationContext:
    """Manage KV cache and conversation history with sliding window"""
    
    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
        self.history = []  # List of {"role": str, "content": str}
        self.current_tokens = 0
    
    def add_turn(self, role: str, content: str, token_count: int):
        """Add a conversation turn with sliding window"""
        self.history.append({
            "role": role,
            "content": content,
            "tokens": token_count
        })
        self.current_tokens += token_count
        
        # Slide window if over limit
        while self.current_tokens > self.max_tokens and len(self.history) > 1:
            removed = self.history.pop(0)
            self.current_tokens -= removed['tokens']
    
    def get_prompt(self, template: str = "chatml") -> str:
        """Format history into prompt with template"""
        if template == "chatml":
            parts = []
            for msg in self.history:
                parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
            parts.append("<|im_start|>assistant\n")
            return "\n".join(parts)
        
        elif template == "llama2":
            parts = ["[INST] "]
            for msg in self.history:
                if msg['role'] == 'user':
                    parts.append(msg['content'])
                else:
                    parts.append(f" [/INST] {msg['content']} [INST] ")
            parts.append(" [/INST]")
            return "".join(parts)
        
        # Add more templates as needed
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.history)
    
    def clear(self):
        """Reset conversation"""
        self.history = []
        self.current_tokens = 0
```

### Task 2.3: Llama.cpp Backend (`backends/llama_cpp.py`)

```python
from llama_cpp import Llama
from typing import Iterator, Dict, Any
import torch

class LlamaCppBackend:
    """llama.cpp inference backend with streaming"""
    
    def __init__(self, model_path: str, n_gpu_layers: int = -1, n_ctx: int = 4096):
        """
        Initialize llama.cpp model.
        
        Args:
            model_path: Path to GGUF file
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
            n_ctx: Context window size
        """
        self.model = Llama(
            model_path=str(model_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=512,
            verbose=False
        )
        self.n_ctx = n_ctx
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = True
    ) -> Iterator[Dict[str, Any]]:
        """
        Generate text with streaming.
        
        Yields:
            {"text": str, "done": bool, "tokens": int}
        """
        for chunk in self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            stop=["</s>", "<|im_end|>"]  # Common stop tokens
        ):
            if stream:
                yield {
                    "text": chunk["choices"][0]["text"],
                    "done": chunk["choices"][0]["finish_reason"] is not None,
                    "tokens": 1  # Approximate
                }
            else:
                # Non-streaming response
                yield {
                    "text": chunk["choices"][0]["text"],
                    "done": True,
                    "tokens": chunk["usage"]["completion_tokens"]
                }
                break
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count"""
        return len(self.model.tokenize(text.encode('utf-8')))
```

### Task 2.4: Main Inference Engine (`engine.py`)

```python
class InferenceEngine:
    """Main orchestrator for inference with model selection and routing"""
    
    def __init__(self, kernel):
        self.kernel = kernel  # VertexCore instance
        self.loader = ModelLoader(kernel.root / 'models')
        self.current_backend = None
        self.current_model = None
        self.context = ConversationContext()
    
    def load_model(self, model_spec: str = None):
        """
        Load model based on hardware or explicit spec.
        
        Args:
            model_spec: Optional "repo/file" string, else auto-select
        """
        if model_spec is None:
            # Auto-select based on hardware
            model_spec = self.kernel.select_model()
        
        # Parse spec
        if '/' in model_spec:
            repo_id, filename = model_spec.rsplit('/', 1)
        else:
            # Assume it's a filename, search local cache
            local = self.loader.list_local_models()
            matches = [m for m in local if model_spec in m['name']]
            if not matches:
                raise ValueError(f"Model {model_spec} not found. Download first.")
            model_path = matches[0]['path']
        
        # Download if needed
        if 'repo_id' in locals():
            model_path = self.loader.download(repo_id, filename)
        
        # Determine GPU layers
        if self.kernel.pulse.tensor_cores:
            n_gpu = -1  # Offload all layers
        else:
            n_gpu = 0  # CPU only
        
        # Load backend
        self.current_backend = LlamaCppBackend(
            model_path=model_path,
            n_gpu_layers=n_gpu,
            n_ctx=4096
        )
        self.current_model = model_spec
        
        print(f"✓ Loaded {model_spec} ({'GPU' if n_gpu else 'CPU'} mode)")
    
    def chat(
        self,
        message: str,
        stream: bool = True,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> Iterator[str]:
        """
        Chat with model (maintains conversation context).
        
        Args:
            message: User message
            stream: Stream tokens or return full response
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
        
        Yields:
            Token strings (if stream=True) or full response (if stream=False)
        """
        if self.current_backend is None:
            self.load_model()  # Auto-load if not loaded
        
        # Add user message to context
        token_count = self.current_backend.count_tokens(message)
        self.context.add_turn("user", message, token_count)
        
        # Build prompt with context
        prompt = self.context.get_prompt(template="chatml")
        
        # Generate response
        response_text = ""
        for chunk in self.current_backend.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream
        ):
            text = chunk["text"]
            response_text += text
            
            if stream:
                yield text
            
            if chunk["done"]:
                break
        
        # Add assistant response to context
        response_tokens = self.current_backend.count_tokens(response_text)
        self.context.add_turn("assistant", response_text, response_tokens)
        
        if not stream:
            yield response_text
```

### Stage 2 Deliverables
- ✅ `zkie/core/inference/` module (5 files)
- ✅ Working llama.cpp integration with streaming
- ✅ Model download with HuggingFace Hub
- ✅ Context management with sliding window
- ✅ Auto model selection based on hardware
- ✅ Test: Load Llama-2-7B and generate "Hello world" response

---

## 🔌 STAGE 3: PLUGIN SYSTEM (Parallel Track B — 400 Credits)

### Objective
Build modular plugin architecture for MCP servers, API providers, and webhooks.

### Architecture
```
zkie/
├── plugins/
│   ├── __init__.py
│   ├── base.py           # Abstract plugin interface
│   ├── manager.py        # Plugin lifecycle management
│   ├── mcp/              # MCP server integration
│   │   ├── __init__.py
│   │   ├── client.py     # SSE client for MCP servers
│   │   ├── registry.py   # Tool registry
│   │   └── executor.py   # Tool execution
│   ├── api/              # Cloud API providers
│   │   ├── __init__.py
│   │   ├── router.py     # Cost-aware routing (15% rule)
│   │   ├── providers/
│   │   │   ├── __init__.py
│   │   │   ├── openai.py
│   │   │   ├── anthropic.py
│   │   │   └── cohere.py
│   │   └── tracker.py    # Usage/cost tracking
│   └── webhook/          # Webhook integration
│       ├── __init__.py
│       ├── server.py     # Receive webhooks
│       └── client.py     # Send webhooks
```

### Task 3.1: Plugin Base (`base.py`)

```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class ZKIEPlugin(ABC):
    """Base class for all ZKIE plugins"""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = False
        self.config = {}
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize plugin with config.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def shutdown(self):
        """Clean up resources"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return plugin capabilities/features"""
        pass
```

### Task 3.2: MCP Client (`mcp/client.py`)

```python
import httpx
import json
from typing import Dict, List, AsyncIterator

class MCPClient:
    """Client for connecting to MCP servers via SSE"""
    
    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url
        self.tools = {}
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def connect(self):
        """Connect to MCP server and list tools"""
        async with self.client.stream("GET", self.url) as stream:
            # Parse SSE stream
            async for line in stream.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    
                    if data.get("type") == "tools/list":
                        self.tools = {
                            t["name"]: t for t in data["tools"]
                        }
                        break
    
    async def call_tool(self, tool_name: str, params: Dict) -> Any:
        """Execute tool on MCP server"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found in {self.name}")
        
        response = await self.client.post(
            f"{self.url}/call",
            json={
                "tool": tool_name,
                "arguments": params
            }
        )
        
        return response.json()
    
    def get_tool_schema(self, tool_name: str) -> Dict:
        """Get JSON schema for tool parameters"""
        return self.tools.get(tool_name, {}).get("inputSchema", {})
```

### Task 3.3: API Router with 15% Rule (`api/router.py`)

```python
class APIRouter:
    """Route requests to local or cloud based on 15% quality threshold"""
    
    def __init__(self, local_engine, budget_manager):
        self.local_engine = local_engine
        self.budget = budget_manager
        self.providers = {}
    
    def add_provider(self, name: str, provider):
        """Register API provider (OpenAI, Anthropic, etc.)"""
        self.providers[name] = provider
    
    async def route_request(
        self,
        prompt: str,
        task_type: str = "general",
        max_cost: float = None
    ):
        """
        Route to best provider using 15% threshold.
        
        Algorithm:
        1. Benchmark local model quality for task_type
        2. Check API providers within budget
        3. If API quality >= local * 1.15 AND cost <= max_cost: use API
        4. Else: use local (free)
        
        Args:
            prompt: User prompt
            task_type: "code", "creative", "analysis", etc.
            max_cost: Maximum acceptable cost in USD
        
        Returns:
            {"provider": str, "response": str, "cost": float}
        """
        # Estimate local quality (from benchmarks)
        local_quality = self._get_local_quality(task_type)
        
        # Check API options
        best_api = None
        best_quality = 0
        
        for name, provider in self.providers.items():
            if not provider.enabled:
                continue
            
            # Estimate cost
            cost = provider.estimate_cost(prompt, max_tokens=512)
            
            if max_cost and cost > max_cost:
                continue  # Over budget
            
            # Check monthly budget
            if not self.budget.can_afford(cost):
                continue  # Would exceed monthly limit
            
            # Get quality score
            quality = provider.get_quality_score(task_type)
            
            # 15% threshold check
            if quality >= local_quality * 1.15:
                if quality > best_quality:
                    best_api = (name, provider, cost, quality)
                    best_quality = quality
        
        # Decision
        if best_api:
            name, provider, cost, quality = best_api
            
            print(f"🌐 Routing to {name} (quality: {quality:.2f} vs local: {local_quality:.2f}, cost: ${cost:.4f})")
            
            response = await provider.complete(prompt)
            self.budget.record_usage(name, cost)
            
            return {
                "provider": name,
                "response": response,
                "cost": cost
            }
        else:
            # Use local (free)
            print(f"💻 Using local model (quality: {local_quality:.2f}, cost: $0)")
            
            response = ""
            for chunk in self.local_engine.chat(prompt):
                response += chunk
            
            return {
                "provider": "local",
                "response": response,
                "cost": 0.0
            }
    
    def _get_local_quality(self, task_type: str) -> float:
        """
        Get quality score for local model on task type.
        
        In production: Run benchmark suite and cache results.
        For now: Use heuristics based on model size.
        """
        # Simplified quality estimation
        model_name = self.local_engine.current_model or ""
        
        if "70B" in model_name:
            base_quality = 0.85
        elif "13B" in model_name or "medium" in model_name:
            base_quality = 0.70
        elif "7B" in model_name:
            base_quality = 0.60
        else:
            base_quality = 0.50  # Small models
        
        # Task-specific adjustments
        if task_type == "code" and "Code" in model_name:
            base_quality += 0.10
        elif task_type == "creative":
            base_quality += 0.05
        
        return min(base_quality, 1.0)
```

### Task 3.4: Budget Manager (`api/tracker.py`)

```python
import json
from datetime import datetime
from pathlib import Path

class BudgetManager:
    """Track API usage and enforce budget limits"""
    
    def __init__(self, monthly_limit: float = 100.0):
        self.monthly_limit = monthly_limit
        self.usage_file = Path.home() / '.zkie' / 'api_usage.json'
        self.usage_file.parent.mkdir(exist_ok=True, parents=True)
        self.usage = self._load_usage()
    
    def _load_usage(self) -> Dict:
        """Load usage history"""
        if self.usage_file.exists():
            return json.loads(self.usage_file.read_text())
        return {"history": [], "monthly_total": 0.0}
    
    def _save_usage(self):
        """Save usage history atomically"""
        import tempfile, os
        
        tmp = tempfile.NamedTemporaryFile(
            mode='w',
            delete=False,
            dir=self.usage_file.parent
        )
        json.dump(self.usage, tmp, indent=2)
        tmp.close()
        os.replace(tmp.name, self.usage_file)
    
    def can_afford(self, cost: float) -> bool:
        """Check if cost fits within budget"""
        # Reset if new month
        self._check_month_rollover()
        
        return (self.usage["monthly_total"] + cost) <= self.monthly_limit
    
    def record_usage(self, provider: str, cost: float):
        """Record API usage"""
        self._check_month_rollover()
        
        self.usage["history"].append({
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "cost": cost
        })
        self.usage["monthly_total"] += cost
        
        self._save_usage()
    
    def _check_month_rollover(self):
        """Reset monthly total if new month"""
        if not self.usage["history"]:
            return
        
        last_entry = self.usage["history"][-1]
        last_date = datetime.fromisoformat(last_entry["timestamp"])
        
        if last_date.month != datetime.now().month:
            self.usage["monthly_total"] = 0.0
    
    def get_summary(self) -> Dict:
        """Get usage summary"""
        return {
            "monthly_total": self.usage["monthly_total"],
            "monthly_limit": self.monthly_limit,
            "remaining": self.monthly_limit - self.usage["monthly_total"],
            "utilization": self.usage["monthly_total"] / self.monthly_limit
        }
```

### Stage 3 Deliverables
- ✅ Plugin architecture with base class
- ✅ MCP client with SSE support
- ✅ API router with 15% threshold enforcement
- ✅ Budget tracking and limits
- ✅ OpenAI + Anthropic provider implementations
- ✅ Test: Connect to MCP server, execute tool, verify 15% routing logic

---

## 🎨 STAGE 4: UI LAYER (Parallel Track C — 300 Credits)

### Objective
Build Gradio chat interface and FastAPI REST API with OpenAI-compatible endpoints.

### Architecture
```
zkie/
├── ui/
│   ├── __init__.py
│   ├── gradio_app.py    # Gradio web interface
│   ├── api_server.py    # FastAPI REST API
│   └── templates/
│       └── chat.html    # Optional custom UI
```

### Task 4.1: Gradio Interface (`gradio_app.py`)

```python
import gradio as gr
from zkie.core.kernel import VertexCore
from zkie.core.inference.engine import InferenceEngine

class ZKIEGradioApp:
    """Gradio web interface for ZKIE"""
    
    def __init__(self):
        self.kernel = VertexCore()
        self.engine = InferenceEngine(self.kernel)
        self.engine.load_model()  # Auto-load based on hardware
    
    def chat_fn(self, message, history):
        """Chat function for Gradio interface"""
        response = ""
        
        for chunk in self.engine.chat(message, stream=True):
            response += chunk
            yield response
    
    def launch(self, share=False, server_port=7860):
        """Launch Gradio interface"""
        # Create interface
        demo = gr.ChatInterface(
            fn=self.chat_fn,
            title="⚡ Zenith Kernel Inference Engine",
            description=f"""
            **Model:** {self.engine.current_model or 'Auto-selected'}  
            **Hardware:** {self.kernel.pulse.flops_per_sec/1e9:.1f} GFLOPS, 
            {self.kernel.pulse.ram_gb:.1f}GB RAM, 
            {self.kernel.pulse.vram_gb:.1f}GB VRAM  
            **Mode:** {'GPU' if self.kernel.pulse.tensor_cores else 'CPU'}
            """,
            examples=[
                "Write a Python function to calculate Fibonacci numbers",
                "Explain quantum entanglement in simple terms",
                "What are the key differences between REST and GraphQL?"
            ],
            theme=gr.themes.Soft(),
            analytics_enabled=False  # Privacy-first
        )
        
        # Launch
        demo.launch(
            share=share,
            server_name="0.0.0.0",
            server_port=server_port,
            show_error=True
        )

def main():
    app = ZKIEGradioApp()
    app.launch()

if __name__ == "__main__":
    main()
```

### Task 4.2: FastAPI Server (`api_server.py`)

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import asyncio

app = FastAPI(title="ZKIE API", version="1.0.0")

# Global engine instance
engine = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "auto"
    messages: List[ChatMessage]
    stream: bool = False
    max_tokens: int = 512
    temperature: float = 0.7

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]

@app.on_event("startup")
async def startup():
    """Initialize engine on startup"""
    global engine
    from zkie.core.kernel import VertexCore
    from zkie.core.inference.engine import InferenceEngine
    
    kernel = VertexCore()
    engine = InferenceEngine(kernel)
    engine.load_model()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat endpoint"""
    
    # Get last user message
    user_message = None
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break
    
    if not user_message:
        raise HTTPException(400, "No user message found")
    
    if request.stream:
        # Streaming response
        async def generate():
            for chunk in engine.chat(
                message=user_message,
                stream=True,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            ):
                sse_data = {
                    "id": "chatcmpl-zkie",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": engine.current_model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(sse_data)}\n\n"
            
            # Final chunk
            yield f"data: {json.dumps({**sse_data, 'choices': [{'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    
    else:
        # Non-streaming response
        response_text = ""
        for chunk in engine.chat(
            message=user_message,
            stream=False,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        ):
            response_text = chunk  # Will be full response
        
        return ChatCompletionResponse(
            id="chatcmpl-zkie",
            created=int(time.time()),
            model=engine.current_model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }]
        )

@app.get("/v1/models")
async def list_models():
    """List available models"""
    local_models = engine.loader.list_local_models()
    
    return {
        "object": "list",
        "data": [
            {
                "id": m["name"],
                "object": "model",
                "owned_by": "zkie",
                "size": m["size_gb"]
            }
            for m in local_models
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": engine.current_model,
        "hardware": {
            "ram_gb": engine.kernel.pulse.ram_gb,
            "vram_gb": engine.kernel.pulse.vram_gb,
            "tensor_cores": engine.kernel.pulse.tensor_cores
        }
    }

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
```

### Stage 4 Deliverables
- ✅ Gradio chat interface (50 lines)
- ✅ FastAPI server with OpenAI-compatible endpoints
- ✅ Streaming support for both UIs
- ✅ Health check and model listing
- ✅ Test: Open Gradio UI, send message, verify streaming works
- ✅ Test: curl API endpoint, verify OpenAI compatibility

---

## 🔒 STAGE 5: PRIVACY & OFFLINE MODE (Sequential — 300 Credits)

### Objective
Implement comprehensive offline/privacy system with air-gap support, connection controls, and compliance features.

### Architecture
```
zkie/
├── privacy/
│   ├── __init__.py
│   ├── controller.py     # Connection permission system
│   ├── modes.py          # Privacy mode definitions
│   ├── audit.py          # Audit logging
│   └── offline.py        # Offline bundle manager
```

### Task 5.1: Privacy Controller (`controller.py`)

```python
from enum import Enum
from typing import Optional, Set
import json
from pathlib import Path

class PrivacyMode(Enum):
    FULL_PRIVACY = "full_privacy"      # Air-gap, no connections
    BALANCED = "balanced"               # Safe defaults (recommended)
    SELECTIVE = "selective"             # Ask every time
    CLOUD_FIRST = "cloud_first"        # Allow all

class OfflineController:
    """Centralized control for all external connections"""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.mode = PrivacyMode.BALANCED
        self.allowed_domains: Set[str] = set()
        self.audit_log = AuditLog(config_path.parent / 'audit.log')
        self._load_config()
    
    def can_connect(self, domain: str, reason: str, data_summary: Optional[str] = None) -> bool:
        """
        Check if connection is allowed.
        
        Args:
            domain: Target domain (e.g., "huggingface.co")
            reason: Purpose ("model_download", "mcp_tool", "api_call", etc.)
            data_summary: Optional summary of data being sent
        
        Returns:
            True if allowed, False if blocked
        """
        # Log attempt
        decision = self._make_decision(domain, reason)
        
        self.audit_log.log(
            domain=domain,
            reason=reason,
            allowed=decision,
            data_summary=data_summary,
            mode=self.mode.value
        )
        
        return decision
    
    def _make_decision(self, domain: str, reason: str) -> bool:
        """Internal decision logic"""
        
        if self.mode == PrivacyMode.FULL_PRIVACY:
            # Never allow
            return False
        
        if self.mode == PrivacyMode.CLOUD_FIRST:
            # Always allow
            return True
        
        if self.mode == PrivacyMode.SELECTIVE:
            # Ask user (would integrate with UI)
            return self._prompt_user(domain, reason)
        
        # BALANCED mode - smart defaults
        BLOCKED_REASONS = {
            'telemetry',
            'analytics',
            'code_update'  # Disabled by default for security
        }
        
        if reason in BLOCKED_REASONS:
            return False
        
        ALLOWED_REASONS = {
            'model_download',   # User-initiated
            'mcp_tool_use',     # User-initiated
            'security_update'   # Critical patches only
        }
        
        if reason in ALLOWED_REASONS:
            # Check if domain trusted
            if domain in self.allowed_domains:
                return True
            
            # First time - would prompt in UI
            # For automated testing, auto-trust common domains
            if domain in {'huggingface.co', 'github.com'}:
                self.allowed_domains.add(domain)
                self._save_config()
                return True
        
        # Default: block unknown
        return False
    
    def _prompt_user(self, domain: str, reason: str) -> bool:
        """Prompt user for permission (would integrate with UI)"""
        # For now, return False for safety
        # In production: Show UI dialog
        return False
    
    def _load_config(self):
        """Load privacy settings"""
        if self.config_path.exists():
            config = json.loads(self.config_path.read_text())
            self.mode = PrivacyMode(config.get('mode', 'balanced'))
            self.allowed_domains = set(config.get('allowed_domains', []))
    
    def _save_config(self):
        """Save privacy settings atomically"""
        import tempfile, os
        
        config = {
            'mode': self.mode.value,
            'allowed_domains': list(self.allowed_domains)
        }
        
        tmp = tempfile.NamedTemporaryFile(
            mode='w',
            delete=False,
            dir=self.config_path.parent
        )
        json.dump(config, tmp, indent=2)
        tmp.close()
        os.replace(tmp.name, self.config_path)
```

### Task 5.2: Audit Log (`audit.py`)

```python
import json
from datetime import datetime
from pathlib import Path
import hashlib

class AuditLog:
    """Immutable append-only audit log for compliance"""
    
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(exist_ok=True, parents=True)
    
    def log(
        self,
        domain: str,
        reason: str,
        allowed: bool,
        data_summary: Optional[str] = None,
        mode: str = "balanced"
    ):
        """Log connection attempt"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'domain': domain,
            'reason': reason,
            'allowed': allowed,
            'mode': mode
        }
        
        # Hash data summary (don't store PII)
        if data_summary:
            entry['data_hash'] = hashlib.sha256(
                data_summary.encode()
            ).hexdigest()[:16]
        
        # Append-only log
        with self.log_path.open('a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def export_report(self, start_date: datetime, end_date: datetime) -> str:
        """Generate compliance report"""
        entries = []
        
        with self.log_path.open('r') as f:
            for line in f:
                entry = json.loads(line)
                entry_date = datetime.fromisoformat(entry['timestamp'])
                
                if start_date <= entry_date <= end_date:
                    entries.append(entry)
        
        # Generate report
        report = f"# ZKIE Audit Report\n"
        report += f"Period: {start_date.date()} to {end_date.date()}\n\n"
        report += f"Total connections: {len(entries)}\n"
        report += f"Allowed: {sum(1 for e in entries if e['allowed'])}\n"
        report += f"Blocked: {sum(1 for e in entries if not e['allowed'])}\n\n"
        
        report += "## Connection Log\n"
        for entry in entries:
            status = "✓" if entry['allowed'] else "✗"
            report += f"{status} {entry['timestamp']} | {entry['domain']} | {entry['reason']}\n"
        
        return report
```

### Task 5.3: Offline Bundle Manager (`offline.py`)

```python
import zstandard as zstd
import json
import hashlib
from pathlib import Path
from datetime import datetime

class OfflineUpdater:
    """Create and apply offline update bundles for air-gap environments"""
    
    def __init__(self, zkie_root: Path):
        self.zkie_root = zkie_root
    
    def create_bundle(self, output_path: Path):
        """
        Create update bundle for offline transfer.
        
        Bundle includes:
        - All Python source files
        - SHA256 checksums
        - Ed25519 signature (if private key available)
        - Version metadata
        """
        bundle = {
            'version': self._get_version(),
            'created': datetime.now().isoformat(),
            'files': {},
            'checksums': {}
        }
        
        # Collect all Python files
        for py_file in self.zkie_root.rglob('*.py'):
            rel_path = py_file.relative_to(self.zkie_root)
            content = py_file.read_bytes()
            
            bundle['files'][str(rel_path)] = content.decode('utf-8')
            bundle['checksums'][str(rel_path)] = hashlib.sha256(content).hexdigest()
        
        # Sign bundle (if key available)
        bundle_json = json.dumps(bundle, sort_keys=True)
        
        try:
            from cryptography.hazmat.primitives.asymmetric import ed25519
            
            # Load private key (would be stored securely)
            privkey_path = self.zkie_root / '.signing_key'
            if privkey_path.exists():
                private_key = ed25519.Ed25519PrivateKey.from_private_bytes(
                    privkey_path.read_bytes()
                )
                signature = private_key.sign(bundle_json.encode())
                bundle['signature'] = signature.hex()
        except Exception as e:
            print(f"⚠ Could not sign bundle: {e}")
        
        # Compress and write
        compressed = zstd.compress(bundle_json.encode(), level=9)
        output_path.write_bytes(compressed)
        
        print(f"✓ Created bundle: {output_path} ({len(compressed) / 1e6:.1f} MB)")
        print(f"  Files: {len(bundle['files'])}")
        print(f"  Signed: {'Yes' if 'signature' in bundle else 'No'}")
    
    def apply_bundle(self, bundle_path: Path):
        """
        Apply offline update bundle.
        
        Steps:
        1. Decompress bundle
        2. Verify signature
        3. Verify checksums
        4. Create backup
        5. Apply update atomically
        """
        # Decompress
        compressed = bundle_path.read_bytes()
        bundle_json = zstd.decompress(compressed).decode('utf-8')
        bundle = json.loads(bundle_json)
        
        print(f"📦 Applying bundle version {bundle['version']}...")
        
        # Verify signature
        if 'signature' in bundle:
            if not self._verify_signature(bundle_json, bundle['signature']):
                raise SecurityError("Invalid bundle signature")
            print("✓ Signature verified")
        else:
            print("⚠ Bundle not signed")
        
        # Verify checksums
        for rel_path, expected_hash in bundle['checksums'].items():
            content = bundle['files'][rel_path].encode('utf-8')
            actual_hash = hashlib.sha256(content).hexdigest()
            
            if actual_hash != expected_hash:
                raise SecurityError(f"Checksum mismatch: {rel_path}")
        
        print("✓ All checksums verified")
        
        # Create backup
        backup_dir = self.zkie_root.parent / f"zkie_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copytree(self.zkie_root, backup_dir)
        print(f"✓ Backup created: {backup_dir}")
        
        # Apply update
        for rel_path, content in bundle['files'].items():
            target = self.zkie_root / rel_path
            target.parent.mkdir(exist_ok=True, parents=True)
            target.write_text(content)
        
        print(f"✓ Applied {len(bundle['files'])} files")
        print(f"✓ Updated to version {bundle['version']}")
    
    def _get_version(self) -> str:
        """Get current ZKIE version"""
        # Would read from __version__.py or similar
        return "1.0.0"
    
    def _verify_signature(self, data: str, signature_hex: str) -> bool:
        """Verify Ed25519 signature"""
        try:
            from cryptography.hazmat.primitives.asymmetric import ed25519
            
            # Load public key (would be embedded in code)
            pubkey_path = self.zkie_root / '.public_key'
            if not pubkey_path.exists():
                return False
            
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(
                pubkey_path.read_bytes()
            )
            
            signature = bytes.fromhex(signature_hex)
            public_key.verify(signature, data.encode())
            return True
        except Exception:
            return False
```

### Stage 5 Deliverables
- ✅ Privacy controller with 4 modes
- ✅ Audit logging for compliance
- ✅ Offline bundle creation/application
- ✅ Connection permission system
- ✅ Test: Set FULL_PRIVACY mode, verify all connections blocked
- ✅ Test: Create bundle, apply bundle, verify update successful

---

## 🧪 STAGE 6: INTEGRATION TESTING & REFINEMENT (Sequential — 400 Credits)

### Objective
Run comprehensive close-loop tests, document issues, create refinement plan, fix dependencies.

### Testing Framework
```
zkie/
├── tests/
│   ├── __init__.py
│   ├── test_core.py           # Core kernel tests
│   ├── test_trinity.py        # Ring buffer tests
│   ├── test_hyper.py          # Hyper layer tests
│   ├── test_inference.py      # Inference engine tests
│   ├── test_plugins.py        # Plugin system tests
│   ├── test_privacy.py        # Privacy/offline tests
│   ├── test_integration.py    # End-to-end tests
│   └── benchmarks/
│       ├── flops_accuracy.py  # Verify FLOPS measurements
│       ├── latency.py         # Measure inference latency
│       └── memory.py          # Memory usage profiling
```

### Task 6.1: Core Test Suite

```python
# tests/test_core.py
import pytest
import time
from zkie.core.kernel import VertexCore

def test_gpu_benchmark_sync():
    """Verify GPU benchmark uses synchronization"""
    core = VertexCore()
    
    # Should have CUDA sync if tensor cores available
    if core.pulse.tensor_cores:
        import torch
        # Verify FLOPS are reasonable (not 10-50× too high)
        # Expected: 10-100 TFLOPS for modern GPU
        assert 10e12 < core.pulse.flops_per_sec < 100e12
    
def test_cpu_features_cached():
    """Verify /proc/cpuinfo is cached"""
    core = VertexCore()
    
    start = time.time()
    for _ in range(1000):
        model = core.select_model()
    elapsed = time.time() - start
    
    # Should be <100ms for 1000 calls if cached (vs 5000ms if not)
    assert elapsed < 0.5, f"CPU features not cached: {elapsed}s for 1000 calls"

def test_atomic_file_writes():
    """Verify rules.json writes are atomic"""
    import os, signal, subprocess
    
    # This would require process forking to test properly
    # For now, verify tempfile pattern is used
    core = VertexCore()
    assert hasattr(core, 'save_rules'), "Atomic write method missing"

def test_disk_speed_detection():
    """Verify disk speed detection works"""
    core = VertexCore()
    
    if core.disk_tiers:
        for path, speed, size in core.disk_tiers:
            # Speed should be reasonable (100-1000 MB/s)
            assert 100 <= speed <= 1000, f"Invalid speed: {speed} for {path}"

# Add 20+ more tests...
```

### Task 6.2: Integration Test Suite

```python
# tests/test_integration.py
import pytest
import asyncio
from zkie.core.kernel import VertexCore
from zkie.core.inference.engine import InferenceEngine

def test_end_to_end_inference():
    """Test complete inference pipeline"""
    # 1. Initialize core
    kernel = VertexCore()
    
    # 2. Load inference engine
    engine = InferenceEngine(kernel)
    
    # 3. Load model (auto-select based on hardware)
    engine.load_model()
    
    # 4. Generate response
    response = ""
    for chunk in engine.chat("What is 2+2?", stream=True):
        response += chunk
    
    # 5. Verify response is reasonable
    assert len(response) > 10, "Response too short"
    assert "4" in response.lower(), "Incorrect answer"

@pytest.mark.asyncio
async def test_mcp_integration():
    """Test MCP server connection and tool execution"""
    from zkie.plugins.mcp.client import MCPClient
    
    # Would use mock MCP server for testing
    # For now, test client initialization
    client = MCPClient("test", "https://example.com/mcp")
    assert client.name == "test"

def test_api_router_15_percent_rule():
    """Verify 15% threshold is enforced"""
    from zkie.plugins.api.router import APIRouter
    from zkie.core.kernel import VertexCore
    from zkie.core.inference.engine import InferenceEngine
    
    kernel = VertexCore()
    engine = InferenceEngine(kernel)
    
    # Mock local quality = 0.60
    # Mock API quality = 0.68 (13% better)
    # Should use local (doesn't meet 15% threshold)
    
    # Mock API quality = 0.70 (16.7% better)
    # Should use API (meets 15% threshold)
    
    # Test implementation...

def test_privacy_modes():
    """Test all privacy modes"""
    from zkie.privacy.controller import OfflineController, PrivacyMode
    from pathlib import Path
    
    ctrl = OfflineController(Path('/tmp/zkie_test_config.json'))
    
    # FULL_PRIVACY: should block everything
    ctrl.mode = PrivacyMode.FULL_PRIVACY
    assert not ctrl.can_connect("huggingface.co", "model_download")
    assert not ctrl.can_connect("api.openai.com", "api_call")
    
    # BALANCED: should allow model downloads, block telemetry
    ctrl.mode = PrivacyMode.BALANCED
    assert ctrl.can_connect("huggingface.co", "model_download")
    assert not ctrl.can_connect("analytics.example.com", "telemetry")

# Add 30+ more integration tests...
```

### Task 6.3: Close-Loop Testing Process

1. **Run all tests** → Document failures
2. **Analyze failures** → Categorize by severity
3. **Create refinement plan** → Priority-ordered fixes
4. **Fix bugs iteratively** → Re-test after each fix
5. **Verify hermetic sealing** → No external dependencies leak

**Test Execution:**
```bash
# Run all tests
pytest zkie/tests/ -v --tb=short

# Generate coverage report
pytest zkie/tests/ --cov=zkie --cov-report=html

# Run benchmarks
python zkie/tests/benchmarks/flops_accuracy.py
python zkie/tests/benchmarks/latency.py
python zkie/tests/benchmarks/memory.py
```

### Task 6.4: Issue Documentation

Create structured issue log:

```markdown
# ZKIE Integration Test Results — 2025-12-31

## Test Summary
- Total tests: 78
- Passed: 64
- Failed: 14
- Skipped: 0
- Coverage: 87%

## Critical Issues (Block Release)

### Issue #1: Trinity ring atomics fail on ARM64
**Severity:** CATASTROPHIC  
**Test:** `test_trinity.py::test_concurrent_writes`  
**Symptom:** Data corruption in 94% of runs on Raspberry Pi 4  
**Root Cause:** `ctypes.pythonapi.PyThread_acquire_lock` not a memory fence  
**Fix:** Replace with `multiprocessing.shared_memory` + C extension for CAS  
**ETA:** 2 days  

### Issue #2: GPU benchmark measures queue time not exec time
**Severity:** CRITICAL  
**Test:** `test_core.py::test_flops_accuracy`  
**Symptom:** FLOPS 30× too high (300 TFLOPS on RTX 4090, should be ~80)  
**Root Cause:** Missing `torch.cuda.synchronize()`  
**Fix:** Add sync before and after benchmark loop  
**ETA:** 30 minutes  

... (document all 14 failures)

## High-Priority Enhancements

### Enhancement #1: Add llama.cpp quantization support
**Benefit:** Reduce VRAM usage by 50-75%  
**Effort:** 3 days  
**Priority:** HIGH  

... (10+ enhancements)
```

### Task 6.5: Refinement Plan

```markdown
# ZKIE Refinement Plan

## Phase 1: Critical Bugs (Days 1-3)
- [ ] Fix Trinity ring atomics (Issue #1)
- [ ] Fix GPU benchmark sync (Issue #2)
- [ ] Fix GGUF bounds validation (Issue #3)
- [ ] Fix id() pointer bug (Issue #4)
- [ ] Add code signing (Issue #5)

## Phase 2: High-Priority Features (Days 4-7)
- [ ] Add quantization support
- [ ] Optimize compression oracle parallelization
- [ ] Cache CPU features
- [ ] Implement atomic file writes
- [ ] Add proper error messages

## Phase 3: Polish (Days 8-10)
- [ ] Add logging framework
- [ ] Improve CLI experience
- [ ] Add progress bars
- [ ] Write comprehensive docs
- [ ] Create tutorial videos

## Success Criteria
- ✅ All 78 tests passing
- ✅ 95%+ code coverage
- ✅ Zero memory leaks (valgrind clean)
- ✅ <2s startup time
- ✅ <100ms first token latency
```

### Stage 6 Deliverables
- ✅ Comprehensive test suite (78+ tests)
- ✅ Issue documentation (14 critical issues cataloged)
- ✅ Refinement plan with timeline
- ✅ Fixed all critical bugs
- ✅ Hermetic dependency verification
- ✅ Benchmark results showing real vs theoretical FLOPS

---

## 📦 STAGE 7: PACKAGING & DOCUMENTATION (Sequential — 300 Credits)

### Objective
Create master zip file, file tree diagram, user manual, and installation scripts.

### Task 7.1: File Tree Generator

```python
# zkie/tools/generate_tree.py
from pathlib import Path

def generate_tree(root: Path, output: Path, max_depth: int = 5):
    """Generate ASCII file tree diagram"""
    
    def tree_line(path: Path, prefix: str = "", depth: int = 0):
        if depth > max_depth:
            return []
        
        lines = []
        
        # Current item
        if path.is_dir():
            lines.append(f"{prefix}📁 {path.name}/")
            
            # Children
            children = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            for i, child in enumerate(children):
                is_last = (i == len(children) - 1)
                new_prefix = prefix + ("    " if is_last else "│   ")
                child_prefix = prefix + ("└── " if is_last else "├── ")
                
                lines.append(tree_line(child, child_prefix, depth + 1)[0])
                
                if child.is_dir() and depth < max_depth:
                    lines.extend(tree_line(child, new_prefix, depth + 1)[1:])
        else:
            icon = "📄" if path.suffix == ".py" else "📋"
            size = path.stat().st_size
            if size < 1024:
                size_str = f"{size}B"
            elif size < 1024**2:
                size_str = f"{size/1024:.1f}KB"
            else:
                size_str = f"{size/1024**2:.1f}MB"
            
            lines.append(f"{prefix}{icon} {path.name} ({size_str})")
        
        return lines
    
    tree = tree_line(root)
    output.write_text("\n".join(tree))
    print(f"✓ Generated tree: {output}")

if __name__ == "__main__":
    generate_tree(
        Path(__file__).parent.parent.parent,
        Path("ZKIE_FILE_TREE.txt")
    )
```

**Expected Output:**
```
📁 zkie/
├── 📄 __init__.py (248B)
├── 📁 core/
│   ├── 📄 __init__.py (156B)
│   ├── 📄 kernel.py (8.5KB)
│   ├── 📄 trinity.py (5.9KB)
│   ├── 📄 hyper.py (9.3KB)
│   └── 📁 inference/
│       ├── 📄 __init__.py (203B)
│       ├── 📄 engine.py (4.2KB)
│       ├── 📄 loader.py (2.8KB)
│       ├── 📄 context.py (2.1KB)
│       └── 📁 backends/
│           ├── 📄 __init__.py (89B)
│           ├── 📄 base.py (1.1KB)
│           └── 📄 llama_cpp.py (3.4KB)
├── 📁 plugins/
│   ├── 📄 __init__.py (178B)
│   ├── 📄 base.py (1.3KB)
│   ├── 📄 manager.py (2.7KB)
│   ├── 📁 mcp/
│   │   ├── 📄 __init__.py (134B)
│   │   ├── 📄 client.py (3.1KB)
│   │   ├── 📄 registry.py (1.8KB)
│   │   └── 📄 executor.py (2.4KB)
│   ├── 📁 api/
│   │   ├── 📄 __init__.py (156B)
│   │   ├── 📄 router.py (4.6KB)
│   │   ├── 📄 tracker.py (2.9KB)
│   │   └── 📁 providers/
│   │       ├── 📄 __init__.py (112B)
│   │       ├── 📄 openai.py (2.3KB)
│   │       └── 📄 anthropic.py (2.1KB)
│   └── 📁 webhook/
│       ├── 📄 __init__.py (98B)
│       ├── 📄 server.py (2.8KB)
│       └── 📄 client.py (1.9KB)
├── 📁 privacy/
│   ├── 📄 __init__.py (145B)
│   ├── 📄 controller.py (3.8KB)
│   ├── 📄 modes.py (567B)
│   ├── 📄 audit.py (2.4KB)
│   └── 📄 offline.py (4.1KB)
├── 📁 ui/
│   ├── 📄 __init__.py (123B)
│   ├── 📄 gradio_app.py (2.6KB)
│   └── 📄 api_server.py (4.3KB)
├── 📁 tests/
│   ├── 📄 test_core.py (5.7KB)
│   ├── 📄 test_trinity.py (4.2KB)
│   ├── 📄 test_inference.py (3.8KB)
│   ├── 📄 test_plugins.py (3.1KB)
│   ├── 📄 test_privacy.py (2.9KB)
│   └── 📄 test_integration.py (6.4KB)
├── 📄 setup.py (1.8KB)
├── 📄 requirements.txt (487B)
├── 📄 README.md (12.3KB)
├── 📄 LICENSE (1.1KB)
└── 📄 CHANGELOG.md (3.4KB)

Total: 47 files, 89.7KB source code
```

### Task 7.2: User Manual

```markdown
# ZENITH KERNEL INFERENCE ENGINE — USER MANUAL
## Version 1.0 | 2025-12-31

---

## Table of Contents
1. Introduction
2. Installation
3. Quick Start
4. Hardware Requirements
5. Features
6. Configuration
7. Privacy & Security
8. API Reference
9. Troubleshooting
10. FAQ

---

## 1. Introduction

**Zenith Kernel Inference Engine (ZKIE)** is the world's first privacy-first, universally adaptive AI inference orchestrator. It runs on any hardware from Celeron processors to datacenter clusters, with unique features:

- ✅ **Auto-hardware detection** — Selects optimal model for your hardware
- ✅ **Native MCP support** — Connect to Asana, Gmail, GitHub, and more
- ✅ **15% cost optimization** — Only uses paid APIs if 15%+ better than local
- ✅ **Offline air-gap mode** — Complete privacy, no internet required
- ✅ **Streaming inference** — Real-time token generation
- ✅ **OpenAI-compatible API** — Drop-in replacement for existing apps

### What Makes ZKIE Different?

| Feature | ZKIE | Ollama | LM Studio | text-gen-webui |
|---------|------|--------|-----------|----------------|
| **Auto model selection** | ✅ VRAM-aware | ❌ Manual | ❌ Manual | ❌ Manual |
| **MCP servers** | ✅ Native | ❌ No | ❌ No | ❌ No |
| **Cost routing** | ✅ 15% rule | ❌ No | ❌ No | ❌ No |
| **Offline mode** | ✅ Air-gap | ⚠️ Partial | ⚠️ Partial | ⚠️ Partial |
| **Privacy controls** | ✅ 4 modes | ❌ No | ❌ No | ❌ No |

---

## 2. Installation

### Prerequisites
- Python 3.10+
- 8GB RAM minimum (16GB recommended)
- Optional: NVIDIA GPU with CUDA 12+ (for acceleration)

### Quick Install

```bash
# Install ZKIE
pip install zkie

# Or install from source
git clone https://github.com/yourusername/zkie.git
cd zkie
pip install -e .
```

### Verify Installation

```bash
# Check version
zkie --version

# Run health check
zkie doctor

# Expected output:
# ✓ Python 3.11.5
# ✓ CUDA 12.2 detected
# ✓ 24GB VRAM available
# ✓ All dependencies installed
```

---

## 3. Quick Start

### Start Web Interface

```bash
# Launch Gradio UI (opens in browser)
zkie ui

# Custom port
zkie ui --port 8080

# Share publicly (via Gradio tunnel)
zkie ui --share
```

### Start API Server

```bash
# Launch FastAPI server
zkie serve --port 8000

# Test with curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is quantum entanglement?"}],
    "stream": true
  }'
```

### Python API

```python
from zkie import ZKIE

# Initialize
engine = ZKIE()

# Auto-load model based on hardware
engine.load_model()

# Chat (streaming)
for chunk in engine.chat("Write a Python function to sort a list"):
    print(chunk, end="", flush=True)
```

---

## 4. Hardware Requirements

### Minimum (CPU-only)
- Intel Core i5 / AMD Ryzen 5 (2015+)
- 8GB RAM
- 10GB disk space
- Runs: Llama-2-7B-Q4, Phi-3-mini, TinyLlama

### Recommended (GPU)
- NVIDIA RTX 3060+ (12GB VRAM)
- 16GB RAM
- 50GB disk space
- Runs: Llama-3.1-70B-Q4, Mixtral-8x7B, CodeLlama-34B

### High-end (Datacenter)
- NVIDIA A100 / H100
- 256GB+ RAM
- 500GB disk space
- Runs: Any model, multiple simultaneously

### Hardware Detection

ZKIE auto-detects your hardware and selects the best model:

```bash
zkie info

# Output:
# 🖥️  Hardware Profile
# CPU: AMD Ryzen 9 5950X (32 threads)
# RAM: 64.0 GB
# GPU: NVIDIA RTX 4090
# VRAM: 24.0 GB
# Tensor Cores: Yes (Ada Lovelace)
# 
# 📊 Performance
# FLOPS: 82.4 TFLOPS
# Memory Bandwidth: 1,008 GB/s
# 
# 🎯 Recommended Model
# Llama-3.1-70B-Instruct-Q4_K_M (fits in VRAM with 4GB headroom)
```

---

## 5. Features

### 5.1 Model Management

```bash
# List available models
zkie models list

# Download model
zkie models download "TheBloke/Llama-2-13B-GGUF" "llama-2-13b.Q4_K_M.gguf"

# Remove model
zkie models remove "llama-2-13b.Q4_K_M.gguf"

# Search HuggingFace
zkie models search "code" --format gguf
```

### 5.2 MCP Server Integration

```bash
# Connect to MCP server
zkie mcp add asana https://mcp.asana.com/sse

# List connected servers
zkie mcp list

# Test tool execution
zkie mcp test asana create_task '{"name": "Test task"}'
```

**Python API:**
```python
from zkie import ZKIE

engine = ZKIE()
engine.connect_mcp("asana", "https://mcp.asana.com/sse")

# Use tools in chat
response = engine.chat(
    "Create an Asana task to review Q4 metrics",
    tools_enabled=True
)
```

### 5.3 API Provider Routing

```bash
# Add API provider
zkie api add openai --key sk-...

# Set monthly budget
zkie api budget set 50  # $50/month

# Enable 15% rule
zkie api config --threshold 15

# View usage
zkie api usage

# Output:
# 📊 API Usage (December 2025)
# OpenAI: $12.34 / $50.00 (24.7%)
# Anthropic: $3.21 / $50.00 (6.4%)
# Total: $15.55 / $50.00 (31.1%)
# 
# 🎯 Routing Stats
# Local: 847 requests (91.3%)
# OpenAI: 72 requests (7.8%)
# Anthropic: 9 requests (0.9%)
```

### 5.4 Privacy Controls

```bash
# Set privacy mode
zkie privacy mode full     # Air-gap (no connections)
zkie privacy mode balanced  # Recommended (safe defaults)
zkie privacy mode selective # Ask every time
zkie privacy mode cloud    # Allow all

# View audit log
zkie privacy audit --last 7days

# Export compliance report
zkie privacy report --start 2025-01-01 --end 2025-12-31 --output report.pdf
```

**Privacy Modes Explained:**

| Mode | Model Downloads | MCP Tools | API Calls | Code Updates |
|------|----------------|-----------|-----------|--------------|
| **Full Privacy** | ❌ Blocked | ❌ Blocked | ❌ Blocked | ❌ Blocked |
| **Balanced** | ✅ Allowed (ask first) | ✅ Allowed (ask first) | ❌ Blocked | ❌ Blocked |
| **Selective** | ⚠️ Ask every time | ⚠️ Ask every time | ⚠️ Ask every time | ❌ Blocked |
| **Cloud-First** | ✅ Allowed | ✅ Allowed | ✅ Allowed | ⚠️ Ask first |

---

## 6. Configuration

### Config File Location
`~/.zkie/config.json`

### Example Config

```json
{
  "model": {
    "auto_select": true,
    "default": "llama-3.1-70b-q4",
    "context_window": 4096
  },
  "privacy": {
    "mode": "balanced",
    "allowed_domains": ["huggingface.co", "github.com"]
  },
  "api": {
    "enabled": true,
    "threshold_percent": 15,
    "monthly_budget": 50.0,
    "providers": {
      "openai": {
        "enabled": true,
        "models": ["gpt-4-turbo", "gpt-3.5-turbo"]
      }
    }
  },
  "mcp": {
    "servers": [
      {
        "name": "asana",
        "url": "https://mcp.asana.com/sse",
        "enabled": true
      }
    ]
  },
  "ui": {
    "port": 7860,
    "theme": "soft"
  }
}
```

### Environment Variables

```bash
# Core settings
export ZKIE_HOME="~/.zkie"
export ZKIE_MODEL_CACHE="/mnt/ssd/models"

# Privacy
export ZKIE_PRIVACY_MODE="balanced"
export ZKIE_AUTO_UPDATE="false"  # Disable self-updates

# API keys (more secure than config file)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## 7. Privacy & Security

### Data Handling

**What ZKIE stores locally:**
- ✅ Chat history (encrypted, opt-in)
- ✅ Model cache
- ✅ Configuration
- ✅ Audit logs

**What ZKIE NEVER sends:**
- ❌ Your conversations (unless you explicitly use API providers)
- ❌ System information
- ❌ Telemetry
- ❌ Analytics

### Offline Air-Gap Mode

For maximum privacy (HIPAA, SOX, classified environments):

```bash
# Enable full privacy mode
zkie privacy mode full

# Create offline update bundle (on connected machine)
zkie offline create-bundle --output zkie-update-v1.1.0.bundle

# Transfer bundle via USB to air-gapped machine

# Apply update (on air-gapped machine)
zkie offline apply-bundle zkie-update-v1.1.0.bundle

# Verify no connections
zkie doctor --check-offline
# Expected: ✓ No internet connections detected
```

### Security Features

- ✅ **Ed25519 code signing** — All updates cryptographically signed
- ✅ **Audit logging** — Every connection attempt logged
- ✅ **GGUF bounds checking** — Prevents malicious model files
- ✅ **Atomic file writes** — Crash-safe configuration
- ✅ **No telemetry** — Zero data collection

---

## 8. API Reference

### REST API Endpoints

#### POST /v1/chat/completions
OpenAI-compatible chat endpoint.

**Request:**
```json
{
  "model": "auto",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "stream": true,
  "max_tokens": 512,
  "temperature": 0.7
}
```

**Response (streaming):**
```
data: {"id":"chatcmpl-zkie","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}

data: {"id":"chatcmpl-zkie","object":"chat.completion.chunk","choices":[{"delta":{"content":"!"}}]}

data: [DONE]
```

#### GET /v1/models
List available models.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama-3.1-70b-q4",
      "object": "model",
      "owned_by": "zkie",
      "size": 38.5
    }
  ]
}
```

### Python API

```python
from zkie import ZKIE

# Initialize
engine = ZKIE(
    model="auto",          # Auto-select or specify model
    privacy_mode="balanced",
    enable_mcp=True,
    enable_api=False
)

# Load model
engine.load_model("TheBloke/Llama-2-13B-GGUF/llama-2-13b.Q4_K_M.gguf")

# Chat (streaming)
for chunk in engine.chat("Write a Python sorting function"):
    print(chunk, end="", flush=True)

# Chat (non-streaming)
response = engine.chat("What is the capital of France?", stream=False)

# Chat with tools (MCP)
response = engine.chat(
    "Create a GitHub issue titled 'Fix bug in authentication'",
    tools_enabled=True
)

# Get conversation history
history = engine.get_history()

# Clear conversation
engine.clear_history()

# Model info
info = engine.get_model_info()
# Returns: {"name": "llama-2-13b-q4", "size_gb": 7.2, "context": 4096}
```

---

## 9. Troubleshooting

### Common Issues

#### Issue: "CUDA out of memory"
**Solution:**
```bash
# Check VRAM usage
nvidia-smi

# Switch to smaller model
zkie models download "TheBloke/Llama-2-7B-GGUF" "llama-2-7b.Q4_K_M.gguf"
zkie config set model.default "llama-2-7b-q4"

# Or use CPU-only
zkie config set model.gpu_layers 0
```

#### Issue: "Model download slow"
**Solution:**
```bash
# Use HuggingFace CLI (supports resume)
huggingface-cli download TheBloke/Llama-2-13B-GGUF llama-2-13b.Q4_K_M.gguf \
  --local-dir ~/.zkie/models/

# Then link in ZKIE
zkie models link ~/.zkie/models/llama-2-13b.Q4_K_M.gguf
```

#### Issue: "MCP server connection timeout"
**Solution:**
```bash
# Check server status
curl https://mcp.asana.com/sse

# Increase timeout
zkie mcp config --timeout 60

# Check firewall
sudo ufw allow out to any port 443
```

### Debug Mode

```bash
# Enable verbose logging
zkie --debug ui

# View logs
tail -f ~/.zkie/logs/zkie.log

# Check system info
zkie doctor --verbose
```

### Getting Help

- 📖 Documentation: https://docs.zkie.ai
- 💬 Discord: https://discord.gg/zkie
- 🐛 Issues: https://github.com/yourusername/zkie/issues
- ✉️ Email: support@zkie.ai

---

## 10. FAQ

**Q: Is ZKIE really free?**  
A: Yes! ZKIE is 100% open-source (MIT license). You only pay for optional API providers if you choose to use them.

**Q: How does the 15% rule work?**  
A: ZKIE benchmarks your local model, then only uses paid APIs if they're 15%+ better quality. This ensures you only pay for meaningful improvements.

**Q: Can I use ZKIE offline?**  
A: Absolutely! Set `zkie privacy mode full` for complete air-gap operation.

**Q: Does ZKIE phone home?**  
A: Never. Zero telemetry, zero analytics, zero data collection.

**Q: Can I use my own models?**  
A: Yes! ZKIE supports any GGUF model. Just download and load it.

**Q: How does ZKIE compare to ChatGPT?**  
A: ZKIE runs models locally (private, free after setup), while ChatGPT is cloud-based (costs per use). Quality depends on the model you run.

**Q: What's the catch?**  
A: No catch! We built ZKIE because we believe AI should be private, free, and under your control.

---

## Appendix A: Model Recommendations

### By Hardware

| Hardware | Model | VRAM | Quality | Speed |
|----------|-------|------|---------|-------|
| Celeron, 8GB RAM | TinyLlama-1.1B-Q5 | 0GB | ⭐⭐ | ⚡⚡⚡ |
| i5 + GTX 1660 (6GB) | Llama-2-7B-Q4 | 4GB | ⭐⭐⭐ | ⚡⚡⚡ |
| i7 + RTX 3060 (12GB) | Llama-3.1-8B-Q4 | 5GB | ⭐⭐⭐⭐ | ⚡⚡⚡ |
| Ryzen 9 + RTX 4080 (16GB) | CodeLlama-34B-Q4 | 20GB | ⭐⭐⭐⭐ | ⚡⚡ |
| Threadripper + RTX 4090 (24GB) | Llama-3.1-70B-Q4 | 40GB | ⭐⭐⭐⭐⭐ | ⚡⚡ |

### By Task

| Task | Model | Why |
|------|-------|-----|
| **Coding** | CodeLlama-34B-Q4 | Trained on code, function calling |
| **Creative Writing** | Llama-3.1-70B-Q4 | Large context, creative |
| **Research** | Mixtral-8x7B-Q4 | Expert routing, factual |
| **Chat** | Llama-2-13B-Chat-Q4 | Conversational tuning |
| **Low VRAM** | Phi-3-medium-Q5 | Efficient, good quality |

---

*Manual version 1.0 — Last updated: 2025-12-31*
```

### Task 7.3: Master Zip Creation

```bash
# Create distribution structure
mkdir -p zkie-v1.0.0/
cp -r zkie/ zkie-v1.0.0/
cp -r tests/ zkie-v1.0.0/
cp README.md LICENSE CHANGELOG.md setup.py requirements.txt zkie-v1.0.0/

# Generate file tree
python zkie/tools/generate_tree.py

# Create master zip
zip -r zkie-v1.0.0-master.zip zkie-v1.0.0/ \
    -x "*.pyc" -x "__pycache__/*" -x ".git/*"

# Verify zip
unzip -l zkie-v1.0.0-master.zip

# Expected size: 150-200 KB (source code only)
```

### Task 7.4: Installation Scripts

**install.sh (Linux/Mac):**
```bash
#!/bin/bash
set -e

echo "🚀 Installing Zenith Kernel Inference Engine..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ $(echo "$PYTHON_VERSION < 3.10" | bc) -eq 1 ]]; then
    echo "❌ Python $PYTHON_VERSION found, but 3.10+ required"
    exit 1
fi

echo "✓ Python $PYTHON_VERSION found"

# Create virtual environment
python3 -m venv zkie-venv
source zkie-venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install ZKIE
pip install -e .

# Verify installation
if zkie --version &> /dev/null; then
    echo "✅ ZKIE installed successfully!"
    echo ""
    echo "Quick start:"
    echo "  1. Activate environment: source zkie-venv/bin/activate"
    echo "  2. Launch UI: zkie ui"
    echo "  3. Or start API: zkie serve"
else
    echo "❌ Installation failed"
    exit 1
fi
```

### Stage 7 Deliverables
- ✅ `zkie-v1.0.0-master.zip` (master distribution)
- ✅ `ZKIE_FILE_TREE.txt` (file tree diagram)
- ✅ `USER_MANUAL.md` (comprehensive user manual)
- ✅ `install.sh` + `install.bat` (installation scripts)
- ✅ `README.md` (project overview)
- ✅ `CHANGELOG.md` (version history)
- ✅ `CONTRIBUTING.md` (developer guide)

---

## ✅ FINAL CHECKLIST

- [ ] All 47 critical bugs fixed
- [ ] Inference engine working (llama.cpp integrated)
- [ ] MCP server support implemented
- [ ] API routing with 15% threshold
- [ ] Privacy/offline mode complete
- [ ] Gradio UI functional
- [ ] FastAPI server operational
- [ ] All 78 tests passing
- [ ] Code coverage >90%
- [ ] User manual complete
- [ ] Master zip created
- [ ] File tree generated
- [ ] Installation scripts tested

---

## 📊 CREDIT USAGE ESTIMATE

| Stage | Tasks | Est. Credits | Actual | Variance |
|-------|-------|--------------|--------|----------|
| 1: Bug Fixes | 3 files | 200 | - | - |
| 2: Core Engine | 5 modules | 400 | - | - |
| 3: Plugins | 12 modules | 400 | - | - |
| 4: UI | 2 apps | 300 | - | - |
| 5: Privacy | 4 modules | 300 | - | - |
| 6: Testing | 78 tests | 400 | - | - |
| 7: Packaging | Docs + zip | 300 | - | - |
| **Total** | **7 stages** | **2,300** | - | - |
| Buffer (15%) | - | 500 | - | - |
| **Grand Total** | - | **2,800** | - | - |

---

## 🎯 SUCCESS METRICS

**Technical:**
- ✅ Zero memory leaks (valgrind clean)
- ✅ <2s startup time
- ✅ <100ms first token latency
- ✅ 95%+ test coverage
- ✅ All critical bugs sealed

**Functional:**
- ✅ Can load and run any GGUF model
- ✅ Streams tokens in real-time
- ✅ Connects to MCP servers
- ✅ Routes to APIs when >15% better
- ✅ Works 100% offline

**User Experience:**
- ✅ Installs in <5 minutes
- ✅ Works out-of-box (auto hardware detection)
- ✅ Clear error messages
- ✅ Comprehensive documentation
- ✅ Beautiful web interface

---

## 🚀 DEPLOYMENT

Once all stages complete:

```bash
# Final verification
cd zkie-v1.0.0/
./install.sh

# Run full test suite
pytest tests/ -v

# Launch and verify
zkie ui

# Create GitHub release
gh release create v1.0.0 \
    zkie-v1.0.0-master.zip \
    --title "Zenith Kernel v1.0.0 — First Release" \
    --notes "$(cat CHANGELOG.md)"
```

---

## 📝 MANUS EXECUTION NOTES

**To Manus:** Execute this prompt in the order specified. Each stage depends on previous stages completing successfully. Use maximum parallelism where indicated (stages 2-5). Document all issues encountered. Create refinement plans for any failures. Fix dependencies as you go. Ensure hermetic sealing at each stage.

**Expected Timeline:**
- Stages 1-5 (parallel): 2-3 days
- Stage 6 (testing + refinement): 1-2 days
- Stage 7 (packaging): 1 day
- **Total: 4-6 days**

**Credits Budget:** 2,800 (with 500 buffer)

---

*End of Master Prompt*
