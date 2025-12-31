#!/usr/bin/env python3
# vertex_core.py — CORRECTED VERSION
# 2025-12-31 vFinal.FIXED — critical bugs resolved
import os, json, hashlib, time, subprocess, platform, psutil, zstandard as zstd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.request import urlopen

VERSION = 'vertex-sealed.2'
REPO_PRIVATE = 'git+ssh://git@github.com/yourname/vertex-private.git'
REPO_PUBLIC = 'https://huggingface.co/yourname/vertex-core'

@dataclass
class HardwarePulse:
    flops_per_sec: float
    tensor_cores: bool
    ram_gb: float
    vram_gb: float  # FIX: Add VRAM tracking

class VertexCore:
    def __init__(self):
        self.root = Path(os.getenv('VERTEX_HOME', str(Path.home() / '.vertex')))
        self.root.mkdir(exist_ok=True)
        (self.root / 'cache').mkdir(exist_ok=True)
        self._disk_tiers = None  # Lazy load
        self._rules = None  # FIX: Make lazy
        self.pulse = self.take_pulse()

    @property
    def rules(self) -> Dict:
        """FIX: Convert to cached property"""
        if self._rules is None:
            self._rules = self.load_rules()
        return self._rules

    @property
    def disk_tiers(self) -> List[tuple]:
        if self._disk_tiers is None:
            disks = []
            for part in psutil.disk_partitions():
                # FIX: Use try/except instead of exists() check (TOCTOU race)
                try:
                    path = Path(part.mountpoint)
                    usage = psutil.disk_usage(part.mountpoint)
                    
                    # FIX: Read actual speed from /sys/block
                    speed = self._detect_disk_speed(part.device)
                    disks.append((path, speed, usage.total))
                except (PermissionError, OSError):
                    continue
            disks.sort(key=lambda x: -x[1])
            self._disk_tiers = disks
        return self._disk_tiers

    def _detect_disk_speed(self, device: str) -> int:
        """FIX: Proper disk speed detection using /sys/block"""
        try:
            # Extract block device name (e.g., /dev/nvme0n1 → nvme0n1)
            dev_name = device.split('/')[-1]
            # Handle partition numbers (nvme0n1p1 → nvme0n1)
            if dev_name[-1].isdigit():
                dev_name = dev_name.rstrip('0123456789').rstrip('p')
            
            sys_path = Path(f'/sys/block/{dev_name}')
            if not sys_path.exists():
                # Fallback to old heuristic
                if 'nvme' in device: return 900
                if 'sd' in device: return 550
                return 120
            
            # Read rotational flag (0 = SSD, 1 = HDD)
            rotational = (sys_path / 'queue/rotational').read_text().strip() == '1'
            if rotational:
                return 120  # HDD
            
            # For SSDs, check if NVMe
            if 'nvme' in dev_name or 'vd' in dev_name:
                return 900  # NVMe or virtio-blk (fast)
            
            return 550  # SATA SSD
        except (FileNotFoundError, PermissionError, OSError):
            # Fallback heuristic
            if 'nvme' in device or 'vd' in device or 'xvd' in device:
                return 900
            return 550

    def compress(self, data: bytes) -> bytes:
        return zstd.compress(data, level=9)

    def decompress(self, blob: bytes) -> bytes:
        return zstd.decompress(blob)

    def delta_check(self):
        """FIX: Use git fetch + compare instead of hash-then-pull race"""
        try:
            # Fetch without merging
            result = subprocess.run(
                ['git', 'fetch', 'origin', 'main'],
                cwd=self.root,
                capture_output=True,
                timeout=5,
                check=True
            )
            
            # Compare local vs origin/main
            local_rev = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.root,
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()
            
            remote_rev = subprocess.run(
                ['git', 'rev-parse', 'origin/main'],
                cwd=self.root,
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()
            
            if local_rev != remote_rev:
                print('△ New vertex detected — merging')
                subprocess.run(
                    ['git', 'merge', 'origin/main'],
                    cwd=self.root,
                    check=True
                )
        except subprocess.CalledProcessError as e:
            print(f'△ Delta check failed: {e}')
        except Exception:
            pass

    def take_pulse(self) -> HardwarePulse:
        """FIX: Larger benchmark with warmup and cleanup"""
        # FIX: Set CPU affinity for stable timing
        try:
            os.sched_setaffinity(0, {0})
        except:
            pass
        
        start = time.perf_counter_ns()
        vram_gb = 0.0
        
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # FIX: Track VRAM
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            # FIX: Larger tensor, warmup iterations
            x = torch.randn(2048, 2048, device=device)
            
            with torch.inference_mode():
                # 20 warmup iterations (discarded)
                for _ in range(20):
                    _ = x @ x
                
                # Measure 200 iterations
                start = time.perf_counter_ns()
                for _ in range(200):
                    _ = x @ x
                elapsed_ns = time.perf_counter_ns() - start
            
            # FIX: Correct FLOPS calculation (2*N^3 per matmul)
            flops = (2 * 2048**3 * 200) / (elapsed_ns / 1e9)
            tensor = bool(torch.cuda.is_available())
            
            # FIX: Explicit cleanup
            del x
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except ImportError:
            flops, tensor = 1e8, False
        
        return HardwarePulse(flops, tensor, psutil.virtual_memory().total/1e9, vram_gb)

    def select_model(self) -> str:
        """FIX: Use VRAM as primary selector, check CPU features"""
        # FIX: Check CPU features for quantized model support
        has_avx2 = False
        has_avx512 = False
        try:
            with open('/proc/cpuinfo', 'r') as f:
                flags = f.read()
                has_avx2 = 'avx2' in flags
                has_avx512 = 'avx512' in flags
        except:
            pass
        
        # Primary: Use VRAM if tensor cores available
        if self.pulse.tensor_cores:
            if self.pulse.vram_gb >= 40:
                return 'meta-llama/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf'
            elif self.pulse.vram_gb >= 16:
                return 'microsoft/Phi-3-medium-4k-instruct-q5_K_M.gguf'
            elif self.pulse.vram_gb >= 8:
                return 'TinyLlama/TinyLlama-1.1B-Chat-v1.0-Q5_K_M.gguf'
        
        # Fallback: CPU-based selection
        if has_avx512 and self.pulse.ram_gb >= 32:
            return 'microsoft/Phi-3-medium-4k-instruct-q5_K_M.gguf'
        elif has_avx2 and self.pulse.ram_gb >= 16:
            return 'TinyLlama/TinyLlama-1.1B-Chat-v1.0-Q5_K_M.gguf'
        
        return 'google/gemma-2b-it-Q4_K_M.gguf'

    def load_rules(self) -> Dict:
        rule_file = self.root / 'rules.json.zst'
        try:
            return json.loads(self.decompress(rule_file.read_bytes()))
        except (FileNotFoundError, OSError):
            # File doesn't exist or can't be read, create default
            pass
        
        default = {
            'compression': {'.json': 'zstd:9', '.bin': 'zstd:6', '.gguf': 'none'},
            'lazy_hotplug': True,
            'shadow_validate': True
        }
        rule_file.write_bytes(self.compress(json.dumps(default).encode()))
        return default

    def run(self):
        self.delta_check()
        model = self.select_model()
        print(f'Vertex awake — {platform.system()} {platform.machine()} — selected → {model.split("/")[-1]}')
        print(f'Hardware: {self.pulse.flops_per_sec/1e9:.1f} GFLOPS, '
              f'RAM: {self.pulse.ram_gb:.1f}GB, VRAM: {self.pulse.vram_gb:.1f}GB')
        print('Ready. Fire when you are.')

if __name__ == '__main__':
    VertexCore().run()
