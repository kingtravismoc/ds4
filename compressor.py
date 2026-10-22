--- compressor.py (原始)


+++ compressor.py (修改后)
#!/usr/bin/env python3
"""
DeepSeek Model Compressor & Data Ingestion System
-------------------------------------------------
Combines:
  1. Statistical range compression (block‑wise min‑max quantization)
  2. Geometric folding + zlib compression
  3. Custom container format (DSZ) for vector‑storage neural arrays
  4. Data ingestion pipeline for continuous learning

Usage:
  python compressor.py compress <input_model.safetensors> <output.dsz> [--bits 4] [--block-size 256]
  python compressor.py decompress <input.dsz> <output.safetensors>
  python compressor.py ingest <data_dir> --output <ingested.dsz>
"""

import argparse
import struct
import zlib
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional
import sys
import json
import hashlib
import time
from datetime import datetime
import os

# ----------------------------------------------------------------------
# 1. Geometric folding compression / decompression
# ----------------------------------------------------------------------
def geometric_fold_compress(data: bytes) -> bytes:
    """Apply geometric folding (2D quadrant averaging) + zlib."""
    if len(data) == 0:
        return zlib.compress(b'', level=6)

    length = len(data)
    size = int(np.ceil(np.sqrt(length)))
    total = size * size
    matrix = bytearray(total)
    matrix[:length] = data

    half = size // 2
    if half == 0:
        return zlib.compress(bytes(matrix[:half*half]), level=6)

    folded = bytearray(half * half)
    for i in range(half):
        for j in range(half):
            a = matrix[i * size + j]
            b = matrix[(size - 1 - i) * size + j] if (size - 1 - i) < size else 0
            c = matrix[i * size + (size - 1 - j)] if (size - 1 - j) < size else 0
            d = matrix[(size - 1 - i) * size + (size - 1 - j)] if ((size - 1 - i) < size and (size - 1 - j) < size) else 0
            folded[i * half + j] = (a + b + c + d) // 4

    return zlib.compress(folded, level=6)

def geometric_fold_decompress(compressed: bytes) -> bytes:
    """Decompress geometric folding: inflate + unfold."""
    if len(compressed) == 0:
        return b''

    folded = zlib.decompress(compressed)
    half = int(np.sqrt(len(folded)))
    if half == 0:
        return b''

    full_size = half * 2
    full = bytearray(full_size * full_size)

    for i in range(half):
        for j in range(half):
            val = folded[i * half + j]
            full[i * full_size + j] = val
            if (full_size - 1 - i) < full_size:
                full[(full_size - 1 - i) * full_size + j] = val
            if (full_size - 1 - j) < full_size:
                full[i * full_size + (full_size - 1 - j)] = val
            if (full_size - 1 - i) < full_size and (full_size - 1 - j) < full_size:
                full[(full_size - 1 - i) * full_size + (full_size - 1 - j)] = val

    return bytes(full)

# ----------------------------------------------------------------------
# 2. Statistical range compression (block‑wise min‑max quantization)
# ----------------------------------------------------------------------
def quantize_block(block: np.ndarray, bits: int) -> Tuple[bytes, float, float]:
    """Quantize a 1D block of float32 values to `bits` bits."""
    min_val = float(block.min())
    max_val = float(block.max())

    if max_val - min_val < 1e-8:
        packed = bytes([0] * ((len(block) * bits + 7) // 8))
        return packed, min_val, 0.0

    max_q = (1 << bits) - 1
    scale = (max_val - min_val) / max_q
    quantized = np.round((block - min_val) / scale).astype(np.uint8)

    packed = []
    bit_buffer = 0
    bits_in_buffer = 0
    for q in quantized:
        bit_buffer |= (int(q) << bits_in_buffer)
        bits_in_buffer += bits
        while bits_in_buffer >= 8:
            packed.append(bit_buffer & 0xFF)
            bit_buffer >>= 8
            bits_in_buffer -= 8
    if bits_in_buffer > 0:
        packed.append(bit_buffer & 0xFF)

    return bytes(packed), min_val, scale

def dequantize_block(packed: bytes, min_val: float, scale: float, bits: int, num_vals: int) -> np.ndarray:
    """Restore a block from packed bytes and statistics."""
    if scale == 0.0:
        return np.full(num_vals, min_val, dtype=np.float32)

    max_q = (1 << bits) - 1
    quantized = np.zeros(num_vals, dtype=np.uint8)
    bit_pos = 0

    for i in range(num_vals):
        byte_idx = bit_pos // 8
        shift = bit_pos % 8

        if byte_idx >= len(packed):
            break

        q_val = (packed[byte_idx] >> shift) & max_q
        if shift + bits > 8 and byte_idx + 1 < len(packed):
            q_val |= (packed[byte_idx + 1] << (8 - shift)) & max_q

        quantized[i] = q_val
        bit_pos += bits

    return (min_val + quantized.astype(np.float32) * scale).astype(np.float32)

def compress_tensor_statistical(tensor: np.ndarray, bits: int = 4, block_size: int = 256) -> Tuple[bytes, List[float], List[float]]:
    """Apply statistical range compression to a whole tensor."""
    flat = tensor.flatten().astype(np.float32)
    n = flat.size
    num_blocks = (n + block_size - 1) // block_size
    packed_parts = []
    min_vals = []
    scales = []

    for blk in range(num_blocks):
        start = blk * block_size
        end = min(start + block_size, n)
        block = flat[start:end]
        p, mn, sc = quantize_block(block, bits)
        packed_parts.append(p)
        min_vals.append(mn)
        scales.append(sc)

    return b''.join(packed_parts), min_vals, scales

def decompress_tensor_statistical(packed: bytes, min_vals: List[float], scales: List[float],
                                  bits: int, block_size: int, orig_len: int) -> np.ndarray:
    """Restore the original flattened tensor."""
    result = np.empty(orig_len, dtype=np.float32)
    pos = 0
    block_start = 0

    for mn, sc in zip(min_vals, scales):
        block_end = min(block_start + block_size, orig_len)
        num_vals = block_end - block_start
        bytes_needed = (num_vals * bits + 7) // 8

        if pos + bytes_needed > len(packed):
            bytes_needed = len(packed) - pos

        block_packed = packed[pos:pos + bytes_needed]
        block_data = dequantize_block(block_packed, mn, sc, bits, num_vals)
        result[block_start:block_end] = block_data

        block_start = block_end
        pos += bytes_needed

    return result

# ----------------------------------------------------------------------
# 3. Tensor extraction and saving
# ----------------------------------------------------------------------
def extract_tensors(model_path: str) -> Dict[str, np.ndarray]:
    """Load all tensors as float32 numpy arrays."""
    tensors = {}
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_path.endswith('.safetensors'):
        with safe_open(model_path, framework="np") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    elif model_path.endswith('.bin') or model_path.endswith('.pt'):
        state = torch.load(model_path, map_location='cpu', weights_only=True)
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                tensors[key] = value.numpy().astype(np.float32)
    elif model_path.endswith('.dsz'):
        return load_compressed_model(model_path)
    else:
        raise ValueError(f"Unsupported file type: {model_path}. Use .safetensors, .bin, .pt, or .dsz")

    return tensors

def save_tensors_as_safetensors(tensors: Dict[str, np.ndarray], output_path: str):
    """Save a dict of numpy arrays as a safetensors file."""
    torch_dict = {k: torch.from_numpy(v) for k, v in tensors.items()}
    save_file(torch_dict, output_path)

# ----------------------------------------------------------------------
# 4. Custom DSZ container format
# ----------------------------------------------------------------------
def save_compressed_model(tensors: Dict[str, np.ndarray], output_path: str,
                          bits: int = 4, block_size: int = 256, metadata: Optional[Dict] = None):
    """Compress all tensors and write to DSZ file."""
    with open(output_path, 'wb') as f:
        f.write(b'DSZS')
        f.write(struct.pack('<I', len(tensors)))

        global_metadata = metadata or {}
        meta_json = json.dumps(global_metadata).encode('utf-8')
        f.write(struct.pack('<I', len(meta_json)))
        f.write(meta_json)

        for name, tensor in tensors.items():
            print(f"Compressing {name} shape {tensor.shape} ...")
            quant_packed, min_vals, scales = compress_tensor_statistical(tensor, bits, block_size)
            final_compressed = geometric_fold_compress(quant_packed)

            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)

            shape = tensor.shape
            f.write(struct.pack('<I', len(shape)))
            for dim in shape:
                f.write(struct.pack('<Q', int(dim)))

            f.write(struct.pack('<B', bits))
            f.write(struct.pack('<I', block_size))
            f.write(struct.pack('<I', len(min_vals)))

            for mn, sc in zip(min_vals, scales):
                f.write(struct.pack('<dd', float(mn), float(sc)))

            f.write(struct.pack('<I', len(final_compressed)))
            f.write(final_compressed)

    orig_size = sum(t.nbytes for t in tensors.values())
    comp_size = Path(output_path).stat().st_size
    ratio = comp_size / orig_size if orig_size > 0 else 0

    print(f"\nCompression complete:")
    print(f"  Original size: {orig_size / 1e9:.2f} GB")
    print(f"  Compressed size: {comp_size / 1e6:.2f} MB")
    print(f"  Compression ratio: {ratio:.1%}")

    return {'original_size': orig_size, 'compressed_size': comp_size, 'ratio': ratio}

def load_compressed_model(input_path: str) -> Dict[str, np.ndarray]:
    """Load a DSZ file and restore all tensors."""
    tensors = {}

    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'DSZS':
            raise ValueError("Not a valid DSZ file")

        num_tensors = struct.unpack('<I', f.read(4))[0]

        meta_len = struct.unpack('<I', f.read(4))[0]
        metadata = json.loads(f.read(meta_len).decode('utf-8')) if meta_len > 0 else {}

        print(f"Loading {num_tensors} tensors from {input_path}")
        if metadata:
            print(f"Metadata: {metadata}")

        for _ in range(num_tensors):
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')

            ndim = struct.unpack('<I', f.read(4))[0]
            shape = tuple(struct.unpack('<Q', f.read(8))[0] for _ in range(ndim))
            orig_len = int(np.prod(shape))

            bits = struct.unpack('<B', f.read(1))[0]
            block_size = struct.unpack('<I', f.read(4))[0]

            num_blocks = struct.unpack('<I', f.read(4))[0]
            min_vals = []
            scales = []
            for _ in range(num_blocks):
                mn, sc = struct.unpack('<dd', f.read(16))
                min_vals.append(mn)
                scales.append(sc)

            comp_len = struct.unpack('<I', f.read(4))[0]
            comp_data = f.read(comp_len)

            quant_packed = geometric_fold_decompress(comp_data)
            flat = decompress_tensor_statistical(quant_packed, min_vals, scales, bits, block_size, orig_len)
            tensors[name] = flat.reshape(shape)

    return tensors

# ----------------------------------------------------------------------
# 5. Data Ingestion Pipeline
# ----------------------------------------------------------------------
class DataIngestor:
    """Handles data ingestion, validation, and preparation for model training."""

    def __init__(self, ingest_dir: str = "./ingested_data"):
        self.ingest_dir = Path(ingest_dir)
        self.ingest_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.ingest_dir / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict:
        """Load or create ingestion manifest."""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "datasets": [],
            "total_samples": 0,
            "total_size_bytes": 0
        }

    def _save_manifest(self):
        """Save manifest to disk."""
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)

    def ingest_directory(self, data_dir: str, dataset_name: str, file_patterns: List[str] = None) -> Dict:
        """Ingest all matching files from a directory."""
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        patterns = file_patterns or ['*.json', '*.txt', '*.csv', '*.parquet']
        ingested_files = []
        total_samples = 0
        total_size = 0

        for pattern in patterns:
            for file_path in data_path.glob(pattern):
                print(f"Ingesting: {file_path}")
                file_data = self._process_file(file_path, dataset_name)
                if file_data:
                    ingested_files.append(file_data)
                    total_samples += file_data.get('samples', 0)
                    total_size += file_path.stat().st_size

        dataset_entry = {
            "name": dataset_name,
            "source_dir": str(data_path),
            "ingested_at": datetime.now().isoformat(),
            "files": ingested_files,
            "total_samples": total_samples,
            "total_size_bytes": total_size,
            "checksum": hashlib.md5(str(total_samples).encode()).hexdigest()
        }

        self.manifest["datasets"].append(dataset_entry)
        self.manifest["total_samples"] += total_samples
        self.manifest["total_size_bytes"] += total_size
        self._save_manifest()

        print(f"\nIngestion complete for '{dataset_name}':")
        print(f"  Files processed: {len(ingested_files)}")
        print(f"  Total samples: {total_samples}")
        print(f"  Total size: {total_size / 1e6:.2f} MB")

        return dataset_entry

    def _process_file(self, file_path: Path, dataset_name: str) -> Optional[Dict]:
        """Process a single file based on its extension."""
        ext = file_path.suffix.lower()

        try:
            if ext == '.json':
                return self._process_json(file_path, dataset_name)
            elif ext == '.txt':
                return self._process_text(file_path, dataset_name)
            elif ext == '.csv':
                return self._process_csv(file_path, dataset_name)
            elif ext == '.parquet':
                return self._process_parquet(file_path, dataset_name)
            else:
                print(f"  Skipping unsupported format: {ext}")
                return None
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            return None

    def _process_json(self, file_path: Path, dataset_name: str) -> Dict:
        """Process JSON file (expecting list of records or dict with 'data' key)."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            samples = len(data)
        elif isinstance(data, dict) and 'data' in data:
            samples = len(data['data']) if isinstance(data['data'], list) else 1
        else:
            samples = 1

        # Save processed data
        output_path = self.ingest_dir / f"{dataset_name}_{file_path.stem}.processed.json"
        with open(output_path, 'w') as f:
            json.dump(data, f)

        return {
            "file": str(file_path),
            "processed_file": str(output_path),
            "samples": samples,
            "size_bytes": file_path.stat().st_size,
            "checksum": hashlib.md5(open(file_path, 'rb').read()).hexdigest()
        }

    def _process_text(self, file_path: Path, dataset_name: str) -> Dict:
        """Process text file (line-by-line or full document)."""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        samples = len([l for l in lines if l.strip()])

        output_path = self.ingest_dir / f"{dataset_name}_{file_path.stem}.processed.txt"
        with open(output_path, 'w') as f:
            f.writelines(lines)

        return {
            "file": str(file_path),
            "processed_file": str(output_path),
            "samples": samples,
            "size_bytes": file_path.stat().st_size,
            "checksum": hashlib.md5(open(file_path, 'rb').read()).hexdigest()
        }

    def _process_csv(self, file_path: Path, dataset_name: str) -> Dict:
        """Process CSV file."""
        import csv
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        samples = len(rows) - 1 if len(rows) > 1 else 0

        output_path = self.ingest_dir / f"{dataset_name}_{file_path.stem}.processed.csv"
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        return {
            "file": str(file_path),
            "processed_file": str(output_path),
            "samples": samples,
            "size_bytes": file_path.stat().st_size,
            "checksum": hashlib.md5(open(file_path, 'rb').read()).hexdigest()
        }

    def _process_parquet(self, file_path: Path, dataset_name: str) -> Dict:
        """Process Parquet file."""
        try:
            import pandas as pd
            df = pd.read_parquet(file_path)
            samples = len(df)

            output_path = self.ingest_dir / f"{dataset_name}_{file_path.stem}.processed.parquet"
            df.to_parquet(output_path)

            return {
                "file": str(file_path),
                "processed_file": str(output_path),
                "samples": samples,
                "size_bytes": file_path.stat().st_size,
                "checksum": hashlib.md5(open(file_path, 'rb').read()).hexdigest()
            }
        except ImportError:
            print("  pandas/pyarrow not installed, skipping parquet file")
            return None

    def get_manifest(self) -> Dict:
        """Return current manifest."""
        return self.manifest

    def create_training_tensor(self, tensor_name: str = "ingested_data") -> np.ndarray:
        """Convert ingested data into a tensor for model training."""
        all_data = []

        for dataset in self.manifest.get("datasets", []):
            for file_info in dataset.get("files", []):
                processed_file = Path(file_info.get("processed_file", ""))
                if processed_file.exists():
                    ext = processed_file.suffix.lower()
                    try:
                        if ext == '.json':
                            with open(processed_file, 'r') as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    all_data.extend(data)
                        elif ext == '.txt':
                            with open(processed_file, 'r') as f:
                                lines = f.readlines()
                                all_data.extend([l.strip() for l in lines if l.strip()])
                        elif ext == '.csv':
                            import pandas as pd
                            df = pd.read_csv(processed_file)
                            all_data.extend(df.values.tolist())
                    except Exception as e:
                        print(f"Error reading {processed_file}: {e}")

        if not all_data:
            raise ValueError("No data available for tensor creation")

        # Convert to numerical tensor (simple embedding approach)
        # In production, this would use proper tokenization/embedding
        flat_data = []
        for item in all_data:
            if isinstance(item, (int, float)):
                flat_data.append(float(item))
            elif isinstance(item, str):
                # Simple character-level encoding
                flat_data.extend([ord(c) / 256.0 for c in item[:100]])
            elif isinstance(item, (list, tuple)):
                for subitem in item:
                    if isinstance(subitem, (int, float)):
                        flat_data.append(float(subitem))

        if not flat_data:
            raise ValueError("Could not convert data to numerical format")

        tensor = np.array(flat_data, dtype=np.float32)
        return tensor

# ----------------------------------------------------------------------
# 6. Command line interface
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DeepSeek Model Compressor & Data Ingestor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Compress command
    compress_parser = subparsers.add_parser("compress", help="Compress a model")
    compress_parser.add_argument("input", help="Input model file")
    compress_parser.add_argument("output", help="Output compressed file (.dsz)")
    compress_parser.add_argument("--bits", type=int, default=4, choices=range(1,9),
                                 help="Quantization bits (default 4)")
    compress_parser.add_argument("--block-size", type=int, default=256,
                                 help="Block size (default 256)")
    compress_parser.add_argument("--metadata", type=str, help="JSON metadata string")

    # Decompress command
    decompress_parser = subparsers.add_parser("decompress", help="Decompress a DSZ file")
    decompress_parser.add_argument("input", help="Compressed .dsz file")
    decompress_parser.add_argument("output", help="Output file (.safetensors)")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest data directory")
    ingest_parser.add_argument("data_dir", help="Directory containing data files")
    ingest_parser.add_argument("--dataset-name", required=True, help="Name for this dataset")
    ingest_parser.add_argument("--output", help="Output DSZ file with ingested data as tensor")
    ingest_parser.add_argument("--patterns", nargs='+', help="File patterns to match")

    # Manifest command
    manifest_parser = subparsers.add_parser("manifest", help="Show ingestion manifest")
    manifest_parser.add_argument("--ingest-dir", default="./ingested_data", help="Ingestion directory")

    args = parser.parse_args()

    if args.command == "compress":
        print(f"Extracting tensors from {args.input}...")
        tensors = extract_tensors(args.input)
        print(f"Found {len(tensors)} tensors")

        metadata = json.loads(args.metadata) if args.metadata else None
        save_compressed_model(tensors, args.output, bits=args.bits,
                             block_size=args.block_size, metadata=metadata)

    elif args.command == "decompress":
        print(f"Loading compressed model from {args.input}...")
        tensors = load_compressed_model(args.input)
        print(f"Restored {len(tensors)} tensors")
        save_tensors_as_safetensors(tensors, args.output)
        print(f"Saved to {args.output}")

    elif args.command == "ingest":
        ingestor = DataIngestor()
        result = ingestor.ingest_directory(
            args.data_dir,
            args.dataset_name,
            args.patterns
        )

        if args.output:
            print(f"\nCreating compressed tensor from ingested data...")
            tensor = ingestor.create_training_tensor()
            tensors = {"ingested_data": tensor}
            save_compressed_model(tensors, args.output, bits=4, block_size=256,
                                metadata={"source": "ingestion", "dataset": args.dataset_name})

    elif args.command == "manifest":
        ingestor = DataIngestor(args.ingest_dir)
        manifest = ingestor.get_manifest()
        print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
