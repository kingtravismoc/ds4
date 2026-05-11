/**
 * Statistical Range Compression Module
 * Compresses float32 arrays to uint16 with min/max range encoding
 */

import { RangeCompressed } from './types';
import { createGzip, createGunzip } from 'zlib';
import { pipeline } from 'stream/promises';

/**
 * Compress float32 array to 16-bit range-compressed format
 * Reduces data size by 50% (32-bit -> 16-bit)
 */
export function rangeCompress(src: Float32Array | number[]): RangeCompressed {
  const len = src.length;
  if (len === 0) {
    return {
      minVal: 0,
      maxVal: 0,
      compressedData: new Uint16Array(0),
      dataLen: 0,
      origLen: 0,
    };
  }

  // Find min and max values
  let minVal: number = src[0]!;
  let maxVal: number = src[0]!;
  for (let i = 1; i < len; i++) {
    const val = src[i]!;
    if (val < minVal) minVal = val;
    if (val > maxVal) maxVal = val;
  }

  // Calculate range with epsilon to avoid division by zero
  let range = maxVal - minVal;
  if (range < 1e-9) range = 1e-9;

  // Normalize and quantize to 16-bit
  const compressedData = new Uint16Array(len);
  for (let i = 0; i < len; i++) {
    const norm = (src[i]! - minVal) / range;
    compressedData[i] = Math.floor(norm * 65535);
  }

  return {
    minVal,
    maxVal,
    compressedData,
    dataLen: len,
    origLen: len,
  };
}

/**
 * Decompress 16-bit range-compressed data back to float32
 */
export function rangeDecompress(compressed: RangeCompressed): Float32Array {
  const len = compressed.dataLen;
  const dst = new Float32Array(len);

  let range = compressed.maxVal - compressed.minVal;
  if (range < 1e-9) range = 1e-9;

  for (let i = 0; i < len; i++) {
    const norm = compressed.compressedData[i]! / 65535;
    dst[i] = compressed.minVal + norm * range;
  }

  return dst;
}

/**
 * GZIP compress a string
 */
export async function gzipCompress(input: string): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    const gzip = createGzip({ level: 6 });

    gzip.on('data', (chunk) => chunks.push(chunk));
    gzip.on('end', () => resolve(Buffer.concat(chunks)));
    gzip.on('error', reject);

    gzip.end(Buffer.from(input, 'utf-8'));
  });
}

/**
 * GZIP decompress a buffer to string
 */
export async function gzipDecompress(input: Buffer): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    const gunzip = createGunzip();

    gunzip.on('data', (chunk) => chunks.push(chunk));
    gunzip.on('end', () => resolve(Buffer.concat(chunks).toString('utf-8')));
    gunzip.on('error', reject);

    gunzip.end(input);
  });
}

/**
 * Estimate compression ratio for range compression
 */
export function getCompressionRatio(original: number[], compressed: RangeCompressed): number {
  const originalBytes = original.length * 4; // float32 = 4 bytes
  const compressedBytes = compressed.dataLen * 2 + 8; // uint16 = 2 bytes + min/max (8 bytes)
  return originalBytes / compressedBytes;
}
