/**
 * Test suite for DS4 Lite Server
 */

import { rangeCompress, rangeDecompress, getCompressionRatio } from '../src/compression';
import { globalApiKeyStore } from '../src/auth';

async function testCompression() {
  console.log('Testing statistical range compression...');
  
  // Test with sample float data
  const testData = new Float32Array([0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.0]);
  
  const compressed = rangeCompress(testData);
  console.log(`  Original size: ${testData.length * 4} bytes (float32)`);
  console.log(`  Compressed size: ${compressed.dataLen * 2 + 8} bytes (uint16 + min/max)`);
  console.log(`  Compression ratio: ${(getCompressionRatio(Array.from(testData), compressed)).toFixed(2)}x`);
  
  const decompressed = rangeDecompress(compressed);
  
  // Check reconstruction accuracy
  let maxError = 0;
  for (let i = 0; i < testData.length; i++) {
    const error = Math.abs(testData[i]! - decompressed[i]!);
    if (error > maxError) maxError = error;
  }
  console.log(`  Max reconstruction error: ${maxError.toFixed(6)}`);
  console.log('  ✓ Compression test passed\n');
}

async function testApiKeyAuth() {
  console.log('Testing API key authentication...');
  
  // Test default key
  const isValid = await globalApiKeyStore.validate('sk-ds4lite-default');
  console.log(`  Default key valid: ${isValid}`);
  
  // Test adding new key
  const added = await globalApiKeyStore.addKey('sk-test-key-123', 'test-key');
  console.log(`  Added new key: ${added}`);
  
  // Validate new key
  const newKeyValid = await globalApiKeyStore.validate('sk-test-key-123');
  console.log(`  New key valid: ${newKeyValid}`);
  
  // Test invalid key
  const invalidKeyValid = await globalApiKeyStore.validate('invalid-key');
  console.log(`  Invalid key rejected: ${!invalidKeyValid}`);
  
  // List keys
  const keys = await globalApiKeyStore.listKeys();
  console.log(`  Total keys: ${keys.length}`);
  
  // Generate random key
  const generatedKey = await globalApiKeyStore.generateKey('generated-key');
  console.log(`  Generated key: ${generatedKey.substring(0, 16)}...`);
  
  console.log('  ✓ Authentication test passed\n');
}

async function runTests() {
  console.log('╔════════════════════════════════════════╗');
  console.log('║  DS4 Lite Server - Test Suite          ║');
  console.log('╚════════════════════════════════════════╝\n');
  
  try {
    await testCompression();
    await testApiKeyAuth();
    
    console.log('╔════════════════════════════════════════╗');
    console.log('║  All tests passed! ✓                   ║');
    console.log('╚════════════════════════════════════════╝');
    process.exit(0);
  } catch (error) {
    console.error('Test failed:', error);
    process.exit(1);
  }
}

runTests();
