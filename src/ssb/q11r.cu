// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <curand.h>

#include <cuda.h>
#include <cub/util_allocator.cuh>
#include <cub/cub.cuh>

// #include "cub/test/test_util.h"
#include "utils/gpu_utils.h"
#include "ssb_gpu_utils.h"
#include "econfig.h"

using namespace std;
using namespace cub;

/**
 * Globals, constants and typedefs
 */
bool                    g_verbose = false;  // Whether to display input/output to console
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

template<typename T>
T* loadToGPU(T* src, int numEntries, CachingDeviceAllocator& g_allocator) {
  T* dest;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&dest, sizeof(T) * numEntries));
  CubDebugExit(cudaMemcpy(dest, src, sizeof(T) * numEntries, cudaMemcpyHostToDevice));
  return dest;
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void QueryKernel(
    uint* lo_orderdate_val_block_start, uint* lo_orderdate_val_data,
    uint* lo_orderdate_rl_block_start, uint* lo_orderdate_rl_data,
    uint* lo_discount_block_start, uint* lo_discount_data,
    uint* lo_quantity_block_start, uint* lo_quantity_data,
    uint* lo_extendedprice_block_start, uint* lo_extendedprice_data,
    int lo_num_entries, unsigned long long* revenue) {
  typedef cub::BlockReduce<int, BLOCK_THREADS> BlockReduceInt;

  int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
  int tile_idx = blockIdx.x;    // Current tile index
  int tile_offset = tile_idx * tile_size;

  // Allocate shared memory for BlockLoad
  __shared__ union TempStorage
  {
    typename BlockReduceInt::TempStorage reduce;
    uint shared_buffer[BLOCK_THREADS * ITEMS_PER_THREAD * 2 + 128];
  } temp_storage;

  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];

  long long sum = 0;

  int num_tiles = (lo_num_entries + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  bool is_last_tile = false;
  if (tile_idx == num_tiles - 1) {
    num_tile_items = lo_num_entries - tile_offset;
    is_last_tile = true;
  }

    RENCODINGKERNEL<BLOCK_THREADS,ITEMS_PER_THREAD>(
        lo_orderdate_val_block_start, lo_orderdate_rl_block_start, lo_orderdate_val_data, lo_orderdate_rl_data,
        temp_storage.shared_buffer, items, items2, is_last_tile, num_tile_items);

  // Barrier for smem reuse
  __syncthreads();


  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    // Out-of-bounds items are selection_flags
    selection_flags[ITEM] = 1;

    if (!is_last_tile || (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items))
      selection_flags[ITEM] = (items[ITEM] > 19930000 && items[ITEM] < 19940000); 
  }

  __syncthreads();

  ENCODINGKERNEL<BLOCK_THREADS,ITEMS_PER_THREAD>(lo_quantity_block_start, lo_quantity_data, temp_storage.shared_buffer, items, is_last_tile, num_tile_items);

  // Barrier for smem reuse
  __syncthreads();

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (!is_last_tile || (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items))
      selection_flags[ITEM] = selection_flags[ITEM] && items[ITEM] < 25;
  }

  __syncthreads();

  ENCODINGKERNEL<BLOCK_THREADS,ITEMS_PER_THREAD>(lo_discount_block_start, lo_discount_data, temp_storage.shared_buffer, items, is_last_tile, num_tile_items);

  // Barrier for smem reuse
  __syncthreads();

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (!is_last_tile || (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items))
      selection_flags[ITEM] = selection_flags[ITEM] && items[ITEM] >= 1 && items[ITEM ] <= 3;
  }

  __syncthreads();

  ENCODINGKERNEL<BLOCK_THREADS,ITEMS_PER_THREAD>(lo_extendedprice_block_start, lo_extendedprice_data, temp_storage.shared_buffer, items2, is_last_tile, num_tile_items);

  __syncthreads();

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (!is_last_tile || (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items))
      if (selection_flags[ITEM])
        sum += items[ITEM] * items2[ITEM];
  }

  __syncthreads();

  unsigned long long aggregate = BlockReduceInt(temp_storage.reduce).Sum(sum);

  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(revenue, aggregate);
  }
}

float runQuery(encoded_column lo_orderdate_val, encoded_column lo_orderdate_rl, 
  encoded_column lo_discount, encoded_column lo_quantity, 
    encoded_column lo_extendedprice,
    int lo_num_entries, CachingDeviceAllocator&  g_allocator) {
  SETUP_TIMING();

  float time_query;
  chrono::high_resolution_clock::time_point st, finish;
  st = chrono::high_resolution_clock::now();

  cudaEventRecord(start, 0);

  unsigned long long* d_sum = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_sum, sizeof(long long)));
  cudaMemset(d_sum, 0, sizeof(long long));

  // Run
  const int num_threads = 128;
  const int items_per_thread = 4;
  int tile_size = num_threads * items_per_thread;
  QueryKernel<num_threads, items_per_thread><<<(lo_num_entries + tile_size - 1)/tile_size, 128>>>(
          lo_orderdate_val.block_start, lo_orderdate_val.data,
          lo_orderdate_rl.block_start, lo_orderdate_rl.data,
          lo_discount.block_start, lo_discount.data,
          lo_quantity.block_start, lo_quantity.data,
          lo_extendedprice.block_start, lo_extendedprice.data,
          lo_num_entries, d_sum);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_query, start,stop);

  unsigned long long revenue;
  CubDebugExit(cudaMemcpy(&revenue, d_sum, sizeof(long long), cudaMemcpyDeviceToHost));

  finish = chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = finish - st;

  cout << "Revenue: " << revenue << endl;
  cout << "Time Taken Total: " << diff.count() * 1000 << endl;

  CLEANUP(d_sum);

  return time_query;
}

/**
 * Main
 */
int main(int argc, char** argv)
{
  int num_trials  = 3;
  string encoding = ENCODING;

  encoded_column d_lo_extendedprice = loadEncodedColumnToGPU("lo_extendedprice", encoding, LO_LEN, g_allocator);
  encoded_column d_lo_discount = loadEncodedColumnToGPU("lo_discount", encoding, LO_LEN, g_allocator);
  encoded_column d_lo_quantity = loadEncodedColumnToGPU("lo_quantity", encoding, LO_LEN, g_allocator);
  encoded_column d_lo_orderdate_val = loadEncodedColumnToGPURLE("lo_orderdate", "valbin", LO_LEN, g_allocator);
  encoded_column d_lo_orderdate_rl = loadEncodedColumnToGPURLE("lo_orderdate", "rlbin", LO_LEN, g_allocator);

  cout << "** LOADED DATA TO GPU **" << endl;
  cout << "Encoding: " << encoding << endl;

  for (int t = 0; t < num_trials; t++) {
    float time_query;
    time_query = runQuery(d_lo_orderdate_val, d_lo_orderdate_rl, d_lo_discount, d_lo_quantity, 
            d_lo_extendedprice, 
            LO_LEN, g_allocator);
    cout<< "{"
        << "\"query\":11" 
        << ",\"time_query\":" << time_query
        << "}" << endl;
  }

  return 0;
}