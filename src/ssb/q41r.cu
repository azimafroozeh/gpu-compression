// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <curand.h>

#include <cuda.h>
#include <cub/util_allocator.cuh>
#include <cub/cub.cuh>

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

#define HASH_WM(X,Y,Z) ((X-Z) % Y)
#define HASH(X,Y) (X % Y)

#define CHECK_ERROR() { \
  cudaDeviceSynchronize(); \
  cudaError_t error = cudaGetLastError(); \
  if(error != cudaSuccess) \
  { \
    printf("CUDA error: %s\n", cudaGetErrorString(error)); \
    exit(-1); \
  } \
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe(//int* lo_orderdate, int* lo_partkey, int* lo_custkey, int* lo_suppkey, int* lo_revenue, int* lo_supplycost, int lo_len,
    uint* lo_orderdate_val_block_start, uint* lo_orderdate_val_data,
    uint* lo_orderdate_rl_block_start, uint* lo_orderdate_rl_data,
    uint* lo_custkey_val_block_start, uint* lo_custkey_val_data,
    uint* lo_custkey_rl_block_start, uint* lo_custkey_rl_data,
    uint* lo_partkey_block_start, uint* lo_partkey_data,
    uint* lo_suppkey_block_start, uint* lo_suppkey_data,
    uint* lo_revenue_block_start, uint* lo_revenue_data,
    uint* lo_supplycost_block_start, uint* lo_supplycost_data,
    int lo_len,
    int* ht_p, int p_len,
    int* ht_s, int s_len,
    int* ht_c, int c_len,
    int* ht_d, int d_len,
    int* res) {
  // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
  /*typedef cub::BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_TRANSPOSE> BlockLoadInt;*/

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
  int c_nation[ITEMS_PER_THREAD];
  int year[ITEMS_PER_THREAD];
  int revenue[ITEMS_PER_THREAD];

  int num_tiles = (lo_len + tile_size - 1) / tile_size;
  int num_tile_items = tile_size;
  bool is_last_tile = false;
  if (tile_idx == num_tiles - 1) {
    num_tile_items = lo_len - tile_offset;
    is_last_tile = true;
  }

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    selection_flags[ITEM] = 1;
  }

  ENCODINGKERNEL<BLOCK_THREADS,ITEMS_PER_THREAD>(lo_suppkey_block_start, lo_suppkey_data, temp_storage.shared_buffer, items, is_last_tile, num_tile_items);

  // Barrier for smem reuse
  __syncthreads();

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    // Out-of-bounds items are selection_flags
    int hash = HASH(items[ITEM], s_len);

    if (!is_last_tile || (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items)) {
      int slot = ht_s[hash << 1];
      if (slot != 0) {
        /*s_nation[ITEM] = ht_s[(hash << 1) + 1];*/
      } else {
        selection_flags[ITEM] = 0;
      }
    }
  }

  __syncthreads();

  RENCODINGKERNEL<BLOCK_THREADS,ITEMS_PER_THREAD>(
      lo_custkey_val_block_start, lo_custkey_rl_block_start, lo_custkey_val_data, lo_custkey_rl_data,
      temp_storage.shared_buffer, items, revenue, is_last_tile, num_tile_items);

  // Barrier for smem reuse
  __syncthreads();

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (!is_last_tile || (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items)) {
      // Out-of-bounds items are selection_flags
      if (selection_flags[ITEM]) {
        int hash = HASH(items[ITEM], c_len);
        int slot = ht_c[hash << 1];

        if (slot != 0) {
          c_nation[ITEM] = ht_c[(hash << 1) + 1];
        } else {
          selection_flags[ITEM] = 0;
        }
      }
    }
  }

  __syncthreads();

  ENCODINGKERNEL<BLOCK_THREADS,ITEMS_PER_THREAD>(lo_partkey_block_start, lo_partkey_data, temp_storage.shared_buffer, items, is_last_tile, num_tile_items);

  // Barrier for smem reuse
  __syncthreads();

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if (!is_last_tile || (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items)) {
      // Out-of-bounds items are selection_flags
      if (selection_flags[ITEM]) {
        int hash = HASH(items[ITEM], p_len);
        int slot = ht_p[hash << 1];

        if (slot != 0) {
          /*c_nation[ITEM] = ht_c[(hash << 1) + 1];*/
        } else {
          selection_flags[ITEM] = 0;
        }
      }
    }
  }

  __syncthreads();

  RENCODINGKERNEL<BLOCK_THREADS,ITEMS_PER_THREAD>(
      lo_orderdate_val_block_start, lo_orderdate_rl_block_start, lo_orderdate_val_data, lo_orderdate_rl_data,
      temp_storage.shared_buffer, items, revenue, is_last_tile, num_tile_items);

  // Barrier for smem reuse
  __syncthreads();

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    // Out-of-bounds items are selection_flags
    int hash = HASH_WM(items[ITEM], d_len, 19920101);

    if (!is_last_tile || (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items)) {
      if (selection_flags[ITEM]) {
        int slot = ht_d[hash << 1];
        if (slot != 0) {
          year[ITEM] = ht_d[(hash << 1) + 1];
        } else {
          selection_flags[ITEM] = 0;
        }
      }
    }
  }

  __syncthreads();

  ENCODINGKERNEL<BLOCK_THREADS,ITEMS_PER_THREAD>(lo_revenue_block_start, lo_revenue_data, temp_storage.shared_buffer, revenue, is_last_tile, num_tile_items);


  // Barrier for smem reuse
  __syncthreads();

  ENCODINGKERNEL<BLOCK_THREADS,ITEMS_PER_THREAD>(lo_supplycost_block_start, lo_supplycost_data, temp_storage.shared_buffer, items, is_last_tile, num_tile_items);


  // Barrier for smem reuse
  __syncthreads();

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if (!is_last_tile || (int(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items)) {
      if (selection_flags[ITEM]) {
        int hash = (c_nation[ITEM] * 7 +  (year[ITEM] - 1992)) % ((1998-1992+1) * 25);
        res[hash * 4] = year[ITEM];
        res[hash * 4 + 1] = c_nation[ITEM];
        atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 4 + 2]), (long long)(revenue[ITEM] - items[ITEM]));
      }
    }
  }
}

__global__
void build_hashtable_s(int *filter_col, int *dim_key, int num_tuples, int *hash_table, int num_slots) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < num_tuples) {
    int val = filter_col[offset];
    if (val == 1) {
      int key = dim_key[offset];
      int hash = HASH(key, num_slots);
      int init = 0;

      int old = atomicCAS(&hash_table[hash << 1], init, key);
    }
  }
}

__global__
void build_hashtable_p(int *filter_col, int *dim_key, int num_tuples, int *hash_table, int num_slots) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < num_tuples) {
    int val = filter_col[offset];
    if (val == 0 || val == 1) {
      int key = dim_key[offset];
      int hash = HASH(key, num_slots);
      int init = 0;

      int old = atomicCAS(&hash_table[hash << 1], init, key);
    }
  }
}

__global__
void build_hashtable_c(int* filter_col, int *dim_key, int* dim_val, int num_tuples, int *hash_table, int num_slots) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < num_tuples) {
    int f = filter_col[offset];
    if (f == 1) {
      int key = dim_key[offset];
      int val = dim_val[offset];
      int hash = HASH(key, num_slots);
      int init = 0;

      int old = atomicCAS(&hash_table[hash << 1], init, key);
      hash_table[(hash << 1) + 1] = val;
    }
  }
}

__global__
void build_hashtable_d(int *dim_key, int *dim_val, int num_tuples, int *hash_table, int num_slots, int val_min) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < num_tuples) {
      int key = dim_key[offset];
      int val = dim_val[offset];
      int hash = HASH_WM(key, num_slots, val_min);

      int init = 0;

      int old = atomicCAS(&hash_table[hash << 1], init, key);
      hash_table[(hash << 1) + 1] = val;
  }
}

float runQuery(encoded_column lo_orderdate_val, encoded_column lo_orderdate_rl, encoded_column lo_custkey_val, encoded_column lo_custkey_rl, encoded_column lo_partkey,
    encoded_column lo_suppkey, encoded_column lo_revenue, encoded_column lo_supplycost, int lo_len,
    int *d_datekey, int* d_year, int d_len,
    int *p_partkey, int* p_mfgr, int p_len,
    int *s_suppkey, int* s_region, int s_len,
    int *c_custkey, int* c_region, int* c_nation, int c_len,
    CachingDeviceAllocator&  g_allocator) {
  SETUP_TIMING();

  float time_query;
  chrono::high_resolution_clock::time_point st, finish;
  st = chrono::high_resolution_clock::now();

  cudaEventRecord(start, 0);

  int *ht_d, *ht_c, *ht_s, *ht_p;
  int d_val_len = 19981230 - 19920101 + 1;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&ht_d, 2 * d_val_len * sizeof(int)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&ht_s, 2 * s_len * sizeof(int)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&ht_c, 2 * c_len * sizeof(int)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&ht_p, 2 * p_len * sizeof(int)));

  CubDebugExit(cudaMemset(ht_d, 0, 2 * d_val_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_s, 0, 2 * s_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_c, 0, 2 * c_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_p, 0, 2 * p_len * sizeof(int)));

  build_hashtable_s<<<(s_len + 127)/128, 128>>>(s_region, s_suppkey, s_len, ht_s, s_len);

  build_hashtable_c<<<(c_len + 127)/128, 128>>>(c_region, c_custkey, c_nation, c_len, ht_c, c_len);

  build_hashtable_p<<<(p_len + 127)/128, 128>>>(p_mfgr, p_partkey, p_len, ht_p, p_len);

  int d_val_min = 19920101;
  build_hashtable_d<<<(d_len + 127)/128, 128>>>(d_datekey, d_year, d_len, ht_d, d_val_len, d_val_min);

  int *res;
  int res_size = ((1998-1992+1) * 25);
  int ht_entries = 4; // int,int,long long
  int res_array_size = res_size * ht_entries;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&res, res_array_size * sizeof(int)));

  CubDebugExit(cudaMemset(res, 0, res_array_size * sizeof(int)));

  // Run
  const int num_threads = 128;
  const int items_per_thread = 4;
  int tile_size = num_threads * items_per_thread;
  probe<num_threads, items_per_thread><<<(lo_len + tile_size - 1)/tile_size, 128>>>(
      lo_orderdate_val.block_start, lo_orderdate_val.data,
      lo_orderdate_rl.block_start, lo_orderdate_rl.data,
      lo_custkey_val.block_start, lo_custkey_val.data,
      lo_custkey_rl.block_start, lo_custkey_rl.data,
      lo_partkey.block_start, lo_partkey.data,
      lo_suppkey.block_start, lo_suppkey.data,
      lo_revenue.block_start, lo_revenue.data,
      lo_supplycost.block_start, lo_supplycost.data, 
      lo_len, 
      ht_p, p_len, ht_s, s_len, ht_c, c_len, ht_d, d_val_len, res);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_query, start,stop);

  int* h_res = new int[res_array_size];
  CubDebugExit(cudaMemcpy(h_res, res, res_array_size * sizeof(int), cudaMemcpyDeviceToHost));
  finish = chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = finish - st;

  cout << "Result:" << endl;
  int res_count = 0;
  for (int i=0; i<res_size; i++) {
    if (h_res[4*i] != 0) {
      cout << h_res[4*i] << " " << h_res[4*i + 1] << " " << reinterpret_cast<unsigned long long*>(&h_res[4*i + 2])[0]  << endl;
      res_count += 1;
    }
  }

  cout << "Res Count: " << res_count << endl;
  cout << "Time Taken Total: " << diff.count() * 1000 << endl;

  delete[] h_res;

  return time_query;
}

/**
 * Main
 */
int main(int argc, char** argv)
{
  int num_trials          = 3;

  int *h_d_datekey = loadColumn<int>("d_datekey", D_LEN);
  int *h_d_year = loadColumn<int>("d_year", D_LEN);
  int *h_d_yearmonthnum = loadColumn<int>("d_yearmonthnum", D_LEN);

  int *h_s_suppkey = loadColumn<int>("s_suppkey", S_LEN);
  int *h_s_region = loadColumn<int>("s_region", S_LEN);

  int *h_p_partkey = loadColumn<int>("p_partkey", P_LEN);
  int *h_p_mfgr = loadColumn<int>("p_mfgr", P_LEN);

  int *h_c_custkey = loadColumn<int>("c_custkey", C_LEN);
  int *h_c_region = loadColumn<int>("c_region", C_LEN);
  int *h_c_nation = loadColumn<int>("c_nation", C_LEN);

  cout << "** LOADED DATA **" << endl;

  encoded_column d_lo_orderdate_val = loadEncodedColumnToGPURLE("lo_orderdate", "valbin", LO_LEN, g_allocator);
  encoded_column d_lo_orderdate_rl = loadEncodedColumnToGPURLE("lo_orderdate", "rlbin", LO_LEN, g_allocator);
  encoded_column d_lo_custkey_val = loadEncodedColumnToGPURLE("lo_custkey", "valbin", LO_LEN, g_allocator);
  encoded_column d_lo_custkey_rl = loadEncodedColumnToGPURLE("lo_custkey", "rlbin", LO_LEN, g_allocator);
  encoded_column d_lo_suppkey = loadEncodedColumnToGPU("lo_suppkey", ENCODING, LO_LEN, g_allocator);
  encoded_column d_lo_partkey = loadEncodedColumnToGPU("lo_partkey", ENCODING, LO_LEN, g_allocator);
  encoded_column d_lo_revenue = loadEncodedColumnToGPU("lo_revenue", ENCODING, LO_LEN, g_allocator);
  encoded_column d_lo_supplycost = loadEncodedColumnToGPU("lo_supplycost", ENCODING, LO_LEN, g_allocator);

  int *d_d_datekey = loadColumnToGPU<int>(h_d_datekey, D_LEN, g_allocator);
  int *d_d_year = loadColumnToGPU<int>(h_d_year, D_LEN, g_allocator);

  int *d_p_partkey = loadColumnToGPU<int>(h_p_partkey, P_LEN, g_allocator);
  int *d_p_mfgr = loadColumnToGPU<int>(h_p_mfgr, P_LEN, g_allocator);

  int *d_s_suppkey = loadColumnToGPU<int>(h_s_suppkey, S_LEN, g_allocator);
  int *d_s_region = loadColumnToGPU<int>(h_s_region, S_LEN, g_allocator);

  int *d_c_custkey = loadColumnToGPU<int>(h_c_custkey, C_LEN, g_allocator);
  int *d_c_region = loadColumnToGPU<int>(h_c_region, C_LEN, g_allocator);
  int *d_c_nation = loadColumnToGPU<int>(h_c_nation, C_LEN, g_allocator);

  cout << "** LOADED DATA TO GPU **" << endl;

  for (int t = 0; t < num_trials; t++) {
    float time_query;
    time_query = runQuery(
        d_lo_orderdate_val, d_lo_orderdate_rl, d_lo_custkey_val, d_lo_custkey_rl, 
        d_lo_partkey, d_lo_suppkey, d_lo_revenue, d_lo_supplycost, LO_LEN,
        d_d_datekey, d_d_year, D_LEN,
        d_p_partkey, d_p_mfgr, P_LEN,
        d_s_suppkey, d_s_region, S_LEN,
        d_c_custkey, d_c_region, d_c_nation, C_LEN,
        g_allocator);
    cout<< "{"
        << "\"query\":41"
        << ",\"time_query\":" << time_query
        << "}" << endl;
  }

  return 0;
}