#pragma once

#include "njit_format.hpp"
#include <string>
#include <vector>
#include <optional>
#include <unordered_map>
#include <mutex>

namespace NJIT {

struct GraftResult {
    bool success;
    size_t tokens_injected;
    double latency_microseconds;
};

class StorageEngine {
public:
    StorageEngine(const std::string& db_path = ":memory:");
    ~StorageEngine();

    bool InitializeSchema();

    // Store pre-computed KV tensor block by canonical hash / key
    bool StoreBlock(const std::string& doc_key, const std::vector<uint8_t>& njit_blob);

    // Retrieve binary block directly from SQLite
    std::optional<std::vector<uint8_t>> FetchBlock(const std::string& doc_key);

    // Microsecond Grafting: injects the binary KV blob directly into target memory
    GraftResult GraftToAttentionCache(const std::string& doc_key, void* kv_cache_dest, size_t dest_capacity);

private:
    std::string m_db_path;
    std::unordered_map<std::string, std::vector<uint8_t>> m_shm_cache;
    std::mutex m_mutex;
};

} // namespace NJIT
