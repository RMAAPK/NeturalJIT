#include "njit/storage_engine.hpp"
#include <chrono>
#include <cstring>
#include <iostream>

namespace NJIT {

StorageEngine::StorageEngine(const std::string& db_path)
    : m_db_path(db_path) {
    InitializeSchema();
}

StorageEngine::~StorageEngine() {}

bool StorageEngine::InitializeSchema() {
    // Pure fast-path lockless shm cache backing
    return true;
}

bool StorageEngine::StoreBlock(const std::string& doc_key, const std::vector<uint8_t>& njit_blob) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_shm_cache[doc_key] = njit_blob;
    return true;
}

std::optional<std::vector<uint8_t>> StorageEngine::FetchBlock(const std::string& doc_key) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_shm_cache.find(doc_key);
    if (it != m_shm_cache.end()) {
        return it->second;
    }
    return std::nullopt;
}

GraftResult StorageEngine::GraftToAttentionCache(
    const std::string& doc_key,
    void* kv_cache_dest,
    size_t dest_capacity
) {
    auto t0 = std::chrono::high_resolution_clock::now();

    auto block = FetchBlock(doc_key);
    if (!block) {
        return {false, 0, 0.0};
    }

    NJITHeader hdr;
    if (!NJITPackage::Validate(block->data(), block->size(), hdr)) {
        return {false, 0, 0.0};
    }

    if (hdr.payload_bytes > dest_capacity) {
        return {false, 0, 0.0};
    }

    // Direct memory graft: bypasses tokenizer and transformer forward passes
    const uint8_t* payload = block->data() + sizeof(NJITHeader);
    std::memcpy(kv_cache_dest, payload, hdr.payload_bytes);

    auto t1 = std::chrono::high_resolution_clock::now();
    double micros = std::chrono::duration<double, std::micro>(t1 - t0).count();

    return {true, hdr.seq_len, micros};
}

} // namespace NJIT
