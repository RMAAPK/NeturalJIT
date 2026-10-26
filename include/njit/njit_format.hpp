#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>

namespace NJIT {

#pragma pack(push, 1)

// NJIT Binary Container Magic: "NJIT"
constexpr uint32_t NJIT_MAGIC = 0x54494A4E;
constexpr uint16_t NJIT_VERSION = 0x0100; // v1.0

enum class TensorType : uint8_t {
    KV_CACHE_F16 = 0x01,
    KV_CACHE_Q8_0 = 0x02,
    KV_CACHE_Q4_0 = 0x03,
    LORA_DELTA_F16 = 0x10,
    EMBEDDING_F32 = 0x20
};

struct NJITHeader {
    uint32_t magic;           // 0x54494A4E ('NJIT')
    uint16_t version;         // Format version
    uint8_t  tensor_type;     // TensorType
    uint8_t  reserved;
    uint64_t content_id;      // 64-bit deterministic cryptographic hash
    uint32_t num_layers;      // Transformer layer count
    uint32_t head_dim;        // Attention head dimension
    uint32_t seq_len;         // Pre-computed sequence length
    uint64_t payload_bytes;   // Size of compressed tensor payload in bytes
};

#pragma pack(pop)

class NJITPackage {
public:
    NJITPackage();
    ~NJITPackage();

    // Create an in-memory package
    static std::vector<uint8_t> Serialize(
        uint64_t cid,
        TensorType type,
        uint32_t num_layers,
        uint32_t head_dim,
        uint32_t seq_len,
        const uint8_t* tensor_data,
        size_t tensor_len
    );

    // Validate and inspect a .njit container
    static bool Validate(const uint8_t* buffer, size_t size, NJITHeader& out_header);
};

} // namespace NJIT
