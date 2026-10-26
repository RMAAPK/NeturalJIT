#include "njit/njit_format.hpp"
#include <cstring>
#include <iostream>

namespace NJIT {

NJITPackage::NJITPackage() {}
NJITPackage::~NJITPackage() {}

std::vector<uint8_t> NJITPackage::Serialize(
    uint64_t cid,
    TensorType type,
    uint32_t num_layers,
    uint32_t head_dim,
    uint32_t seq_len,
    const uint8_t* tensor_data,
    size_t tensor_len
) {
    std::vector<uint8_t> buffer(sizeof(NJITHeader) + tensor_len);

    NJITHeader* hdr = reinterpret_cast<NJITHeader*>(buffer.data());
    hdr->magic = NJIT_MAGIC;
    hdr->version = NJIT_VERSION;
    hdr->tensor_type = static_cast<uint8_t>(type);
    hdr->reserved = 0;
    hdr->content_id = cid;
    hdr->num_layers = num_layers;
    hdr->head_dim = head_dim;
    hdr->seq_len = seq_len;
    hdr->payload_bytes = tensor_len;

    if (tensor_data && tensor_len > 0) {
        std::memcpy(buffer.data() + sizeof(NJITHeader), tensor_data, tensor_len);
    }

    return buffer;
}

bool NJITPackage::Validate(const uint8_t* buffer, size_t size, NJITHeader& out_header) {
    if (!buffer || size < sizeof(NJITHeader)) {
        return false;
    }

    const NJITHeader* hdr = reinterpret_cast<const NJITHeader*>(buffer);
    if (hdr->magic != NJIT_MAGIC) {
        return false;
    }

    if (hdr->payload_bytes + sizeof(NJITHeader) > size) {
        return false;
    }

    out_header = *hdr;
    return true;
}

} // namespace NJIT
