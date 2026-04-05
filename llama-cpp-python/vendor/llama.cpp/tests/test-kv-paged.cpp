/**
 * test-kv-paged.cpp — Unit test cho paged KV block pool (không cần model)
 *
 * Test trực tiếp struct paged_block_t và logic LRU bằng cách
 * gọi API cấp thấp qua một mock đơn giản.
 *
 * Build:
 *   g++ -std=c++17 -O0 -g test-kv-paged.cpp -o test-kv-paged && ./test-kv-paged
 */

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <algorithm>

// --------------------------------------------------------------------------
// Minimal standalone replica của paged block pool (không phụ thuộc llama.cpp)
// --------------------------------------------------------------------------

struct paged_block_t {
    int32_t ref_cnt  = 0;
    int32_t lru_prev = -1;
    int32_t lru_next = -1;
};

struct paged_pool_t {
    uint32_t page_size;
    uint32_t n_blocks;

    std::vector<paged_block_t> blocks;
    int32_t lru_head = -1;
    int32_t lru_tail = -1;
    int32_t n_free   = 0;

    // seq_id → list of physical block_ids (logical block order)
    std::unordered_map<int32_t, std::vector<int32_t>> seq_blocks;

    // ---------- init ----------
    void init(uint32_t ps, uint32_t nb) {
        page_size = ps;
        n_blocks  = nb;
        blocks.resize(nb);
        n_free   = (int32_t)nb;
        lru_head = 0;
        lru_tail = (int32_t)nb - 1;
        for (uint32_t i = 0; i < nb; ++i) {
            blocks[i].ref_cnt  = 0;
            blocks[i].lru_prev = (i == 0)    ? -1 : (int32_t)(i - 1);
            blocks[i].lru_next = (i + 1 < nb) ? (int32_t)(i + 1) : -1;
        }
        seq_blocks.clear();
    }

    // ---------- LRU list ----------
    void lru_remove(int32_t bid) {
        auto & b = blocks[bid];
        if (b.lru_prev >= 0) blocks[b.lru_prev].lru_next = b.lru_next;
        else                 lru_head = b.lru_next;
        if (b.lru_next >= 0) blocks[b.lru_next].lru_prev = b.lru_prev;
        else                 lru_tail = b.lru_prev;
        b.lru_prev = b.lru_next = -1;
    }

    void lru_push_head(int32_t bid) {
        auto & b = blocks[bid];
        b.lru_prev = -1;
        b.lru_next = lru_head;
        if (lru_head >= 0) blocks[lru_head].lru_prev = bid;
        else               lru_tail = bid;
        lru_head = bid;
    }

    // ---------- alloc / free ----------
    int32_t alloc_block() {
        if (n_free <= 0) return -1;
        int32_t bid = lru_head;
        lru_remove(bid);
        n_free--;
        blocks[bid].ref_cnt = 1;
        return bid;
    }

    void free_block(int32_t bid) {
        assert(bid >= 0 && (uint32_t)bid < n_blocks);
        auto & b = blocks[bid];
        assert(b.ref_cnt > 0);
        if (--b.ref_cnt == 0) {
            lru_push_head(bid);
            n_free++;
        }
    }

    // ---------- alloc for sequence ----------
    // Returns cell indices [physical_block * page_size + slot] for each pos
    std::vector<uint32_t> alloc_for_seq(int32_t seq_id,
                                        const std::vector<int32_t> & positions) {
        auto & sblocks = seq_blocks[seq_id];
        std::vector<uint32_t> cell_idxs;
        cell_idxs.reserve(positions.size());

        for (int32_t pos : positions) {
            uint32_t lb   = (uint32_t)pos / page_size;
            uint32_t slot = (uint32_t)pos % page_size;

            while (sblocks.size() <= lb) {
                int32_t bid = alloc_block();
                if (bid < 0) return {};  // OOM
                sblocks.push_back(bid);
            }

            int32_t phys = sblocks[lb];
            cell_idxs.push_back((uint32_t)phys * page_size + slot);
        }
        return cell_idxs;
    }

    // ---------- free sequence ----------
    void free_seq(int32_t seq_id) {
        auto it = seq_blocks.find(seq_id);
        if (it == seq_blocks.end()) return;
        for (int32_t bid : it->second) {
            if (bid >= 0) free_block(bid);
        }
        seq_blocks.erase(it);
    }
};

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

static int n_pass = 0;
static int n_fail = 0;

#define CHECK(cond) do { \
    if (cond) { n_pass++; } \
    else { n_fail++; printf("FAIL: %s  (line %d)\n", #cond, __LINE__); } \
} while(0)

void test_init() {
    printf("=== test_init ===\n");
    paged_pool_t pool;
    pool.init(16, 4);

    CHECK(pool.n_free == 4);
    CHECK(pool.lru_head == 0);
    CHECK(pool.lru_tail == 3);
    CHECK(pool.blocks[0].ref_cnt == 0);
    CHECK(pool.blocks[0].lru_prev == -1);
    CHECK(pool.blocks[0].lru_next == 1);
    CHECK(pool.blocks[3].lru_prev == 2);
    CHECK(pool.blocks[3].lru_next == -1);
}

void test_alloc_free() {
    printf("=== test_alloc_free ===\n");
    paged_pool_t pool;
    pool.init(16, 4);

    int32_t b0 = pool.alloc_block();
    CHECK(b0 == 0);
    CHECK(pool.n_free == 3);
    CHECK(pool.blocks[0].ref_cnt == 1);
    CHECK(pool.lru_head == 1);

    int32_t b1 = pool.alloc_block();
    CHECK(b1 == 1);
    CHECK(pool.n_free == 2);

    pool.free_block(b0);
    CHECK(pool.n_free == 3);
    CHECK(pool.blocks[0].ref_cnt == 0);
    CHECK(pool.lru_head == 0);  // returned to head

    // Alloc again: should get b0 back (head of free list)
    int32_t b0_again = pool.alloc_block();
    CHECK(b0_again == 0);
}

void test_oom() {
    printf("=== test_oom ===\n");
    paged_pool_t pool;
    pool.init(16, 2);

    int32_t b0 = pool.alloc_block();
    int32_t b1 = pool.alloc_block();
    CHECK(b0 >= 0 && b1 >= 0);
    CHECK(pool.n_free == 0);

    int32_t b2 = pool.alloc_block();
    CHECK(b2 == -1);  // OOM
}

void test_seq_alloc() {
    printf("=== test_seq_alloc ===\n");
    paged_pool_t pool;
    pool.init(4, 8);  // page_size=4, 8 blocks = 32 cells

    // Sequence 0: positions 0..7 → needs 2 blocks
    std::vector<int32_t> pos0 = {0, 1, 2, 3, 4, 5, 6, 7};
    auto idxs0 = pool.alloc_for_seq(0, pos0);

    CHECK(idxs0.size() == 8);
    CHECK(pool.seq_blocks[0].size() == 2);
    CHECK(pool.n_free == 6);

    // Verify cell indices: positions 0-3 in block 0, 4-7 in block 1
    int32_t blk0 = pool.seq_blocks[0][0];
    int32_t blk1 = pool.seq_blocks[0][1];
    CHECK(idxs0[0] == (uint32_t)blk0 * 4 + 0);
    CHECK(idxs0[3] == (uint32_t)blk0 * 4 + 3);
    CHECK(idxs0[4] == (uint32_t)blk1 * 4 + 0);
    CHECK(idxs0[7] == (uint32_t)blk1 * 4 + 3);

    // Sequence 1: positions 0..3 → needs 1 block
    auto idxs1 = pool.alloc_for_seq(1, {0, 1, 2, 3});
    CHECK(pool.seq_blocks[1].size() == 1);
    CHECK(pool.n_free == 5);

    // Free sequence 0
    pool.free_seq(0);
    CHECK(pool.n_free == 7);
    CHECK(pool.seq_blocks.find(0) == pool.seq_blocks.end());
}

void test_incremental_alloc() {
    printf("=== test_incremental_alloc ===\n");
    paged_pool_t pool;
    pool.init(2, 4);  // page_size=2, 4 blocks

    // Allocate positions 0-1 first
    auto idxs_a = pool.alloc_for_seq(0, {0, 1});
    CHECK(pool.seq_blocks[0].size() == 1);
    CHECK(pool.n_free == 3);

    // Then positions 2-3 (second block)
    auto idxs_b = pool.alloc_for_seq(0, {2, 3});
    CHECK(pool.seq_blocks[0].size() == 2);
    CHECK(pool.n_free == 2);

    // Cell indices should use same blocks as before
    int32_t blk0 = pool.seq_blocks[0][0];
    int32_t blk1 = pool.seq_blocks[0][1];
    CHECK(idxs_a[0] == (uint32_t)blk0 * 2 + 0);
    CHECK(idxs_a[1] == (uint32_t)blk0 * 2 + 1);
    CHECK(idxs_b[0] == (uint32_t)blk1 * 2 + 0);
    CHECK(idxs_b[1] == (uint32_t)blk1 * 2 + 1);
}

void test_multi_seq() {
    printf("=== test_multi_seq ===\n");
    paged_pool_t pool;
    pool.init(4, 6);

    // Seq 0: 8 tokens → 2 blocks
    pool.alloc_for_seq(0, {0,1,2,3,4,5,6,7});
    CHECK(pool.n_free == 4);

    // Seq 1: 4 tokens → 1 block
    pool.alloc_for_seq(1, {0,1,2,3});
    CHECK(pool.n_free == 3);

    // Seq 2: 8 tokens → 2 blocks
    pool.alloc_for_seq(2, {0,1,2,3,4,5,6,7});
    CHECK(pool.n_free == 1);

    // Verify blocks don't overlap
    std::vector<int32_t> all_blocks;
    for (auto & [sid, sblks] : pool.seq_blocks) {
        for (int32_t b : sblks) all_blocks.push_back(b);
    }
    std::sort(all_blocks.begin(), all_blocks.end());
    auto it = std::adjacent_find(all_blocks.begin(), all_blocks.end());
    CHECK(it == all_blocks.end());  // no duplicates
}

void test_reuse_after_free() {
    printf("=== test_reuse_after_free ===\n");
    paged_pool_t pool;
    pool.init(4, 2);

    pool.alloc_for_seq(0, {0,1,2,3,4,5,6,7});
    CHECK(pool.n_free == 0);

    pool.free_seq(0);
    CHECK(pool.n_free == 2);

    // Should be able to allocate again
    auto idxs = pool.alloc_for_seq(1, {0,1,2,3,4,5,6,7});
    CHECK(!idxs.empty());
    CHECK(pool.n_free == 0);
}

int main() {
    test_init();
    test_alloc_free();
    test_oom();
    test_seq_alloc();
    test_incremental_alloc();
    test_multi_seq();
    test_reuse_after_free();

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
