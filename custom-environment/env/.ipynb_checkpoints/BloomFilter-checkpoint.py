class BloomFilter:
    def __init__(self, size:int = 100, hash_count:int = 3):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = [0] * size
    def _hashes(self, item):
        for i in range(self.hash_count):
            yield hash((item, i)) % self.size
    def add(self, item):
        for h in self._hashes(item):
            self.bit_array[h] = 1
    def might_contain(self, item):
        return all(self.bit_array[h] for h in self._hashes(item))
'''
## BF is not symmetric
agent_count = 2
bloom_filters = [BloomFilter() for _ in range(agent_count)]
bloom_filters[0].add(1)
print("Agent 0 has seen Agent 1:", bloom_filters[0].might_contain(1)) 
print("Agent 1 has seen Agent 0:", bloom_filters[1].might_contain(0))
'''