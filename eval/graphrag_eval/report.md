# GraphRAG vs BM25 Retrieval Evaluation

## Summary

| Metric | BM25 | GraphRAG | Delta (G-B) | Relative | 95% CI BM25 | 95% CI GraphRAG |
|---|---:|---:|---:|---:|---:|---:|
| recall@10 | 0.6417 | 0.5033 | -0.1383 | -21.56% | [0.4831, 0.8167] | [0.3367, 0.6667] |
| recall@20 | 0.6500 | 0.5033 | -0.1467 | -22.56% | [0.4833, 0.8333] | [0.3367, 0.6667] |
| mrr | 0.5372 | 0.4639 | -0.0733 | -13.65% | [0.3744, 0.7050] | [0.3000, 0.6361] |
| ndcg@10 | 0.5659 | 0.4611 | -0.1048 | -18.52% | [0.4067, 0.7344] | [0.3009, 0.6198] |
| precision@10 | 0.1100 | 0.0833 | -0.0267 | -24.24% | [0.0700, 0.1600] | [0.0500, 0.1234] |

## Conclusion

On this test set, GraphRAG is -22.56% worse than BM25 on Recall@20 (GraphRAG CI [0.3367, 0.6667]), and -13.65% worse on MRR (GraphRAG CI [0.3000, 0.6361]).

## Per-query Breakdown (top 10 deltas)

### Worst for GraphRAG (BM25 wins)

- Query: What did researchers say about Welcome Mixtral - a SOTA Mixture of Experts on Hugging Face?
  - Gold IDs: 6818
  - BM25 top10: 6805, 3871, 6817, 6818, 6838, 6508, 3776, 6810, 6814, 6811 (first gold rank: 4)
  - GraphRAG top10: 6805, 6817, 6810, 6811, 6814, 6808, 3871, 6508 (first gold rank: None)

- Query: What is the update regarding Rocket Money x Hugging Face: Scaling Volatile ML Models in Production​?
  - Gold IDs: 7105, 7103
  - BM25 top10: 7098, 7105, 7106, 7103, 6955, 7486, 7102, 6234, 7321, 6155 (first gold rank: 2)
  - GraphRAG top10: 7098, 6182, 3631, 7469, 2804, 5970, 5971, 5028 (first gold rank: None)

- Query: What is A Knapsack Public Key Cryptosystem Based on Arithmetic in Finite Fields (1988) [pdf] about?
  - Gold IDs: 12252
  - BM25 top10: 12252, 8087, 8056, 11418, 12354, 9284, 11111, 8683, 2533, 3162 (first gold rank: 1)
  - GraphRAG top10: 12245, 12023, 11848, 11604, 11946, 12014, 11616, 12016 (first gold rank: None)

- Query: What is Powerful ASR + diarization + speculative decoding with Hugging Face Inference Endpoints about?
  - Gold IDs: 6066, 6070
  - BM25 top10: 6066, 6070, 6073, 6067, 6071, 4952, 5144, 5188, 6955, 6072 (first gold rank: 1)
  - GraphRAG top10: 6066, 6182, 3631, 7469, 2804, 5970, 5971, 5028 (first gold rank: 1)

- Query: What did the article on Training and Finetuning Sparse Embedding Models with Sentence Transformers v5 report?
  - Gold IDs: 3920, 3297, 3319, 3299
  - BM25 top10: 3292, 5917, 3920, 3919, 3297, 4564, 4559, 3319, 2091, 4568 (first gold rank: 3)
  - GraphRAG top10: 3292, 3316, 3304, 3297, 3319, 3294, 3300, 5917 (first gold rank: 4)

- Query: What are the key details about Hugging Face Text Generation Inference available for AWS Inferentia2?
  - Gold IDs: 5951, 6543, 5952, 5954, 5953
  - BM25 top10: 6543, 5951, 5952, 6893, 5954, 6897, 6551, 5953, 6544, 5591 (first gold rank: 1)
  - GraphRAG top10: 6543, 5951, 5952, 6551, 6897, 5591, 6544, 6546 (first gold rank: 1)

- Query: What are the main points in Correspondence Between Don Knuth and Peter van Emde Boas on Priority Deques 1977 [pdf]?
  - Gold IDs: 
  - BM25 top10: 3898, 11412, 3897, 3454, 5985, 11382, 2529, 8455, 2405, 3894 (first gold rank: None)
  - GraphRAG top10: 1812, 2003, 1586, 1534, 1954, 1917, 1527, 1538 (first gold rank: None)

- Query: What is the significance of Train 400x faster Static Embedding Models with Sentence Transformers in AI?
  - Gold IDs: 4559
  - BM25 top10: 4559, 4564, 4574, 4568, 4573, 3935, 4608, 4607, 5917, 4606 (first gold rank: 1)
  - GraphRAG top10: 4559, 4564, 4608, 4574, 4568, 4573, 4603, 4606 (first gold rank: 1)

- Query: What problem does Granite 4.0 Nano: Just how small can you go? address?
  - Gold IDs: 2040, 2041, 2042
  - BM25 top10: 2040, 2041, 11119, 2042, 322, 11015, 6883, 3477, 8189, 10913 (first gold rank: 1)
  - GraphRAG top10: 2040, 2041, 2042, 2297, 322, 8189, 33, 10933 (first gold rank: 1)

- Query: What are the key details about Correspondence Between Don Knuth and Peter van Emde Boas on Priority Deques 1977 [pdf]?
  - Gold IDs: 
  - BM25 top10: 11412, 3898, 11382, 10864, 3897, 4231, 9233, 3894, 5539, 6961 (first gold rank: None)
  - GraphRAG top10: 1812, 2003, 1586, 1534, 1954, 1917, 1527, 1538 (first gold rank: None)

### Best for GraphRAG (GraphRAG wins)

- Query: What are the main points in Correspondence Between Don Knuth and Peter van Emde Boas on Priority Deques 1977 [pdf]?
  - Gold IDs: 
  - BM25 top10: 3898, 11412, 3897, 3454, 5985, 11382, 2529, 8455, 2405, 3894 (first gold rank: None)
  - GraphRAG top10: 1812, 2003, 1586, 1534, 1954, 1917, 1527, 1538 (first gold rank: None)

- Query: What is the significance of Train 400x faster Static Embedding Models with Sentence Transformers in AI?
  - Gold IDs: 4559
  - BM25 top10: 4559, 4564, 4574, 4568, 4573, 3935, 4608, 4607, 5917, 4606 (first gold rank: 1)
  - GraphRAG top10: 4559, 4564, 4608, 4574, 4568, 4573, 4603, 4606 (first gold rank: 1)

- Query: What problem does Granite 4.0 Nano: Just how small can you go? address?
  - Gold IDs: 2040, 2041, 2042
  - BM25 top10: 2040, 2041, 11119, 2042, 322, 11015, 6883, 3477, 8189, 10913 (first gold rank: 1)
  - GraphRAG top10: 2040, 2041, 2042, 2297, 322, 8189, 33, 10933 (first gold rank: 1)

- Query: What are the key details about Correspondence Between Don Knuth and Peter van Emde Boas on Priority Deques 1977 [pdf]?
  - Gold IDs: 
  - BM25 top10: 11412, 3898, 11382, 10864, 3897, 4231, 9233, 3894, 5539, 6961 (first gold rank: None)
  - GraphRAG top10: 1812, 2003, 1586, 1534, 1954, 1917, 1527, 1538 (first gold rank: None)

- Query: How does Correspondence Between Don Knuth and Peter van Emde Boas on Priority Deques 1977 [pdf] work?
  - Gold IDs: 
  - BM25 top10: 3898, 4231, 11412, 7428, 3896, 6886, 4384, 3894, 3897, 11417 (first gold rank: None)
  - GraphRAG top10: 1812, 2003, 1586, 1534, 1954, 1917, 1527, 1538 (first gold rank: None)

- Query: What is the update regarding Project Dropstone: A Neuro-Symbolic Runtime for Long-Horizon Engineering [pdf]?
  - Gold IDs: 
  - BM25 top10: 8541, 409, 8452, 8881, 3249, 8453, 143, 400, 8471, 3495 (first gold rank: None)
  - GraphRAG top10: 8954, 8955, 8956, 8961, 8962, 8963, 9115, 8958 (first gold rank: None)

- Query: Why is ChainReaction: Causal Chain-Guided Reasoning for Modular and Explainable Causal-Why Video Question Answering important?
  - Gold IDs: 8827
  - BM25 top10: 8827, 11391, 5228, 5212, 4968, 8765, 8766, 6094, 8828, 5205 (first gold rank: 1)
  - GraphRAG top10: 8827, 4968, 6097, 5212, 6094, 6095, 5279, 5205 (first gold rank: 1)

- Query: What is Correspondence Between Don Knuth and Peter van Emde Boas on Priority Deques 1977 [pdf] about?
  - Gold IDs: 
  - BM25 top10: 3898, 11382, 3489, 8701, 3897, 3894, 4231, 11412, 11286, 11466 (first gold rank: None)
  - GraphRAG top10: 1812, 2003, 1586, 1534, 1954, 1917, 1527, 1538 (first gold rank: None)

- Query: What are the main points in Project Dropstone: A Neuro-Symbolic Runtime for Long-Horizon Engineering [pdf]?
  - Gold IDs: 
  - BM25 top10: 8541, 409, 8452, 143, 8881, 3249, 400, 8453, 3072, 8731 (first gold rank: None)
  - GraphRAG top10: 8954, 8955, 8956, 8961, 8962, 8963, 9115, 8958 (first gold rank: None)

- Query: What did the article on Remote VAEs for decoding with Inference Endpoints 🤗 report?
  - Gold IDs: 4159
  - BM25 top10: 4159, 4160, 4162, 6066, 7659, 6946, 5144, 5195, 6955, 6854 (first gold rank: 1)
  - GraphRAG top10: 4159, 4160, 4162, 6066, 7659, 6070, 6073, 6946 (first gold rank: 1)

## Notes

- TopK used: 20
- GraphRAG mapping notes: {"used_doc_ids": false, "extracted_from_text": false, "doc_level": false}
