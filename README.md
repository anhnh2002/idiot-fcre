
---
## BERT

### Running with `K = 1`

1. Navigate to the BERT directory:
   ```
   cd bert
   ```
2. Run the TACRED training:
   ```
   python train_tacred.py --task_name Tacred --num_k 5 >> tacred_1_2_05_1_des1.txt
   ```
3. Run the FewRel training:
   ```
   python train_fewrel.py --task_name FewRel --num_k 5 >> fewrel_1_1_1_1_des1.txt
   ```

### Running with `K > 1`

1. Modify the `num_gen_augment` value in the `config.ini` file to match the expected `K`.
2. Update the `config.relation_description` path in both `train_multi_k_fewrel.py` and `train_multi_k_tacred.py` to the correct path corresponding to the expected `K`.
3. Run the TACRED training:
   ```
   python train_multi_k_tacred.py --task_name Tacred --num_k 5 >> tacred_1_2_05_1_desk.txt
   ```
4. Run the FewRel training:
   ```
   python train_multi_k_fewrel.py --task_name FewRel --num_k 5 >> fewrel_1_1_1_1_desk.txt
   ```

---

## LLM2Vec

### Setup

1. Navigate to the `llm2vec` directory:
   ```
   cd llm2vec
   ```
2. Install necessary packages:
   ```
   !pip install transformers==4.40.0 torch==2.3.0 scikit-learn ==1.4.2
   !pip install llm2vec==0.2.2
   !pip install flash-attn --no-build-isolation
   ```
3. Log in to Hugging Face:
   ```
   !huggingface-cli login --token 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
   ```

### Running with `K = 1`

1. Run the TACRED training:
   ```
   python train_tacred.py --task_name Tacred --num_k 5 >> tacred_1_2_05_1_llm2vec.txt
   ```
2. Run the FewRel training:
   ```
   python train_fewrel.py --task_name FewRel --num_k 5 >> fewrel_1_1_1_1_llm2vec.txt
   ```
ours setting in llm2vec only for k = 1
Requires: Python >=3.8
--- 

