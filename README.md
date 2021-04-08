# Transformer

This example trains a Transformer on a language modeling task. By default, the training script uses the Wikitext-2 dataset, provided. The trained model can then be used by the generate script to generate new text.

```bash
python main.py --cuda --epochs 6 --model Transformer --lr 5   
                                           # Train a Transformer model on Wikitext-2 with CUDA
python generate.py --cuda --model Transformer
                                           # Generate samples from the trained Transformer model.
```
