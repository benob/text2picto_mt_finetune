# Finetuning a machine translation model on AAC pictograms

This repository contains code to finetune the nllb-200 model for the task of translating from French to a sequence of Arasaac pictograms and from pictograms back to French. Provided you have aligned texts, it can trivially be adapted to translate from and to another written language.

The finetuning script is based on https://github.com/hunterschep/nllb-200-600m-finetuning. The Arasaac lexicon has been downloaded from https://api.arasaac.org/v1/pictograms/all/en.

The trained model is available at https://huggingface.co/benoitfavre/nllb-200-distilled-600m_text2picto.

It is assumed that training data is a hugginface dataset with instances containing a "text" column and a "pictos" column.
Pictogram token embeddings are initialized from the mean of their associated lemmas embeddings. This boosts convergence quite a bit.

The script is configured to train on 80k steps which runs in ~10h on a single A100. The resulting model is saved in `./<model_id>_text2picto`. It can be uploaded with `hf upload <new-model_id> <directory>`.

Example inference code is provided for converting a parquet dataset of sentences to pictos. In includes a logits processor to enforce that only tokens from Arassac are generated.

Limitations:
- translation quality really depends on training data coverage, generalization is much worse than with a LLM
- only concepts from the Arasaac lexicon can be represented, meaning that most named entities will be generalized or hallucinated
- although the meaning of pictogram sequences is ambiguous, only one translation is generated

TODO:
- Remove the requirement for space tokens between picto tokens
