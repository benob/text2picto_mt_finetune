import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from datasets import load_dataset, Dataset, disable_progress_bar
disable_progress_bar()
from tqdm import tqdm

def main(output_path: str, dataset_path: str, model_id: str = 'benoitfavre/nllb-200-distilled-600m_text2picto', device: str='cuda', batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)

    # filter predictions to only output pictograms
    class RestrictTokensProcessor(LogitsProcessor):
        def __init__(self, allowed_token_ids):
            self.allowed_token_ids = set(allowed_token_ids)
            self.mask = None

        def __call__(self, input_ids, scores):
            # scores: (batch_size, vocab_size)
            if self.mask is None or self.mask.shape != scores.shape:
                self.mask = scores.new_full(scores.shape, float("-inf"))
                for token_id in self.allowed_token_ids:
                    self.mask[:, token_id] = 0  # keep allowed tokens
            return scores + self.mask

    picto_token_ids = RestrictTokensProcessor(list(tokenizer._added_tokens_decoder.keys()) + tokenizer.all_special_ids + tokenizer.encode(' '))

    @torch.no_grad()
    def gen_dir(texts, from_lid, to_lid, max_length=128):
            tokenizer.src_lang = from_lid
            enc = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            forced_id = tokenizer.convert_tokens_to_ids(to_lid)
            with torch.no_grad():
              gen_out = model.generate(
                  **enc,
                  num_beams=1,
                  #max_new_tokens=48,
                  min_new_tokens=2,
                  no_repeat_ngram_size=3,
                  repetition_penalty=1.2,
                  #length_penalty=1.05,
                  forced_bos_token_id=forced_id,
                  decoder_start_token_id=forced_id,
                  eos_token_id=tokenizer.eos_token_id,
                  pad_token_id=tokenizer.pad_token_id,
                  logits_processor=LogitsProcessorList([picto_token_ids]) if to_lid == 'picto' else None,
              )
            if to_lid == 'picto':
                return [' '.join([tokenizer.decode(x) for x in sequence if x not in tokenizer.all_special_ids + [tokenizer.encode(' ')]]) for sequence in gen_out]
            else:
                return tokenizer.batch_decode(gen_out, skip_special_tokens=True)




    dataset = Dataset.from_parquet(dataset_path)
    total = sum([len(instance['simplified']) for instance in dataset])

    def iterator():
        batch = []
        for i, instance in enumerate(dataset):
            for j, sentence in enumerate(instance['simplified']):
                batch.append({'id': f'{os.path.basename(dataset_path)}:{i}:{j}', 'text': sentence})
                if len(batch) == batch_size:
                    yield batch
                    batch = []
        if len(batch) > 0:
            yield batch
    
    output = []
    for batch in tqdm(iterator(), total=total // batch_size):
        result = gen_dir([x['text'].strip().lower() for x in batch], 'fra_Latn', 'picto')
        for instance, picto in zip(batch, result):
            instance['pictos'] = picto.split()
            output.append(instance)
            if len(output) % 1000 == 0:
                Dataset.from_list(output).to_parquet(output_path)

    Dataset.from_list(output).to_parquet(output_path)

if __name__ == '__main__':
    import fire
    fire.Fire(main)

