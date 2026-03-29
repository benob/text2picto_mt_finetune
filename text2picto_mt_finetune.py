# Code for finetuning NLLB-200 on text2picto and picto2text tasks
# Based on https://github.com/hunterschep/nllb-200-600m-finetuning

import wandb
wandb.login()

import json
import numpy as np
import random
import os
from tqdm.auto import tqdm, trange

from stop_words import get_stop_words
from unidecode import unidecode
import torch
import datasets
from datasets import concatenate_datasets, load_dataset, Dataset
from torch.amp import autocast, GradScaler
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer, Adafactor, get_constant_schedule_with_warmup

model_id = 'facebook/nllb-200-distilled-600m'
#model_id = 'facebook/nllb-200-distilled-1.3B'
#model_id = 'facebook/nllb-200-1.3B'
#model_id = 'facebook/nllb-200-3.3B'

# datasets on which we are going to train
dataset_ids = ['benoitfavre/llmpicto-commonvoice-v2', 'benoitfavre/picto-arasaac-random']

device = torch.device('mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

steps          = 80000
batch_size     = 128
max_length     = 128
learning_rate  = 1e-4
warmup_steps   = 500
weight_decay   = 1e-3
max_grad_norm  = 1.0
grad_accum     = 1
p_src2tgt      = 0.5         # P(fr→picto), else picto→fr
eval_interval  = 100
log_interval   = 10
use_emb_init   = True

src_lid  = "fra_Latn"
tgt_lid  = "picto"

picto_prefix = '\uE000' # first char of private unicode area

def augment_text(text, p_augment=.25): # fix some instances with capitalization and punctuation, also strip accents
    text = text.strip()
    if len(text) > 0:
        if text[-1].isalnum() and random.random() < p_augment: text += '.'
        if text[0].islower() and random.random() < p_augment: text = text.capitalize()
        if random.random() < p_augment: text = unidecode(text)
    return text

def convert_instance(instance): # add space before pictos for clean tokenization
  return {
    src_lid: instance['text'],
    tgt_lid: picto_prefix + (' '.join(instance['pictos']) if isinstance(instance['pictos'], list) else instance['pictos']).replace(' ', picto_prefix),
  }

stop_words = set(get_stop_words('fr'))

def noise_filter(instance):
    text = instance[src_lid]
    picto = instance[tgt_lid]
    num_content_words = len([x for x in text.lower().strip().split() if x not in stop_words])
    num_picto = len(picto.replace(picto_prefix, ' ').strip().split())
    return num_picto >= num_content_words / 2

# training data was generated from english lexicon
with open('arasaac-en.json') as fp:
    lexicon = json.load(fp)
    added_vocab = [picto_prefix + str(entry['_id']) for entry in lexicon]
    lemmas = {
        picto_prefix + str(entry['_id']):
            [kw['keyword'] for kw in entry['keywords'] if 'keyword' in kw] 
            + [kw['plural'] for kw in entry['keywords'] if 'plural' in kw] 
        for entry in lexicon
    }

    # make dataset from lexicon
    lexicon_entries = []
    for k, v in lemmas.items():
        for lemma in v:
            lexicon_entries.append({
                src_lid: lemma,
                tgt_lid: picto_prefix + k,
            })
    lexicon_dataset = Dataset.from_list(lexicon_entries)

train_data = concatenate_datasets([lexicon_dataset] + [load_dataset(dataset_id, split='train').map(convert_instance) for dataset_id in dataset_ids]).filter(noise_filter)

valid_data = load_dataset(dataset_ids[0], split='validation').map(convert_instance)
test_data = load_dataset(dataset_ids[0], split='test').map(convert_instance)

print(len(train_data), len(valid_data), len(test_data))

# add picto ids to tokenizer
tokenizer = NllbTokenizer.from_pretrained(model_id)
max_token_id = max(tokenizer.get_vocab().values())

tokenizer.add_tokens(added_vocab)

# add token for target language
tokenizer.add_special_tokens({"additional_special_tokens": [tgt_lid]})

# init embeddings for new (picto) tokens
def init_model(model_id):
  model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
  model.resize_token_embeddings(len(tokenizer))

  if use_emb_init:
      emb = model.get_input_embeddings().weight.data
      lm_head = model.lm_head.weight.data

      for token in added_vocab:
        new_id = tokenizer.convert_tokens_to_ids(token)
        if new_id > max_token_id:
          # init new embeddings with mean of lemmas associated with picto
          ids_old = tokenizer.encode(lemmas[token], add_special_tokens=False)
          ids_old = list(set(sum(ids_old, []))) # average over all lemmas
          emb[new_id] = emb[ids_old].mean(0)
          lm_head[new_id] = lm_head[ids_old].mean(0)

      new_lang_id = tokenizer.convert_tokens_to_ids(tgt_lid)
      old_lang_id = tokenizer.convert_tokens_to_ids('eng_Latn') # init lid from English
      emb[new_lang_id] = emb[old_lang_id]
      lm_head[new_lang_id] = lm_head[old_lang_id]

  model = model.to(device)
  return model

print('check picto tokens', [tokenizer.decode(x) for x in tokenizer.encode(train_data[42]['picto'])])

def encode_batch(tokenizer, src_texts, tgt_texts, src_code, max_length, device):
    src_texts = ["" if x is None else str(x) for x in src_texts]
    tgt_texts = ["" if x is None else str(x) for x in tgt_texts]

    # only modify instances if they are in the direction text2picto
    if src_code == src_lid: src_texts = [augment_text(x, p_augment=.25) for x in src_texts]

    tokenizer.src_lang = src_code

    enc = tokenizer(
        src_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_token_type_ids=False,
    )

    lab = tokenizer(
        tgt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=False,
        return_token_type_ids=False,
        add_special_tokens=False,  # no BOS/EOS/LID on labels
    )

    labels = lab["input_ids"]
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id; EOS is required for seq2seq training.")

    # put EOS right after last non-pad token
    with torch.no_grad():
        lengths = (labels != pad_id).sum(dim=1)          # [B]
        L = labels.size(1)
        pos = torch.clamp(lengths, max=L - 1)
        rows = torch.arange(labels.size(0), dtype=torch.long)
        labels[rows, pos] = eos_id

    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    labels = labels.to(device)
    labels[labels == pad_id] = -100  # ignore_index for loss

    return input_ids, attn_mask, labels


def training_step(tokenizer, model, src_texts, tgt_texts, src_code, max_length, device, scaler, forced_bos_id):

    input_ids, attn_mask, labels = encode_batch(
        tokenizer, src_texts, tgt_texts, src_code, max_length, device
    )

    pad_id = tokenizer.pad_token_id
    dec_in = labels.clone()
    dec_in = dec_in.masked_fill(dec_in == -100, pad_id)
    bos_col = torch.full((dec_in.size(0), 1), forced_bos_id,
                         device=device, dtype=dec_in.dtype)
    decoder_input_ids = torch.cat([bos_col, dec_in[:, :-1]], dim=1)

    use_amp = scaler is not None and device.type == "cuda"

    if use_amp:
        with autocast(device_type=device.type):
            out = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
            )
            loss = out.loss
        scaler.scale(loss).backward()
    else:
        out = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
        loss = out.loss
        loss.backward()

    return loss.detach()

#note that we sample a single batch for computing valid loss, it's noisy but comparable to training loss
@torch.no_grad()
def eval_loss(tokenizer, model, valid_data, src_lid, tgt_lid, max_length, device, forced_bos_id, sample_size=64):
    model.eval()

    def compute_loss(src_texts, tgt_texts, src_code):
        input_ids, attn_mask, labels = encode_batch(
            tokenizer, src_texts, tgt_texts, src_code, max_length, device
        )

        pad_id = tokenizer.pad_token_id
        dec_in = labels.clone()
        dec_in = dec_in.masked_fill(dec_in == -100, pad_id)
        bos_col = torch.full((dec_in.size(0), 1), forced_bos_id,
                             device=device, dtype=dec_in.dtype)
        decoder_input_ids = torch.cat([bos_col, dec_in[:, :-1]], dim=1)

        use_amp = scaler is not None and device.type == "cuda"

        if use_amp:
            with autocast(device_type=device.type):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    decoder_input_ids=decoder_input_ids,
                    labels=labels,
                )
                loss = out.loss
        else:
            out = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
            )
            loss = out.loss

        return loss.detach()

    indices = np.random.randint(0, len(valid_data), size=min(sample_size, len(valid_data)))
    sample = valid_data[indices]
    src_samples = sample[src_lid]
    tgt_samples = sample[tgt_lid]

    loss_forward = compute_loss(src_samples, tgt_samples, src_lid)  # fr→picto
    loss_backward = compute_loss(tgt_samples, src_samples, tgt_lid)  # picto→fr

    model.train()
    return loss_forward, loss_backward

model = init_model(model_id)

src_id   = tokenizer.convert_tokens_to_ids(src_lid)
tgt_id   = tokenizer.convert_tokens_to_ids(tgt_lid)

optimizer = Adafactor(
    (p for p in model.parameters() if p.requires_grad),
    scale_parameter=False,
    relative_step=False,
    lr=learning_rate,
    clip_threshold=1.0,
    weight_decay=weight_decay,
)

scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

scaler = GradScaler(device.type if device.type == "cuda" else "cpu",
                    enabled=(device.type == "cuda"))

model.train()
optimizer.zero_grad(set_to_none=True)

train_losses = []
pbar = trange(steps, desc="Training", dynamic_ncols=True)

wandb_project='text2picto_mt_finetune'

wandb_config = {
    'model_id': model_id,
    'dataset_ids': dataset_ids,
    'device': str(device),
    'steps': steps,
    'batch_size': batch_size,
    'max_length': max_length,
    'learning_rate': learning_rate,
    'warmup_steps': warmup_steps,
    'weight_decay': weight_decay,
    'max_grad_norm': max_grad_norm,
    'grad_accum': grad_accum,
    'p_src2tgt': p_src2tgt,
    'use_emb_init': use_emb_init,
}

with wandb.init(project=wandb_project, config=wandb_config) as run:
    for step in pbar:
        try:
            idx = np.random.randint(0, len(train_data), size=batch_size)
            batch = train_data[idx]

            if random.random() < p_src2tgt:
                src_texts = batch[src_lid]
                tgt_texts = batch[tgt_lid]
                src_code  = src_lid
                forced_id = tgt_id
                direction = 'forward'
            else:
                src_texts = batch[tgt_lid]
                tgt_texts = batch[src_lid]
                src_code  = tgt_lid
                forced_id = src_id
                direction = 'backward'

            loss = training_step(
                tokenizer,
                model,
                src_texts,
                tgt_texts,
                src_code,
                max_length,
                device,
                scaler,
                forced_bos_id=forced_id,
            )
            train_losses.append(loss.item())

            if (step + 1) % grad_accum == 0:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            to_log = {"train_loss": train_losses[-1], f"train_loss_{direction}": train_losses[-1]}
            if (step + 1) % log_interval == 0:
                recent = np.mean(train_losses[-log_interval:])
                pbar.set_postfix(loss_train=f"{recent:.4f}")

            if (step + 1) % eval_interval == 0 and len(valid_data):
                loss_forward, loss_backward = eval_loss(tokenizer, model, valid_data, src_lid, tgt_lid, max_length, device, forced_id, sample_size=batch_size)
                pbar.set_postfix(loss_f=f"{loss_forward:.4f}", loss_b=f"{loss_backward:.4f}")
                to_log.update({"val_loss_forward": loss_forward, "val_loss_backward": loss_backward, 'val_loss': (loss_forward + loss_backward) / 2})
            run.log(to_log)

        except RuntimeError as e:
            # simple OOM guard for Colab
            if "out of memory" in str(e).lower():
                print(f"\n[OOM] step {step}: {e}. Clearing cache and continuing.")
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise

print("\nTraining done.")

repo_id = model_id.split('/')[-1] + '_text2picto_v3'
tokenizer.save_pretrained(repo_id)
model.save_pretrained(repo_id)

