import torch


def extract_cls_logits(logits, input_ids, tokenizer, prompt_len=0):
    '''
    :param logits: -> [batch, seq_len, vocab_size]
    :param batch:  -> [batch. seq_len]
    :param tokenizer:
    :return: -> [batch, vocab_size]
    '''
    b_idx, seq_idx = torch.where(input_ids == tokenizer.cls_token_id)
    cls_token_logit = logits[(b_idx, seq_idx + prompt_len)]
    return cls_token_logit


def extract_mask_logits(logits, input_ids, tokenizer, prompt_len=0):
    '''
    :param logits: -> [batch, seq_len, vocab_size]
    :param batch:  -> [batch. seq_len]
    :param tokenizer:
    :return: -> [batch, vocab_size]
    '''
    b_idx, seq_idx = torch.where(input_ids == tokenizer.mask_token_id)
    mask_token_logit = logits[(b_idx, seq_idx + prompt_len)]
    return mask_token_logit


def extract_soft_logits(logits, input_ids, tokenizer, prompt_len=0):
    soft_logits = logits[:, :prompt_len]
    return soft_logits.mean(1)


def extract_cls_hidden(hidden, input_ids, tokenizer, prompt_len=0):
    '''
    :param logits: -> [batch, seq_len, vocab_size]
    :param batch:  -> [batch. seq_len]
    :param tokenizer:
    :return: -> [batch, vocab_size]
    '''
    b_idx, seq_idx = torch.where(input_ids == tokenizer.cls_token_id)
    cls_token_hidden = hidden[(b_idx, seq_idx + prompt_len)]
    # cls_token_hidden = hidden[:, 0]
    return cls_token_hidden


def extract_mask_hidden(hidden, input_ids, tokenizer, prompt_len=0):
    '''
    :param logits: -> [batch, seq_len, vocab_size]
    :param batch:  -> [batch. seq_len]
    :param tokenizer:
    :return: -> [batch, vocab_size]
    '''
    b_idx, seq_idx = torch.where(input_ids == tokenizer.mask_token_id)
    mask_token_hidden = hidden[(b_idx, seq_idx + prompt_len)]
    return mask_token_hidden


def normalizerd_verbalizer_logits_for_sa(logit, token_ids):
    neg_logit = torch.stack([logit[:, y].mean(-1) for y in token_ids[0]], -1).mean(-1)
    pos_logit = torch.stack([logit[:, y].mean(-1) for y in token_ids[1]], -1).mean(-1)
    final_logit = torch.stack([neg_logit, pos_logit], -1)
    return final_logit


def normalizerd_verbalizer_logits_for_nli(logit, token_ids):
    neu_logit = torch.stack([logit[:, y].mean(-1) for y in token_ids[0]], -1).mean(-1)
    neg_logit = torch.stack([logit[:, y].mean(-1) for y in token_ids[1]], -1).mean(-1)
    pos_logit = torch.stack([logit[:, y].mean(-1) for y in token_ids[2]], -1).mean(-1)
    final_logit = torch.stack([neu_logit, neg_logit, pos_logit], -1)
    return final_logit
