import torch.nn.utils.rnn


def batch_collate_for_roberta(samples):
    input_ids = []
    attention_masks = []
    labels = []
    domain_idxs = []
    for input_id, attention_mask, label, domain_idx in samples:
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        labels.append(label)
        domain_idxs.append(domain_idx)
    # for roberta case
    padded_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=1)
    padded_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return padded_ids, padded_masks, torch.tensor(labels), torch.tensor(domain_idxs)

