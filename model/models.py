import torch.nn as nn
from base import BaseModel
from torch import Tensor
from model.model_utils.tokenizer_utils import *
from transformers.utils import ModelOutput
from torch.optim import AdamW


class Domain_classifier(nn.Module):
    def __init__(self, in_dim, num_domains, drop):
        super(Domain_classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(in_dim, num_domains)
        )

    def forward(self, x):
        x = self.net(x)
        return x


from torch.autograd import Function


class GRL(Function):
    # Gradient Reversal Layer for adversarial loss backprop
    @staticmethod
    def forward(ctx, x, alpha):
        # Store context for backprop
        ctx.alpha = alpha

        # Forward pass is a no-op
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass is just to -alpha the gradient
        output = grad_output.neg() * ctx.alpha
        # Must return same number as inputs to forward()
        return output, None


class AdPXT(BaseModel):
    def __init__(self, plm_model, tokenizer, global_length=8, context_length=2, model_lr=1e-2, w_decay=1e-2, drop=0.1,
                 task='nli', adversarial=True, num_domains=4, alpha=0.1):
        super().__init__()
        self.task = task
        self.is_adv = adversarial
        self.num_domains = num_domains
        self.alpha = alpha
        self.config = plm_model.config

        self.roberta_embeddings = plm_model.roberta.embeddings
        self.roberta_encoder = plm_model.roberta.encoder
        self.tokenizer = tokenizer

        self.global_length = global_length
        self.context_length = context_length

        self.n_layer = self.config.num_hidden_layers
        self.n_head = self.config.num_attention_heads
        self.n_embd = self.config.hidden_size // self.config.num_attention_heads
        self.token_max_len = tokenizer.model_max_length
        self.prefix_dropout = nn.Dropout(drop)

        if task == 'nli':
            self.task_head = nn.Sequential(
                nn.Dropout(drop),
                nn.Linear(self.config.hidden_size, 3)
            )
        elif task == 'sa':
            self.task_head = nn.Sequential(
                nn.Dropout(drop),
                nn.Linear(self.config.hidden_size, 1)
            )

        self.global_prefixes = nn.Embedding(self.global_length,
                                            self.config.num_hidden_layers * 2 * self.config.hidden_size)
        self.global_token_idx = torch.arange(self.global_length).long()
        self.query_tokens = nn.Embedding(self.context_length * self.n_layer, self.config.hidden_size)
        self.query_token_idx = torch.arange(self.context_length * self.n_layer).long()

        self.model_optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                     lr=model_lr,
                                     weight_decay=w_decay)

    def get_extended_attention_mask(
            self, attention_mask: Tensor, device: torch.device = None,
            dtype: torch.float = None
    ) -> Tensor:

        if dtype is None:
            for p in self.parameters():
                dt = p.dtype
                break
            dtype = dt
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape) or attention_mask (shape {attention_mask.shape})"
            )

        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def emb_to_prefix(self, prefix, length, batch_size):
        past_key_values = prefix.view(
            batch_size,
            length,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.prefix_dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self, input_ids, attention_mask):
        seq_len = input_ids.shape[-1]
        max_len = self.token_max_len - self.global_length - self.context_length * 2

        if seq_len > max_len:
            input_ids = input_ids[:, :max_len]
            attention_mask = attention_mask[:, :max_len]

        batch_device = input_ids.device
        batch_size, seq_length = input_ids.size()
        text_embedding = self.roberta_embeddings(input_ids=input_ids)

        context_hiddens = torch.tensor([]).to(batch_device)
        cls_hiddens = torch.tensor([]).to(batch_device)

        global_batch_tokens = self.global_token_idx.unsqueeze(0).expand(batch_size, -1).to(batch_device)
        global_attn_mask = torch.ones(batch_size, self.global_length).to(batch_device)
        global_prefixes = self.global_prefixes(global_batch_tokens)
        global_past_key_values = self.emb_to_prefix(global_prefixes, self.global_length, batch_size)

        query_batch_tokens = self.query_token_idx.unsqueeze(0).expand(batch_size, -1).to(batch_device)
        context_attn_mask = torch.ones(batch_size, self.context_length).to(batch_device)
        query_attn_mask = torch.zeros(batch_size, self.context_length).to(batch_device)
        queries = self.query_tokens(query_batch_tokens)

        first_mask = self.get_extended_attention_mask(
            torch.cat([global_attn_mask, query_attn_mask, attention_mask], 1))
        mask = self.get_extended_attention_mask(
            torch.cat([global_attn_mask, context_attn_mask, query_attn_mask, attention_mask], 1))

        input_hidden = text_embedding
        for layer_idx in range(self.n_layer):
            if layer_idx == 0:
                g_token = global_past_key_values[layer_idx]
                query = queries[:, layer_idx * self.context_length: (layer_idx + 1) * self.context_length]
                layer_module = self.roberta_encoder.layer[layer_idx]
                layer_outputs = layer_module(
                    torch.cat([query, input_hidden], 1), first_mask, past_key_value=g_token)
                hidden_states = layer_outputs[0]

            else:
                g_token = global_past_key_values[layer_idx]
                query = queries[:, layer_idx * self.context_length: (layer_idx + 1) * self.context_length]
                layer_module = self.roberta_encoder.layer[layer_idx]
                layer_key = layer_module.attention.self.key
                layer_value = layer_module.attention.self.value

                c_k = layer_module.attention.self.transpose_for_scores(layer_key(context_hidden))
                c_v = layer_module.attention.self.transpose_for_scores(layer_value(context_hidden))
                context_key_value = torch.stack([c_k, c_v], 0)

                layer_outputs = layer_module(
                    torch.cat([query, text_hidden], 1), mask,
                    past_key_value=torch.cat([g_token, context_key_value], -2))
                hidden_states = layer_outputs[0]
            context_hidden = hidden_states[:, :self.context_length]
            text_hidden = hidden_states[:, self.context_length:]
            context_hiddens = torch.cat([context_hiddens, context_hidden], 1)
            cls_hiddens = torch.cat([cls_hiddens, text_hidden[:, :1]], 1)

        cls_hidden = text_hidden[:, 0]
        task_logit = self.task_head(cls_hidden)

        if self.is_adv:
            reversal_feature = GRL.apply(cls_hidden, self.alpha)

            return ModelOutput(
                task_logit=task_logit,
                reversal_feature=reversal_feature,
                hidden_states=hidden_states,
                context_prefixes=context_hiddens,
                cls_hiddens=cls_hiddens
            )
        else:
            return ModelOutput(
                task_logit=task_logit,
                hidden_states=hidden_states,
                context_prefixes=context_hiddens,
                cls_hiddens=cls_hiddens
            )

