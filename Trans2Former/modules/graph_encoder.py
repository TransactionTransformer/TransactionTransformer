from typing import Optional, Tuple

import torch
import torch.nn as nn

from fairseq.modules import FairseqDropout, LayerDropModuleList, LayerNorm
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

from .performer import ProjectionUpdater
from .multihead_attention import MultiheadAttention
from .graph_encoder_layer import Trans2FormerGraphEncoderLayer


def init_trans2former_params(module):
    """
    Initialize the weights specific to the EdgeFormer Model.
    """

    def normal_(data):
        # wtih FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class Trans2FormerGraphEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_edges = cfg.max_edges
        self.num_encoder_layers = cfg.encoder_layers
        self.num_attention_heads = cfg.num_attention_heads
        embed_scale: float = None,
        export: bool = False
        traceable: bool = False
        q_noise: float = 0.0
        qn_block_size: int = 8

        
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.layerdrop = cfg.layerdrop
        self.embedding_dim = cfg.embedding_dim
        self.apply_trans2former_init = cfg.apply_trans2former_init
        self.encoding_method = cfg.encoding_method
        self.traceable = getattr(cfg, 'traceable', False)
        if self.task == 'pretrain':
            if self.encoding_method == 'sum':
                from .tokenizer_pretrain_sum import PretrainEmbedding
                self.graph_edge_feature = PretrainEmbedding(
                    num_nodes=1024,
                    num_edges=self.num_edges,
                    rand_node_id=False,
                    rand_node_id_dim=64,
                    orf_node_id=True,
                    orf_node_id_dim=64,
                    type_id=True,
                    etype_id=True,
                    func_id=False,
                    token_id=True,
                    hidden_dim=self.embedding_dim,
                    n_layers=self.num_encoder_layers
                )
            elif self.encoding_method == 'concat':
                from .tokenizer_pretrain_concat import PretrainEmbedding
                self.graph_edge_feature = PretrainEmbedding(
                    hidden_dim=self.embedding_dim,
                    n_layers=self.num_encoder_layers,
                    num_heads=cfg.num_attention_heads,
                    mer_ratio=0.15,
                    mer_randomize_ratio=[0.8,0.15,0.15],
                    task='pretrain',
                    pretrain_task='NCP',
                    layer_norm_eps=1e-15,
                    dropout=0.1,
                    rand_node_id=False,
                    rand_node_id_dim=64,
                    orf_node_id=True,
                    orf_node_id_dim=64,
                    type_id=True,
                    etype_id=True,
                    func_id=False,
                    token_id=True,
                    attention_type='full_attention'
                )
            else:
                raise NotImplementedError
        else:
            if self.encoding_method == 'sum':
                from .tokenizer_finetune_sum import FinetuneEmbedding 
                self.graph_edge_feature = FinetuneEmbedding(
                    num_nodes=1024,
                    num_edges=self.num_edges,
                    rand_node_id=False,
                    rand_node_id_dim=64,
                    orf_node_id=True,
                    orf_node_id_dim=64,
                    type_id=True,
                    etype_id=True,
                    func_id=False,
                    token_id=True,
                    hidden_dim=self.embedding_dim,
                    n_layers=self.num_encoder_layers, 
                )
            elif self.encoding_method == 'concat':
                from .tokenizer_finetune_concat import FinetuneEmbedding
                self.graph_edge_feature = FinetuneEmbedding(
                    hidden_dim=self.embedding_dim,
                    n_layers=self.num_encoder_layers,
                    num_heads=self.num_attention_heads,
                    mer_ratio=0.15,
                    mer_randomize_ratio=[0.8,0.15,0.15],
                    task='pretrain',
                    pretrain_task='NCP',
                    layer_norm_eps=1e-15,
                    dropout=0.1,
                    rand_node_id=False,
                    rand_node_id_dim=64,
                    orf_node_id=True,
                    orf_node_id_dim=64,
                    type_id=True,
                    etype_id=True,
                    func_id=False,
                    token_id=True,
                    hop_id=True, # for concat encoding
                    bnts=True, # for concat encoding
                    attention_type='full_attention',
                )
            else:
                raise NotImplementedError
        
        self.performer_finetune = cfg.performer_finetune
        self.embed_scale = embed_scale
        
        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                q_noise,
                qn_block_size
            )
        else:
            self.quant_noise = None

        if cfg.encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None
        
        if cfg.pre_layernorm:
            self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)
        
        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])

        self.cached_performer_options = None
        if self.performer_finetune:
            assert self.performer
            self.cached_performer_options = (
                cfg.performer_nb_features,
                cfg.performer_generalized_attention,
                cfg.performer_auto_check_redraw,
                cfg.performer_feature_redraw_interval
            )
            self.performer = False
            cfg.performer = False
            cfg.performer_nb_features = None
            cfg.performer_generalized_attention = False
            cfg.performer_auto_check_redraw = False
            cfg.performer_feature_redraw_interval = None

        self.layers.extend(
            [
                self.build_edgeformer_graph_encoder_layer(cfg)
                for _ in range(cfg.num_encoder_layers)
            ]
        )

        # Apply initialization of model params after building the model
        if self.apply_tran2former_init:
            self.apply(init_trans2former_params)
        
        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False
        
        if cfg.freeze_embeddings:
            raise NotImplementedError("Freezing embeddings is not implemented yet.")
        
        for layer in range(cfg.n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

        if cfg.performer:
            # keeping track of when to redraw projections for all attention layers
            self.performer_auto_check_redraw = cfg.performer_auto_check_redraw
            self.performer_proj_updater = ProjectionUpdater(self.layers, cfg.performer_feature_redraw_interval)

    def performer_fix_projection_matrices_(self):
        self.performer_proj_updater.feature_redraw_interval = None

    def performer_finetune_setup(self):
        assert self.performer_finetune
        (
            performer_nb_features,
            performer_generalized_attention,
            performer_auto_check_redraw,
            performer_feature_redraw_interval
        ) = self.cached_performer_options

        for layer in self.layers:
            layer.performer_finetune_setup(performer_nb_features, 
                                           performer_generalized_attention)
        self.performer = True
        self.performer_auto_check_redraw = performer_auto_check_redraw
        self.performer_proj_updater = ProjectionUpdater(self.layers, performer_feature_redraw_interval)
        
    def build_edgeformer_graph_encoder_layer(self, cfg):
        return Trans2FormerGraphEncoderLayer(
            embedding_dim=cfg.embedding_dim,
            ffn_embedding_dim=cfg.encoder_ffn_embed_dim,
            num_attention_heads=cfg.num_attention_heads,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            activation_dropout=cfg.activation_dropout,
            activation_fn=cfg.activation_fn,
            performer=cfg.performer,
            performer_nb_features=cfg.performer_nb_features,
            performer_generalized_attention=cfg.performer_generalized_attention,
            export=getattr(cfg, 'export', False),
            pre_layernorm=cfg.pre_layernorm,
            q_noise=0.0,
            qn_block_size=8,
        )

    def forward(
        self,
        batched_data,
        perturb=None,
        last_state_only: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # compute padding mask. This is needed for multi-head atttention
        if token_embeddings is not None:
            raise NotImplementedError
        else:
            ret = self.graph_edge_feature(batched_data, perturb)
        if self.encoding_method == 'concat':
            x = ret['padded_feature']
            padding_mask= ret['padding_mask']
        elif self.encoding_method == 'sum':
            x, padding_mask, padded_index = ret
        
        # x: B x T x C
        # attn_bias = self.graph_attn_bias(batched_data)

        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x) 

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0,1)
        
        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        if attn_mask is not None:
            raise NotImplementedError
        
        for i, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                self_attn_bias=None,
            )
            if not last_state_only:
                inner_states.append(x)
        
        graph_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]
        
        if self.traceable:
            return torch.stack(inner_states), graph_rep
        else:
            return inner_states, graph_rep