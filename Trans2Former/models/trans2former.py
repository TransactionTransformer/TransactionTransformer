import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
)
from fairseq.utils import safe_hasattr
from fairseq.meters import StopwatchMeter

from ..modules import init_trans2former_params, Trans2FormerGraphEncoder

logger = logging.getLogger(__name__)

from ..pretrain import load_pretrained_model

@register_model("trans2former")
class Trans2FormerModel(FairseqEncoderModel):
    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

        if getattr(args, "apply_trans2former_init", False):
            self.apply(init_trans2former_params)
        self.encoder_embed_dim = args.encoder_embed_dim
        if args.pretrained_model_name != "None":
            self.load_state_dict(load_pretrained_model(args.ckpt_dir, args.pretrained_model_name))
            if not args.load_pretrained_model_output_layer:
                self.encoder.reset_output_layer_parameters()

        # if args.performer_finetune:
        #     self.encoder.performer_finetune_setup()

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add model-specific arguments to the parser."""
        # Arguments related to dropout
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for" "attention weights",
        )
        parser.add_argument(
            "--act-dropout",
            type=float,
            metavar="D",
            help="dropout probability after" " activation in FFN",
        )

        # Arguments related to hidden states and self-attention
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )

        # Arguments related to input and output embeddings
        parser.add_argument(
            "--infeat-dim",
            type=int,
            metavar="N",
            help="input data dimension",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--share-encoder-input-output-embed",
            action="store_true",
            help="share encoder input" " and output embedding",
        )
        parser.add_argument(
            "--encoder-learned-pos",
            action="store_true",
            help="use learned positional embeddings in the encoder",
        )
        parser.add_argument(
            "--no-token-positional-embeddings",
            action="store_true",
            help="if set, disables positional embeddings" " (outside self attention)",
        )
        parser.add_argument(
            "--max-positions", type=int, help="number of positional embeddings to learn"
        )

        # Performer
        parser.add_argument("--performer", action="store_true", help="linearized self-attention with Performer kernel")
        parser.add_argument("--performer-nb-features", type=int, metavar="N",
                            help="number of random features for Performer, defaults to (d*log(d)) where d is head dim")
        parser.add_argument("--performer-feature-redraw-interval", type=int, metavar="N",
                            help="how frequently to redraw the projection matrix for Performer")
        parser.add_argument("--performer-generalized-attention", action="store_true",
                            help="defaults to softmax approximation, but can be set to True for generalized attention")
        parser.add_argument("--performer-finetune", action="store_true", 
                            help="load softmax checkpoint and fine-tune with performer")

        # Arguments related to parameters initialization
        parser.add_argument(
            "--apply-edgeformer-init",
            default=False,
            action="store_true",
            help="use custom param initialization for Edgeformer"
        )

        # misc params
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use"
        )
        parser.add_argument(
            "--encoder-normalize-before",
            default=False,
            action="store_true",
            help="apply layernorm before each encoder block"
        )
        parser.add_argument(
            "--pre-layernorm",
            default=False,
            action="store_true",
            help="apply layernorm before self-attention and ffn. Without this, post layernorm will be used"
        )

        parser.add_argument(
            "--encoding-method",
            type=str,
            default='sum',
            help="Encoding raw feature for the model"
        )
    def max_nodes(self):
        return self.encoder.max_nodes
    
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance"""
        # make sure all  arguemnts are present in older models
        base_architecture(args)

        if not safe_hasattr(args, "max_nodes"):
            args.max_nodes = args.tokens_per_sample
        
        logger.info(args)

        encoder = Trans2FormerEncoder(args)
        return cls(args, encoder)
    
    def forward(self, batched_data, **kwargs):
        return self.encoder(batched_data, **kwargs)
        

class Trans2FormerEncoder(FairseqEncoder):
    def __init__(self, args):
        super().__init__(dictionary=None)
        self.max_nodes = args.max_nodes
        self.graph_encoder = Trans2FormerGraphEncoder(args)

        self.share_input_output_embed = args.share_encoder_input_output_embed
        self.embed_out = None
        self.lm_output_learned_bias = None

        # Remove head is set to true druing fine-tuning
        self.load_softmax = not getattr(args, "remove_head", False)

        self.lm_head_transform_weight = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        self.activation_fn = utils.get_activation_fn(args.activation_fn)
        self.layer_norm = LayerNorm(args.encoder_embed_dim)

        self.lm_output_learned_bias = None # TODO Lifan: what is this?
        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))

            if not self.share_input_output_embed:
                self.embed_out = nn.Linear(
                    args.encoder_embed_dim, args.num_classes, bias=False
                )
            else:
                raise NotImplementedError
        
        self.is_evaluate = getattr(args, 'is_evaluate', False)
        self.is_generate = getattr(args, 'is_generate', False)
        self.is_finetune = args.task == 'finetune' and not self.is_generate
        if self.is_finetune:
            self.finetune_header = nn.Sequential(
                nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim, bias=False),
                nn.ReLU(),
                nn.Linear(args.encoder_embed_dim, args.num_classes, bias=False)
            )
        self.frozen = args.frozen

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        if self.embed_out is not None:
            self.embed_out.reset_parameters()

    def forward(self, batched_data,  perturb=None, masked_tokens=None, **unused):
        if self.is_finetune and self.frozen:
            with torch.no_grad():    
                inner_state, graph_rep = self.graph_encoder(
                    batched_data,
                    perturb=perturb,
                )
        else:
            inner_state, graph_rep = self.graph_encoder(
                batched_data,
                perturn=perturb
            )
        x = inner_state[-1].transpose(0, 1)
        
        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))
        if self.is_finetune:
            x = self.finetune_header(x)
        # project back to size of vocabulary
        if self.share_input_output_embed and hasattr(
            self.graph_encoder.embed_tokens, "weight"
        ):
            x = F.linear(x, self.graph_encoder.embed_tokens.weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)
        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias
        # if is_generate, output middle output
        if self.is_generate:
            return graph_rep
        else:
            return x

    # def performer_finetune_setup(self):
    #     self.graph_encoder.performer_finetune_setup()
    
    def max_nodes(self):
        """Maximum output length supported by the encoder"""
        return self.max_nodes
    
    def upgrade_state_dict_named(self, state_dict, name):
        if not self.load_softmax:
            for k in list(state_dict.keys()):
                if "embed_out.weight" in k or "lm_output_learned_bias" in k:
                    del state_dict[k]
        if self.is_finetune and not self.is_evaluate:
            state_dict['encoder.finetune_header.0.weight'] = self.state_dict()['finetune_header.0.weight']
            state_dict['encoder.finetune_header.2.weight'] = self.state_dict()['finetune_header.2.weight']

        return state_dict
    

@register_model_architecture("trans2former", "trans2former")    # register an model arch named "trans2former" into trans2former
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.act_dropout = getattr(args, "act_dropout", 0.0)

    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)

    args.infeat_dim = getattr(args, "infeat_dim", 10)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )

    args.performer = getattr(args, "performer", False)
    args.performer_finetune = getattr(args, "performer_finetune", False)
    args.performer_nb_features = getattr(args, "performer_nb_features", None)
    args.performer_feature_redraw_interval = getattr(args, "performer_feature_redraw_interval", 1000)
    args.performer_generalized_attention = getattr(args, "performer_generalized_attention", False)

    args.apply_egdeformer_init = getattr(args, "apply_edgeformer_init", False)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)


@register_model_architecture("trans2former", "trans2former_base")
def graphormer_base_architecture(args):
    if args.pretrained_model_name == "Name to add": # TODO: provide pretrained model later
        pass
    else:
        args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
        args.encoder_layers = getattr(args, "encoder_layers", 12)
        args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 32)
        args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768)
        args.dropout = getattr(args, "dropout", 0.0)
        args.attention_dropout = getattr(args, "attention_dropout", 0.1)
        args.act_dropout = getattr(args, "act_dropout", 0.1)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.apply_edgeformer_init = getattr(args, "apply_edgeformer_init", True)
    args.share_encoder_input_output_embed = getattr(
            args, "share_encoder_input_output_embed", False
        )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.pre_layernorm = getattr(args, "pre_layernorm", False)
    base_architecture(args)
    
    
@register_model_architecture("trans2former", "trans2former_slim")    # register an model arch named "trans2former_slim" into trans2former
def edgeformer_slim_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 80)

    args.encoder_layers = getattr(args, "encoder_layers", 12)
    
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 80)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.apply_egdeformer_init = getattr(args, "apply_edgeformer_init", True)
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.pre_layernorm = getattr(args, "pre_layernorm", False)
    base_architecture(args)

@register_model_architecture("trans2former", "trans2former_large")    # register an model arch named "trans2former_large" into trans2former
def edgeformer_large_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)

    args.encoder_layers = getattr(args, "encoder_layers", 24)
    
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 32)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.apply_egdeformer_init = getattr(args, "apply_edgeformer_init", True)
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.pre_layernorm = getattr(args, "pre_layernorm", False)
    base_architecture(args)

