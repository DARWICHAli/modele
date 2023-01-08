from .transformer import Transformer
from .captioning_model import CaptioningModel
import importlib

def model_factory(args):
    Transformer = importlib.import_module('models.'+args.model).Transformer
    TransformerEncoder = importlib.import_module('models.'+args.model).TransformerEncoder
    TransformerDecoderLayer = importlib.import_module('models.'+args.model).TransformerDecoderLayer
    if args.d_m <= 0:
        ScaledDotProductAttention = importlib.import_module('models.'+args.model).ScaledDotProductAttention
    else:
        ScaledDotProductAttention = importlib.import_module('models.'+args.model).ScaledDotProductAttentionMemory

    return Transformer, TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention
