from funasr import AutoModel
from funasr.register import tables
import torch
import torch.nn.functional as F
from funasr.models.transformer.embedding import (
    SinusoidalPositionEncoder,
    StreamSinusoidalPositionEncoder,
)
from torch import nn

@tables.register("encoder_classes", "ConformerEncoderExport")
class ConformerEncoderExport(nn.Module):
    def __init__(
        self,
        model,
        max_seq_len=512,
        feats_dim=560,
        model_name="encoder",
        onnx: bool = True,
        ctc_linear: nn.Module = None,
    ):
        super().__init__()
        self.embed = model.embed
        if isinstance(self.embed, StreamSinusoidalPositionEncoder):
            self.embed = None
        self.model = model
        self.feats_dim = feats_dim
        self._output_size = model._output_size

        from funasr.utils.torch_function import sequence_mask
        self.make_pad_mask = sequence_mask(max_seq_len, flip=False)

        self.model_name = model_name
        self.ctc_linear = ctc_linear

        self.num_heads = model.encoders[0].self_attn.h
        self.hidden_size = model.encoders[0].self_attn.linear_out.out_features

    def prepare_mask(self, mask):
        mask_3d_btd = mask[:, :, None]
        if len(mask.shape) == 2:
            mask_4d_bhlt = 1 - mask[:, None, None, :]
        elif len(mask.shape) == 3:
            mask_4d_bhlt = 1 - mask[:, None, :]
        mask_4d_bhlt = mask_4d_bhlt * -10000.0
        return mask_3d_btd, mask_4d_bhlt

    def forward(self, speech: torch.Tensor, speech_lengths: torch.Tensor, online: bool = False):
        if not online:
            speech = speech * self._output_size**0.5
        print(speech.shape)
        mask = self.make_pad_mask(speech_lengths)
        mask = self.prepare_mask(mask)

        if self.embed is None:
            xs_pad = speech
        else:
            xs_pad = self.embed(speech, mask[0])

        for layer in self.model.encoders:
            xs_pad, mask = layer(xs_pad, mask)

        xs_pad = self.model.after_norm(xs_pad)

        if self.ctc_linear is not None:
            xs_pad = self.ctc_linear(xs_pad)
            xs_pad = F.softmax(xs_pad, dim=2)

        return xs_pad, speech_lengths

    def get_output_size(self):
        return self.model.encoders[0].self_attn.linear_out.out_features

    def get_dummy_inputs(self):
        feats = torch.randn(1, 100, self.feats_dim)
        return feats

    def get_input_names(self):
        return ["feats"]

    def get_output_names(self):
        return ["encoder_out", "encoder_out_lens", "predictor_weight"]

    def get_dynamic_axes(self):
        return {
            "feats": {1: "feats_length"},
            "encoder_out": {1: "enc_out_length"},
            "predictor_weight": {1: "pre_out_length"},
        }


# print(tables.encoder_classes)
model = AutoModel(
    model="/opt/ai_users/abdurakhim/paraformer_train/checkpoints/paraformer_uzbek_natural_v1",
    model_path="/opt/ai_users/abdurakhim/paraformer_train/checkpoints/paraformer_uzbek_natural_v1/model.pt.best", 
    config_path="/opt/ai_users/abdurakhim/paraformer_train/checkpoints/paraformer_uzbek_natural_v1", 
    config_name="config.yaml"
)

result = model.generate(
            input="/opt/ai_users/abdurakhim/paraformer_train/test.wav",
            batch_size_s=300
        )
# res = model.export(type="onnx", quantize=False, opset_version=20, predictor="CifPredictorV2",device='cpu')