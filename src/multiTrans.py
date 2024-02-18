"""
This code is mostly a modification of the original code from the hugginface library:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/encoder_decoder/modeling_encoder_decoder.py
"""

from dataclasses import dataclass
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.data as data
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from typing import List, Optional, Tuple, Union, Any, Dict
from torch.utils.data._utils.collate import default_collate
from sklearn.metrics import roc_auc_score

from transformers import BertModel, PretrainedConfig, AutoModelForCausalLM
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertOnlyMLMHead,
    SequenceClassifierOutput,
)
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.encoder_decoder.configuration_encoder_decoder import (
    EncoderDecoderConfig,
)
import warnings
import copy
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### DATA


def generate_negative(df, neg_per_pos=5):
    """
    Generate negative data by randomly sampling from the dataset.
    For every positive datapoint, we resample the TCRs among tcrs not binding the same epitope until we have neg_per_pos negative datapoints.

    """
    epit = df["peptide"].unique()
    epitope_A = {
        epitope: df[df["peptide"] == epitope]["CDR3a"].unique() for epitope in epit
    }
    epitope_B = {
        epitope: df[df["peptide"] == epitope]["CDR3b"].unique() for epitope in epit
    }
    df2 = pd.DataFrame(columns=["CDR3b", "CDR3a", "peptide", "binder"])
    for i in range(len(df)):
        neg = 0
        row = df.loc[i]
        while neg < neg_per_pos:
            s = df.sample(n=1).iloc[0]

            if s["peptide"] != row["peptide"]:
                if (s["CDR3a"] == None) & (s["CDR3b"] == None):
                    continue
                elif s["CDR3a"] == None:
                    if s["CDR3b"] in epitope_B[row["peptide"]]:
                        continue

                elif s["CDR3b"] == None:
                    if s["CDR3a"] in epitope_A[row["peptide"]]:
                        continue

                else:
                    if s["CDR3a"] in epitope_A[row["peptide"]]:
                        continue
                    elif s["CDR3b"] in epitope_B[row["peptide"]]:
                        continue
                    else:
                        # new_row = {"peptide":row["peptide"],"CDR3a":s["CDR3a"],"CDR3b":s["CDR3b"], "binder":0, "MHC":row["MHC"]},
                        # Create a new DataFrame with the data you want to append
                        new_data = pd.DataFrame(
                            [
                                {
                                    "peptide": row["peptide"],
                                    "CDR3a": s["CDR3a"],
                                    "CDR3b": s["CDR3b"],
                                    "binder": 0,
                                    "MHC": row["MHC"],
                                }
                            ]
                        )

                        # Use pandas.concat to concatenate the new_data DataFrame to df2
                        df2 = pd.concat([df2, new_data], ignore_index=True)

                        # df2 = pd.concat({"peptide":row["peptide"],"CDR3a":s["CDR3a"],"CDR3b":s["CDR3b"], "binder":0, "MHC":row["MHC"]}, ignore_index=True)
                        neg = neg + 1
    return df2


class TCRDataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader. for TCR data."""

    def __init__(
        self,
        csv_file,
        tokenizer,
        device,
        target_binder=None,
        target_peptide=None,
        excluded_peptide=None,
        mhctok=None,
    ):  # , alpha_maxlength, beta_maxlength, epitope_maxlength):
        self.device = device
        self.tokenizer = tokenizer
        print("Loading the data ...")
        df = pd.read_csv(csv_file)

        if target_binder:
            df = df[df["binder"] == 1]

        if target_peptide:
            df = df[df["peptide"].apply(lambda x: x in target_peptide)]

        if excluded_peptide:
            print("exluded", excluded_peptide)
            iii = df["peptide"].apply(lambda x: x in excluded_peptide)
            df = df[~iii]

        self.alpha = list(df["CDR3a"])
        self.beta = list(df["CDR3b"])
        self.peptide = list(df["peptide"])
        self.binder = list(df["binder"])

        if mhctok:
            self.mhctok = mhctok
            self.MHC = list(df["MHC"])
        self.df = df
        self.reweight = False
        self.chain_masking_proba = 0.0

    @classmethod
    def empty_init(cls, tokenizer, device, mhctok=None):
        """Create an empty instance of the class, with no data."""
        obj = cls.__new__(cls)  # Does not call __init__
        super(
            TCRDataset, obj
        ).__init__()  # Don't forget to call any polymorphic base class initializers
        obj.device = device
        obj.tokenizer = tokenizer
        obj.mhctok = mhctok
        obj.MHC = []
        obj.alpha = []
        obj.beta = []
        obj.peptide = []
        obj.binder = []
        obj.reweight = False
        obj.chain_masking_proba = 0.0
        return obj

    def generate_unconditional_data(
        self, mask_alpha=True, mask_beta=True, mask_peptide=True, mask_mhc=False
    ):
        """Generate a new dataset with the same data, but with some of the data masked."""
        new = self.__class__.empty_init(self.tokenizer, self.device, self.mhctok)
        for i in range(len(self)):
            if mask_alpha:
                alpha = "<MIS>"
            else:
                alpha = self.alpha[i]
            if mask_beta:
                beta = "<MIS>"
            else:
                beta = self.beta[i]
            if mask_peptide:
                peptide = "<MIS>"
            else:
                peptide = self.peptide[i]
            if mask_mhc:
                mhc = "<MIS>"
            else:
                mhc = self.MHC[i]
            new.append(
                MHC=mhc, alpha=alpha, beta=beta, peptide=peptide, binder=self.binder[i]
            )

            # new.append(MHC=self.MHC[i], alpha=self.alpha[i], beta=self.beta[i], peptide=self.peptide[i], binder=self.binder[i])
        return new

    def append(
        self, MHC="<MIS>", alpha="<MIS>", beta="<MIS>", peptide="<MIS>", binder=0
    ):
        self.MHC.append(MHC)
        self.alpha.append(alpha)
        self.beta.append(beta)
        self.peptide.append(peptide)
        self.binder.append(binder)

    def concatenate(self, tcrdata, inplace=True):
        if inplace:
            self.MHC += tcrdata.MHC
            self.alpha += tcrdata.alpha
            self.beta += tcrdata.beta
            self.peptide += tcrdata.peptide
            self.binder += tcrdata.binder
        else:
            new = copy.deepcopy(self)
            new.MHC += tcrdata.MHC
            new.alpha += tcrdata.alpha
            new.beta += tcrdata.beta
            new.peptide += tcrdata.peptide
            new.binder += tcrdata.binder
            return new

    def to_pandas(self):
        return pd.DataFrame(
            {
                "MHC": self.MHC,
                "CDR3a": self.alpha,
                "CDR3b": self.beta,
                "peptide": self.peptide,
                "binder": self.binder,
            }
        )

    def select_binder(self, target_binder=1):
        new = self.__class__.empty_init(self.tokenizer, self.device, self.mhctok)
        for i in range(len(self)):
            if self.binder[i] == target_binder:
                new.append(
                    MHC=self.MHC[i],
                    alpha=self.alpha[i],
                    beta=self.beta[i],
                    peptide=self.peptide[i],
                    binder=self.binder[i],
                )
        return new

    def select_peptide(self, target_peptide):
        new = self.__class__.empty_init(self.tokenizer, self.device, self.mhctok)
        for i in range(len(self)):
            if self.peptide[i] in target_peptide:
                new.append(
                    MHC=self.MHC[i],
                    alpha=self.alpha[i],
                    beta=self.beta[i],
                    peptide=self.peptide[i],
                    binder=self.binder[i],
                )
        return new

    def select_chain(self, target_chain: str = "both"):
        """
        target_chain: 'both', 'alpha', 'beta'

        """
        new = self.__class__.empty_init(self.tokenizer, self.device, self.mhctok)
        if target_chain == "both":
            for i in range(len(self)):
                if self.alpha[i] == "<MIS>":
                    continue
                if self.beta[i] == "<MIS>":
                    continue
                else:
                    new.append(
                        MHC=self.MHC[i],
                        alpha=self.alpha[i],
                        beta=self.beta[i],
                        peptide=self.peptide[i],
                        binder=self.binder[i],
                    )
            return new
        if target_chain == "alpha":
            for i in range(len(self)):
                if self.alpha[i] == "<MIS>":
                    continue
                else:
                    new.append(
                        MHC=self.MHC[i],
                        alpha=self.alpha[i],
                        beta=self.beta[i],
                        peptide=self.peptide[i],
                        binder=self.binder[i],
                    )
            return new
        if target_chain == "beta":
            for i in range(len(self)):
                if self.beta[i] == "<MIS>":
                    continue
                else:
                    new.append(
                        MHC=self.MHC[i],
                        alpha=self.alpha[i],
                        beta=self.beta[i],
                        peptide=self.peptide[i],
                        binder=self.binder[i],
                    )
            return new

    def filter_peptide(self, target_peptide):
        new = self.__class__.empty_init(self.tokenizer, self.device, self.mhctok)
        for i in range(len(self)):
            if self.peptide[i] not in target_peptide:
                new.append(
                    MHC=self.MHC[i],
                    alpha=self.alpha[i],
                    beta=self.beta[i],
                    peptide=self.peptide[i],
                    binder=self.binder[i],
                )
        return new

    @classmethod
    def from_pandas(cls, df, tokenizer, device, mhctok=None):
        obj = cls.__new__(cls)  # Does not call __init__
        super(TCRDataset, obj).__init__()
        obj.device = device
        obj.tokenizer = tokenizer
        obj.mhctok = mhctok

        obj.alpha = list(df["CDR3a"])
        obj.beta = list(df["CDR3b"])
        obj.peptide = list(df["peptide"])
        obj.binder = list(df["binder"])

        if mhctok:
            obj.mhctok = mhctok
            obj.MHC = list(df["MHC"])
        obj.df = df
        obj.reweight = False
        obj.chain_masking_proba = 0.0

        return obj

    def set_chain_masking_proba(self, proba=0.0):
        self.chain_masking_proba = proba

    def generate_negatives(self, neg_per_pos=5):
        """
        Generate negative data by randomly sampling from the dataset.
        For every positive datapoint, we resample the TCRs among tcrs not binding the same epitope until we have neg_per_pos negative datapoints.
        Inplace.
        """
        df = self.to_pandas()
        df = df[df["binder"] == 1]
        df2 = generate_negative(df, neg_per_pos=neg_per_pos)
        for i in range(len(df2)):
            row = df2.iloc[i]
            self.append(
                MHC=row["MHC"],
                alpha=row["CDR3a"],
                beta=row["CDR3b"],
                peptide=row["peptide"],
                binder=row["binder"],
            )

    def __getitem__(self, offset):
        """Return one datapoint from the dataset, at position offset in the table.
        - if reweight is True, will provide a weight for each datapoint.
        - if mhctok is provided will provide an mhc token for each datapoint.
        """
        alpha = self.alpha[offset]
        beta = self.beta[offset]
        peptide = self.peptide[offset]
        binder = self.binder[offset]
        if self.chain_masking_proba > 0.0:
            if alpha != "<MIS>" and beta != "<MIS>":
                rd = np.random.uniform()
                if rd < self.chain_masking_proba / 2:
                    alpha = "<MIS>"
                elif rd < self.chain_masking_proba:
                    beta = "<MIS>"
            if alpha != "<MIS>" or beta != "<MIS>":
                rd = np.random.uniform()
                if rd < self.chain_masking_proba / 2:
                    peptide = "<MIS>"
        if self.mhctok:
            mhc = self.MHC[offset]
            # if self.reweight:
            #     w = self.weights[offset]
            #     return alpha, beta, peptide, binder, mhc, w
            return alpha, beta, peptide, binder, mhc
        return alpha, beta, peptide, binder

    def __len__(self):
        return len(self.peptide)

    def set_reweight(self, alpha):
        """Set the weights for each datapoint, based on the frequency of the peptide in the dataset."""
        freq = (
            self.df["peptide"].value_counts() / self.df["peptide"].value_counts().sum()
        )
        alpha = alpha
        freq = alpha * freq + (1 - alpha) / len(self.df["peptide"].value_counts())
        self.weights = (
            1 / torch.tensor(list(self.df.apply(lambda x: freq[x["peptide"]], 1)))
        ) / len(self.df["peptide"].value_counts())
        self.reweight = True

    def all2allmhc_collate_function(self, batch):
        """Collate function for the Tulip model returning peptide, alpha, beta, binder, mhc and weight if reweight is True"""

        if self.reweight:
            (alpha, beta, peptide, binder, mhc, weight) = zip(*batch)
        else:
            (alpha, beta, peptide, binder, mhc) = zip(*batch)

        peptide = self.tokenizer(
            list(peptide), padding="longest", add_special_tokens=True
        )
        peptide = {
            k: torch.tensor(v).to(self.device) for k, v in peptide.items()
        }  # default_collate(peptide)

        beta = self.tokenizer(list(beta), padding="longest", add_special_tokens=True)
        beta = {k: torch.tensor(v).to(self.device) for k, v in beta.items()}

        alpha = self.tokenizer(list(alpha), padding="longest", add_special_tokens=True)
        alpha = {k: torch.tensor(v).to(self.device) for k, v in alpha.items()}

        binder = default_collate(binder).to(self.device)
        mhc = self.mhctok(
            list(mhc)
        )  # default_collate(self.mhctok(list(mhc))['input_ids'])
        mhc = {k: torch.tensor(v).to(self.device) for k, v in mhc.items()}
        # print(mhc)
        if self.reweight:
            weight = torch.tensor(weight).to(self.device)
            return peptide, alpha, beta, binder, mhc, weight

        return peptide, alpha, beta, binder, mhc


### MODELING


@dataclass
class ClassifCausalLMOutputWithCrossAttentions(ModelOutput):
    lm_loss: Optional[torch.FloatTensor] = None
    lossCLS: Optional[torch.FloatTensor] = None
    pooled_output: Optional[torch.FloatTensor] = None
    clf_logits: torch.FloatTensor = None
    lm_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertLastPooler(nn.Module):
    """Policy for pooling the last (EOS) hidden states of a model into a single vector."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor, targetind) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the LAST token.
        ele = torch.arange(0, hidden_states.shape[0])

        first_token_tensor = hidden_states[
            ele.long(), targetind.long()
        ]  # .gather(1, targetind.view(-1,1))#hidden_states[:, -1]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class TulipPetal(BertPreTrainedModel):
    """TULIP decoder models."""

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # classifier is unused but can easibily be used again to make TULIP a supervised model
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.LMcls = BertOnlyMLMHead(config)
        self.alpha = 0.0
        self.pad_token_id = config.pad_token_id
        print("self.pad_token_id", self.pad_token_id)
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.LMcls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.LMcls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = True  # return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        # get clfPosition:
        temp = input_ids != self.pad_token_id
        # print('temp', temp)
        targetind = torch.sum(temp, dim=1) - 1

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.LMcls(sequence_output)
        pooled_output = (
            self.pooler(sequence_output, targetind) if self.pooler is not None else None
        )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        labelsCLS = labels[0]
        labelsLM = labels[1]
        lossCLS = None
        if labelsCLS is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labelsCLS.dtype == torch.long or labelsCLS.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    lossCLS = loss_fct(logits.squeeze(), labelsCLS.squeeze())
                else:
                    lossCLS = loss_fct(logits, labelsCLS)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                lossCLS = loss_fct(logits.view(-1, self.num_labels), labelsCLS.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                lossCLS = loss_fct(logits, labelsCLS)

        lm_loss = None
        if labelsLM is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labelsLM = labelsLM[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=self.pad_token_id)
            # print(self.pad_token_id)
            lm_loss = loss_fct(
                shifted_prediction_scores.view(-1, self.config.vocab_size),
                labelsLM.view(-1),
            )

        return ClassifCausalLMOutputWithCrossAttentions(
            lm_loss=lm_loss,
            lossCLS=lossCLS,
            pooled_output=pooled_output,
            clf_logits=logits,
            lm_logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, **model_kwargs
    ):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past,
        }


import typing


@dataclass
class ED_LMOutput(ModelOutput):
    clf_loss: Optional[torch.FloatTensor] = None
    clf_logits: Optional[torch.FloatTensor] = None
    decoder_outputsA: typing.Any = None
    encoder_outputsA: typing.Any = None
    decoder_outputsB: typing.Any = None
    encoder_outputsB: typing.Any = None
    decoder_outputsE: typing.Any = None
    encoder_outputsE: typing.Any = None


logger = logging.get_logger(__name__)


def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError(
            "Make sure to set the decoder_start_token_id attribute of the model's configuration."
        )
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError(
            "Make sure to set the pad_token_id attribute of the model's configuration."
        )
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class Tulip(PreTrainedModel):
    config_class = EncoderDecoderConfig
    base_model_prefix = "encoder_decoder"

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoderA: Optional[PreTrainedModel] = None,
        decoderA: Optional[PreTrainedModel] = None,
        encoderB: Optional[PreTrainedModel] = None,
        decoderB: Optional[PreTrainedModel] = None,
        encoderE: Optional[PreTrainedModel] = None,
        decoderE: Optional[PreTrainedModel] = None,
    ):
        if config is None and (encoderA is None or decoderA is None):
            raise ValueError(
                "Either a configuration or an encoder and a decoder has to be provided."
            )
        if config is None:
            config = EncoderDecoderConfig.from_encoder_decoder_configs(
                encoderA.config, decoderA.config
            )
            # config = EncoderDecoderConfig.from_encoder_decoder_configs(encoderA.config, decoderA.config,encoderB.config, decoderB.config,encoderE.config, decoderE.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(
                    f"Config: {config} has to be of type {self.config_class}"
                )

        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
                    f" to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                    f" `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for"
                    " `config.encoder.hidden_size`."
                )

        # initialize with config
        super().__init__(config)

        if encoderA is None:
            from ..auto.modeling_auto import AutoModel

            encoderA = AutoModel.from_config(config.encoder)
        if encoderE is None:
            from ..auto.modeling_auto import AutoModel

            encoderE = AutoModel.from_config(config.encoder)
        if encoderB is None:
            from ..auto.modeling_auto import AutoModel

            encoderB = AutoModel.from_config(config.encoder)

        if decoderA is None:
            from ..auto.modeling_auto import AutoModelForCausalLM

            decoderA = AutoModelForCausalLM.from_config(config.decoder)
        if decoderB is None:
            from ..auto.modeling_auto import AutoModelForCausalLM

            decoderB = AutoModelForCausalLM.from_config(config.decoder)
        if decoderE is None:
            from ..auto.modeling_auto import AutoModelForCausalLM

            decoderE = AutoModelForCausalLM.from_config(config.decoder)
        self.reweight = False
        self.encoderA = encoderA
        self.decoderA = decoderA
        self.encoderB = encoderB
        self.decoderB = decoderB
        self.encoderE = encoderE
        self.decoderE = decoderE
        self.num_labels = 2
        self.MLMHeadA = BertOnlyMLMHead(decoderA.config)
        self.MLMHeadB = BertOnlyMLMHead(decoderB.config)
        self.MLMHeadE = BertOnlyMLMHead(decoderE.config)

        # Miss Mask Implemetation
        self.skipMiss = True
        self.MissA = nn.Parameter(
            torch.zeros((1, encoderA.config.hidden_size)), requires_grad=True
        )
        self.MissB = nn.Parameter(
            torch.zeros((1, encoderB.config.hidden_size)), requires_grad=True
        )
        self.MissE = nn.Parameter(
            torch.zeros((1, encoderE.config.hidden_size)), requires_grad=True
        )
        # This classifier is only here for potential future supervised task
        self.classifier = nn.Linear(3 * decoderA.config.hidden_size, 2)
        self.mhc_embeddings = nn.Embedding(
            encoderA.config.mhc_vocab_size, encoderA.config.hidden_size
        )
        if self.encoderA.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoderA.__class__} is overwritten by shared encoder config:"
                f" {self.config.encoder}"
            )
        if self.decoderA.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoderA.__class__} is overwritten by shared decoder config:"
                f" {self.config.decoder}"
            )

        if (
            self.encoderA.config.hidden_size != self.decoderA.config.hidden_size
            and self.decoderA.config.cross_attention_hidden_size is None
        ):
            self.enc_to_dec_proj = nn.Linear(
                self.encoderA.config.hidden_size, self.decoderA.config.hidden_size
            )

        if self.encoderA.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoderA} should not have a LM Head. Please use a model without LM Head"
            )

        self.tie_weights()

    def tie_weights(self):
        # tie encoder & decoder if needed
        if self.config.tie_encoder_decoder:
            # tie encoder and decoder base model
            decoder_base_model_prefix = self.decoder.base_model_prefix
            self._tie_encoder_decoder_weights(
                self.encoder,
                self.decoder._modules[decoder_base_model_prefix],
                self.decoder.base_model_prefix,
            )

    def get_encoder(self, encoder_name="B"):
        if encoder_name == "A":
            return self.encoderA
        elif encoder_name == "B":
            return self.encoderB
        elif encoder_name == "E":
            return self.encoderE

    def get_decoder(self, decoder_name="B"):
        if decoder_name == "A":
            return self.decoderA
        elif decoder_name == "B":
            return self.decoderB
        elif decoder_name == "E":
            return self.decoderE

    def get_input_embeddings(self, encoder_name="B"):
        if encoder_name == "A":
            return self.encoderA.get_input_embeddings()
        elif encoder_name == "B":
            return self.encoderB.get_input_embeddings()
        elif encoder_name == "E":
            return self.encoderE.get_input_embeddings()

    def get_output_embeddings(self, decoder_name="B"):
        if decoder_name == "A":
            return self.decoderA.get_output_embeddings()
        elif decoder_name == "B":
            return self.decoderB.get_output_embeddings()
        elif decoder_name == "E":
            return self.decoderE.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings, decoder_name="B"):
        if decoder_name == "A":
            return self.decoderA.set_output_embeddings(new_embeddings)
        elif decoder_name == "B":
            return self.decoderB.set_output_embeddings(new_embeddings)
        elif decoder_name == "E":
            return self.decoderE.set_output_embeddings(new_embeddings)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported for composite models
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for EncoderDecoderModel. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        kwargs_encoder = {
            argument[len("encoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("decoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config, kwargs_encoder = AutoConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path,
                    **kwargs_encoder,
                    return_unused_kwargs=True,
                )

                if (
                    encoder_config.is_decoder is True
                    or encoder_config.add_cross_attention is True
                ):
                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = AutoModel.from_pretrained(
                encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder
            )

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                decoder_config, kwargs_decoder = AutoConfig.from_pretrained(
                    decoder_pretrained_model_name_or_path,
                    **kwargs_decoder,
                    return_unused_kwargs=True,
                )

                if (
                    decoder_config.is_decoder is False
                    or decoder_config.add_cross_attention is False
                ):
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention"
                        f" layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if"
                        f" {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True

                kwargs_decoder["config"] = decoder_config

            if (
                kwargs_decoder["config"].is_decoder is False
                or kwargs_decoder["config"].add_cross_attention is False
            ):
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. "
                    f"In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, "
                    "make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` "
                    "passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a "
                    "`decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            decoder = AutoModelForCausalLM.from_pretrained(
                decoder_pretrained_model_name_or_path, **kwargs_decoder
            )

        # instantiate config with corresponding kwargs
        config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder.config, decoder.config, **kwargs
        )
        return cls(encoder=encoder, decoder=decoder, config=config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = (None, None, None),
        attention_mask: Optional[torch.FloatTensor] = (None, None, None),
        # decoder_input_ids: Optional[torch.LongTensor] =  (None, None,None),
        # decoder_attention_mask: Optional[torch.BoolTensor] =  (None, None,None),
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = (None, None, None),
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = (None, None, None),
        inputs_embeds: Optional[torch.FloatTensor] = (None, None, None),
        # decoder_inputs_embeds: Optional[torch.FloatTensor] =  (None, None,None),
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = (None, None, None),
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = (None, None, None),
        mhc=None,
        togenerate=None,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        # print('forward', input_ids)
        input_idsA = input_ids[0]
        input_idsB = input_ids[1]
        input_idsE = input_ids[2]

        # Miss mask Implementation
        # To Do replace hard coded 3 and 4 with bos and miss token ids
        if self.skipMiss:
            if input_idsA != None:
                if input_idsA.shape[1] == 1:
                    MissMaskA = input_idsA.clone().detach()[:, 0] != 3
                else:
                    MissMaskA = input_idsA.clone().detach()[:, 1] == 4
            if input_idsB != None:
                if input_idsB.shape[1] == 1:
                    MissMaskB = input_idsB.clone().detach()[:, 0] != 3
                else:
                    MissMaskB = input_idsB.clone().detach()[:, 1] == 4

            if input_idsE != None:
                if input_idsE.shape[1] == 1:
                    MissMaskE = input_idsE.clone().detach()[:, 0] != 3
                else:
                    MissMaskE = input_idsE.clone().detach()[:, 1] == 4

        attention_maskA = attention_mask[0]
        attention_maskB = attention_mask[1]
        attention_maskE = attention_mask[2]

        encoder_outputsA = encoder_outputs[0]
        encoder_outputsB = encoder_outputs[1]
        encoder_outputsE = encoder_outputs[2]

        past_key_valuesA = past_key_values[0]
        past_key_valuesB = past_key_values[1]
        past_key_valuesE = past_key_values[2]

        inputs_embedsA = inputs_embeds[0]
        inputs_embedsB = inputs_embeds[1]
        inputs_embedsE = inputs_embeds[2]

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        kwargs_encoder = {
            argument: value
            for argument, value in kwargs.items()
            if not argument.startswith("decoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("decoder_")
        }

        if encoder_outputsA is None:
            encoder_outputsA = self.encoderA(
                input_ids=input_idsA,
                attention_mask=attention_maskA,
                inputs_embeds=inputs_embedsA,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputsA, tuple):
            encoder_outputsA = BaseModelOutput(*encoder_outputsA)

        if encoder_outputsB is None:
            encoder_outputsB = self.encoderB(
                input_ids=input_idsB,
                attention_mask=attention_maskB,
                inputs_embeds=inputs_embedsB,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputsB, tuple):
            encoder_outputsB = BaseModelOutput(*encoder_outputsB)

        if encoder_outputsE is None:
            encoder_outputsE = self.encoderE(
                input_ids=input_idsE,
                attention_mask=attention_maskE,
                inputs_embeds=inputs_embedsE,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputsE, tuple):
            encoder_outputsE = BaseModelOutput(*encoder_outputsE)

        encoder_hidden_statesA = encoder_outputsA[0]
        encoder_hidden_statesB = encoder_outputsB[0]
        encoder_hidden_statesE = encoder_outputsE[0]
        # optionally project encoder_hidden_states

        # Miss mask Implementation
        if self.skipMiss:
            if input_idsA != None:
                encoder_hidden_statesA = encoder_hidden_statesA.clone()
                encoder_hidden_statesA[MissMaskA, 0, :] = self.MissA
                encoder_hidden_statesA[MissMaskA, 1, :] = self.MissA
                encoder_hidden_statesA[MissMaskA, 2, :] = self.MissA
            if input_idsB != None:
                encoder_hidden_statesB = encoder_hidden_statesB.clone()
                encoder_hidden_statesB[MissMaskB, 0, :] = self.MissB
                encoder_hidden_statesB[MissMaskB, 1, :] = self.MissB
                encoder_hidden_statesB[MissMaskB, 2, :] = self.MissB
            if input_idsE != None:
                encoder_hidden_statesE = encoder_hidden_statesE.clone()
                encoder_hidden_statesE[MissMaskE, 0, :] = self.MissE
                encoder_hidden_statesE[MissMaskE, 1, :] = self.MissE
                encoder_hidden_statesE[MissMaskE, 2, :] = self.MissE

        if (
            self.encoderA.config.hidden_size != self.decoderA.config.hidden_size
            and self.decoderA.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_statesA = self.enc_to_dec_proj(encoder_hidden_statesA)

        if (
            self.encoderB.config.hidden_size != self.decoderB.config.hidden_size
            and self.decoderB.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_statesB = self.enc_to_dec_proj(encoder_hidden_statesB)

        if (
            self.encoderE.config.hidden_size != self.decoderE.config.hidden_size
            and self.decoderE.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_statesE = self.enc_to_dec_proj(encoder_hidden_statesE)

        mhc_encoded = self.mhc_embeddings(mhc["input_ids"])
        mhc_attention_mask = mhc["attention_mask"]
        # Decode
        if togenerate not in ["B", "E"]:
            labelsA = (labels, input_idsA)
            decoder_outputsA = self.decoderA(
                input_ids=input_idsA,
                attention_mask=attention_maskA,
                encoder_hidden_states=torch.cat(
                    [mhc_encoded, encoder_hidden_statesB, encoder_hidden_statesE], dim=1
                ),
                encoder_attention_mask=torch.cat(
                    [mhc_attention_mask, attention_maskB, attention_maskE], dim=1
                ),
                inputs_embeds=inputs_embedsA,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                labels=labelsA,
                use_cache=use_cache,
                past_key_values=past_key_valuesA,
                return_dict=return_dict,
                **kwargs_decoder,
            )
            pooled_outputA = decoder_outputsA.pooled_output

        if togenerate not in ["A", "E"]:
            labelsB = (labels, input_idsB)
            decoder_outputsB = self.decoderB(
                input_ids=input_idsB,
                attention_mask=attention_maskB,
                encoder_hidden_states=torch.cat(
                    [mhc_encoded, encoder_hidden_statesA, encoder_hidden_statesE], dim=1
                ),
                encoder_attention_mask=torch.cat(
                    [mhc_attention_mask, attention_maskA, attention_maskE], dim=1
                ),
                inputs_embeds=inputs_embedsB,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                labels=labelsB,
                use_cache=use_cache,
                past_key_values=past_key_valuesB,
                return_dict=return_dict,
                **kwargs_decoder,
            )
            pooled_outputB = decoder_outputsB.pooled_output

        if togenerate not in ["A", "B"]:
            labelsE = (labels, input_idsE)
            decoder_outputsE = self.decoderE(
                input_ids=input_idsE,
                attention_mask=attention_maskE,
                encoder_hidden_states=torch.cat(
                    [mhc_encoded, encoder_hidden_statesA, encoder_hidden_statesB], dim=1
                ),
                encoder_attention_mask=torch.cat(
                    [mhc_attention_mask, attention_maskA, attention_maskB], dim=1
                ),
                inputs_embeds=inputs_embedsE,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                labels=labelsE,
                use_cache=use_cache,
                past_key_values=past_key_valuesE,
                return_dict=return_dict,
                **kwargs_decoder,
            )
            pooled_outputE = decoder_outputsE.pooled_output

        lossCLS = None
        logits = None

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None

        if togenerate == "A":
            return Seq2SeqLMOutput(
                loss=loss,
                logits=decoder_outputsA.lm_logits,
                past_key_values=decoder_outputsA.past_key_values,
                decoder_hidden_states=decoder_outputsA.hidden_states,
                decoder_attentions=decoder_outputsA.attentions,
                cross_attentions=decoder_outputsA.cross_attentions,
                encoder_last_hidden_state=torch.cat(
                    [mhc_encoded, encoder_hidden_statesB, encoder_hidden_statesE], dim=1
                ),
                encoder_hidden_states=encoder_outputsE.hidden_states,
                encoder_attentions=torch.cat(
                    [mhc_attention_mask, attention_maskB, attention_maskE], dim=1
                ),
            )
        elif togenerate == "B":
            return Seq2SeqLMOutput(
                loss=loss,
                logits=decoder_outputsB.lm_logits,
                past_key_values=decoder_outputsB.past_key_values,
                decoder_hidden_states=decoder_outputsB.hidden_states,
                decoder_attentions=decoder_outputsB.attentions,
                cross_attentions=decoder_outputsB.cross_attentions,
                encoder_last_hidden_state=torch.cat(
                    [mhc_encoded, encoder_hidden_statesA, encoder_hidden_statesE], dim=1
                ),
                encoder_hidden_states=encoder_outputsE.hidden_states,
                encoder_attentions=torch.cat(
                    [mhc_attention_mask, attention_maskA, attention_maskE], dim=1
                ),
            )
        elif togenerate == "E":
            return Seq2SeqLMOutput(
                loss=loss,
                logits=decoder_outputsE.lm_logits,
                past_key_values=decoder_outputsE.past_key_values,
                decoder_hidden_states=decoder_outputsE.hidden_states,
                decoder_attentions=decoder_outputsE.attentions,
                cross_attentions=decoder_outputsE.cross_attentions,
                encoder_last_hidden_state=torch.cat(
                    [mhc_encoded, encoder_hidden_statesA, encoder_hidden_statesB], dim=1
                ),
                encoder_hidden_states=encoder_outputsE.hidden_states,
                encoder_attentions=torch.cat(
                    [mhc_attention_mask, attention_maskA, attention_maskB], dim=1
                ),
            )

        else:
            return ED_LMOutput(
                clf_loss=lossCLS,
                clf_logits=logits,
                encoder_outputsA=encoder_outputsA,
                decoder_outputsA=decoder_outputsA,
                encoder_outputsB=encoder_outputsB,
                decoder_outputsB=decoder_outputsB,
                encoder_outputsE=encoder_outputsE,
                decoder_outputsE=decoder_outputsE,
            )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        # print('prepare_decoder_input_ids_from_labels')
        return shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=(None, None, None),
        use_cache=None,
        encoder_outputs=(None, None, None),
        **kwargs,
    ):

        # print('prepare_inputs_for_generation')
        togenerate = kwargs["togenerate"]
        if togenerate == "A":
            decoder_inputs = self.decoderA.prepare_inputs_for_generation(
                input_ids, past=past
            )
            decoder_attention_mask = (
                decoder_inputs["attention_mask"]
                if "attention_mask" in decoder_inputs
                else None
            )
            input_dict = {
                "input_ids": (decoder_inputs["input_ids"], None, None),
                "attention_mask": (
                    decoder_attention_mask,
                    attention_mask[1],
                    attention_mask[2],
                ),
                "encoder_outputs": encoder_outputs,
                "past_key_values": (decoder_inputs["past_key_values"], None, None),
                "use_cache": use_cache,
                "togenerate": togenerate,
                "mhc": kwargs["mhc"],
            }
            return input_dict
        elif togenerate == "B":
            decoder_inputs = self.decoderB.prepare_inputs_for_generation(
                input_ids, past=past
            )
            decoder_attention_mask = (
                decoder_inputs["attention_mask"]
                if "attention_mask" in decoder_inputs
                else None
            )
            input_dict = {
                "input_ids": (None, decoder_inputs["input_ids"], None),
                "attention_mask": (
                    attention_mask[0],
                    decoder_attention_mask,
                    attention_mask[2],
                ),
                "encoder_outputs": encoder_outputs,
                "past_key_values": (None, decoder_inputs["past_key_values"], None),
                "use_cache": use_cache,
                "togenerate": togenerate,
                "mhc": kwargs["mhc"],
            }
            return input_dict
        elif togenerate == "E":
            decoder_inputs = self.decoderE.prepare_inputs_for_generation(
                input_ids, past=past
            )
            decoder_attention_mask = (
                decoder_inputs["attention_mask"]
                if "attention_mask" in decoder_inputs
                else None
            )
            input_dict = {
                "input_ids": (None, None, decoder_inputs["input_ids"]),
                "attention_mask": (
                    attention_mask[0],
                    attention_mask[1],
                    decoder_attention_mask,
                ),
                "encoder_outputs": encoder_outputs,
                "past_key_values": (None, None, decoder_inputs["past_key_values"]),
                "use_cache": use_cache,
                "togenerate": togenerate,
                "mhc": kwargs["mhc"],
            }
            return input_dict
        else:
            raise ValueError("togenerate should be A, B or E")

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the EncoderDecoderModel directly is not supported. Please use the"
            " respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or"
            " model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past, beam_idx)

    def set_reweight(self):
        self.reweight = True

    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        """
        This function extracts the model-specific `inputs` for generation.
        """

        input_name = "input_ids"
        model_kwargs = {
            k: v for k, v in model_kwargs.items() if v is not None or k != input_name
        }  # WHYYYYY

        # 5. if `inputs` is still None, try to create `input_ids` from BOS token
        if inputs is None:
            bs = model_kwargs["input_ids"][0].shape[0]
            inputs = torch.ones((bs, 1), dtype=torch.long, device=device) * bos_token_id
            # self._prepare_input_ids_for_generation(bos_token_id, model_kwargs.get("encoder_outputs"))

        # print('_prepare_model_inputs', inputs, input_name, model_kwargs)
        return inputs, input_name, model_kwargs

    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs,
        model_input_name: Optional[str] = None,
    ) -> Dict[str, Any]:

        encoder_kwargs = model_kwargs.copy()
        encoder_kwargs["togenerate"] = None

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = (
            model_input_name if model_input_name is not None else self.main_input_name
        )
        encoder_kwargs["return_dict"] = True
        # encoder_kwargs[model_input_name] = inputs_tensor
        # model_kwargs["encoder_outputs"]: ModelOutput
        out = self.forward(**encoder_kwargs)
        model_kwargs["encoder_outputs"] = (
            out.encoder_outputsA,
            out.encoder_outputsB,
            out.encoder_outputsE,
        )
        model_kwargs["decoder_input_ids"] = inputs_tensor  #### Not NEEDED?
        model_kwargs.pop("input_ids", None)  #### WHY?

        return model_kwargs

    @staticmethod
    def _update_model_kwargs_for_generation(
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
    ) -> Dict[str, Any]:
        # print('_update_model_kwargs_for_generation')
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
            )

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones((attention_mask.shape[0], 1)),
                    ],
                    dim=-1,
                )

        return model_kwargs

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, torch.Tensor],
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
        device: torch.device = None,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
        # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
        # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        # 2. Encoder-decoder models expect the `decoder_input_ids` to start with a special token. Let's ensure that.
        decoder_start_token_id = self._get_decoder_start_token_id(
            decoder_start_token_id, bos_token_id
        )
        if device is None:
            device = self.device
        decoder_input_ids_start = (
            torch.ones((batch_size, 1), dtype=torch.long, device=device)
            * decoder_start_token_id
        )

        # no user input -> use decoder_start_token_id as decoder_input_ids
        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start
        # exception: Donut checkpoints have task-specific decoder starts and don't expect a BOS token
        elif (
            self.config.model_type == "vision-encoder-decoder"
            and "donut" in self.name_or_path.lower()
        ):
            pass
        # user input but doesn't start with decoder_start_token_id -> prepend decoder_start_token_id (and adjust
        # decoder_attention_mask if provided)
        elif (decoder_input_ids[:, 0] != decoder_start_token_id).all().item():
            decoder_input_ids = torch.cat(
                [decoder_input_ids_start, decoder_input_ids], dim=-1
            )
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = torch.cat(
                    (
                        torch.ones_like(decoder_attention_mask)[:, :1],
                        decoder_attention_mask,
                    ),
                    dim=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        return decoder_input_ids, model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: Optional[torch.LongTensor] = (None, None, None),
        encoder_outputs: Optional[Tuple[ModelOutput]] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        # print('_expand_inputs_for_generation')
        expanded_return_idx = (
            torch.arange(input_ids.shape[0])
            .view(-1, 1)
            .repeat(1, expand_size)
            .view(-1)
            .to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)
        model_kwargs["mhc"]["input_ids"] = model_kwargs["mhc"][
            "input_ids"
        ].index_select(0, expanded_return_idx)
        model_kwargs["mhc"]["attention_mask"] = model_kwargs["mhc"][
            "attention_mask"
        ].index_select(0, expanded_return_idx)
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(
                0, expanded_return_idx
            )

        if attention_mask is not None:
            model_kwargs["attention_mask"] = (
                attention_mask[0].index_select(0, expanded_return_idx),
                attention_mask[1].index_select(0, expanded_return_idx),
                attention_mask[2].index_select(0, expanded_return_idx),
            )

        if is_encoder_decoder:
            if encoder_outputs == (None, None, None):
                raise ValueError(
                    "If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined."
                )
            encoder_outputs[0]["last_hidden_state"] = encoder_outputs[
                0
            ].last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs[0].last_hidden_state.device)
            )
            encoder_outputs[1]["last_hidden_state"] = encoder_outputs[
                1
            ].last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs[1].last_hidden_state.device)
            )
            encoder_outputs[2]["last_hidden_state"] = encoder_outputs[
                2
            ].last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs[2].last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs


### TRAIN AND EVAL FUNCTIONS


def compute_loss(predictions, targets, criterion):
    """Compute our custom loss"""
    if len(targets) > 0:
        predictions = predictions[:, :-1, :].contiguous()
        targets = targets[:, 1:]

        rearranged_output = predictions.view(
            predictions.shape[0] * predictions.shape[1], -1
        )
        rearranged_target = targets.contiguous().view(-1)

        loss = criterion(rearranged_output, rearranged_target)
    else:
        loss = 0

    return loss


def MLM_Loss(encoder, head, masker, inputsid, inputsam, observedmask):
    if sum(observedmask) != 0:
        input_ids, labels = masker.torch_mask_tokens(inputsid[observedmask])
        attention_mask = inputsam
        outputs = encoder(
            input_ids,
            attention_mask=attention_mask[observedmask],
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state
        prediction_scores = head(sequence_output)

        masked_lm_loss = None
        loss_fct = CrossEntropyLoss(reduction="sum")  # -100 index = padding token
        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, encoder.config.vocab_size), labels.view(-1)
        )
        return masked_lm_loss
    else:
        loss = 0


def train_unsupervised(model, optimizer, masker, train_dataloader, criterion, alph=1.0):
    model.train()

    epoch_lm_lossA = 0
    epoch_lm_lossB = 0
    epoch_lm_lossE = 0
    epoch_mlm_lossA = 0
    epoch_mlm_lossB = 0
    epoch_mlm_lossE = 0
    count_A = 0
    count_B = 0
    count_E = 0
    for i, (peptide, alpha, beta, binder, mhc) in enumerate(train_dataloader):
        optimizer.zero_grad()
        peptide_input = peptide["input_ids"]
        peptide_mask = peptide["attention_mask"]
        peptide_tokentype = peptide["token_type_ids"]
        alpha_input = alpha["input_ids"]
        alpha_mask = alpha["attention_mask"]
        alpha_tokentype = alpha["token_type_ids"]
        beta_input = beta["input_ids"]
        beta_mask = beta["attention_mask"]
        beta_tokentype = beta["token_type_ids"]

        alpha_observed_mask = alpha_input.clone().detach()[:, 1] != 4
        beta_observed_mask = beta_input.clone().detach()[:, 1] != 4
        peptide_observed_mask = peptide_input.clone().detach()[:, 1] != 4
        clf_label = binder.clone()
        labels = clf_label

        out = model(
            input_ids=(alpha_input, beta_input, peptide_input),
            attention_mask=(alpha_mask, beta_mask, peptide_mask),
            labels=labels,
            mhc=mhc,
        )

        prediction_scoresA = out.decoder_outputsA.lm_logits
        predictionsA = F.log_softmax(prediction_scoresA, dim=2)
        prediction_scoresB = out.decoder_outputsB.lm_logits
        predictionsB = F.log_softmax(prediction_scoresB, dim=2)
        prediction_scoresE = out.decoder_outputsE.lm_logits
        predictionsE = F.log_softmax(prediction_scoresE, dim=2)
        lossa = compute_loss(
            predictionsA[alpha_observed_mask],
            alpha_input[alpha_observed_mask],
            criterion,
        )
        lossb = compute_loss(
            predictionsB[beta_observed_mask], beta_input[beta_observed_mask], criterion
        )
        losse = compute_loss(
            predictionsE[peptide_observed_mask],
            peptide_input[peptide_observed_mask],
            criterion,
        )

        mlm_lossA = MLM_Loss(
            model.encoderA,
            model.MLMHeadA,
            masker,
            alpha_input,
            alpha_mask,
            alpha_observed_mask,
        )
        mlm_lossB = MLM_Loss(
            model.encoderB,
            model.MLMHeadB,
            masker,
            beta_input,
            beta_mask,
            beta_observed_mask,
        )
        mlm_lossE = MLM_Loss(
            model.encoderE,
            model.MLMHeadE,
            masker,
            peptide_input,
            peptide_mask,
            peptide_observed_mask,
        )

        loss = mlm_lossA + mlm_lossB + mlm_lossE + alph * (lossa + lossb + losse)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        count_A += sum(alpha_observed_mask)
        count_B += sum(beta_observed_mask)
        count_E += sum(peptide_observed_mask)
        epoch_lm_lossA += lossa
        epoch_lm_lossB += lossb
        epoch_lm_lossE += losse
        epoch_mlm_lossA += mlm_lossA
        epoch_mlm_lossB += mlm_lossB
        epoch_mlm_lossE += mlm_lossE

    epoch_lm_lossA /= count_A
    epoch_lm_lossB /= count_B
    epoch_lm_lossE /= count_E
    epoch_mlm_lossA /= count_A
    epoch_mlm_lossB /= count_B
    epoch_mlm_lossE /= count_E

    return (
        epoch_lm_lossA,
        epoch_lm_lossB,
        epoch_lm_lossE,
        epoch_mlm_lossA,
        epoch_mlm_lossB,
        epoch_mlm_lossE,
    )


def eval_unsupervised(model, masker, test_dataloader, criterion):
    model.eval()
    with torch.no_grad():
        epoch_lm_lossA = 0
        epoch_lm_lossB = 0
        epoch_lm_lossE = 0
        epoch_mlm_lossA = 0
        epoch_mlm_lossB = 0
        epoch_mlm_lossE = 0
        count_A = 0
        count_B = 0
        count_E = 0
        for i, (peptide, alpha, beta, binder, mhc) in enumerate(test_dataloader):
            peptide_input = peptide["input_ids"]
            peptide_mask = peptide["attention_mask"]
            peptide_tokentype = peptide["token_type_ids"]
            alpha_input = alpha["input_ids"]
            alpha_mask = alpha["attention_mask"]
            alpha_tokentype = alpha["token_type_ids"]
            beta_input = beta["input_ids"]
            beta_mask = beta["attention_mask"]
            beta_tokentype = beta["token_type_ids"]

            alpha_observed_mask = alpha_input.clone().detach()[:, 1] != 4
            beta_observed_mask = beta_input.clone().detach()[:, 1] != 4
            peptide_observed_mask = peptide_input.clone().detach()[:, 1] != 4
            # alpha_observed_mask = torch.tensor(alpha_input)[:,1] != 4
            # beta_observed_mask = torch.tensor(beta_input)[:,1] != 4
            # peptide_observed_mask = torch.tensor(peptide_input)[:,1] != 4
            # lm_labels = alphabeta_output.clone()
            clf_label = binder.clone()
            labels = clf_label

            out = model(
                input_ids=(alpha_input, beta_input, peptide_input),
                attention_mask=(alpha_mask, beta_mask, peptide_mask),
                labels=labels,
                mhc=mhc,
            )

            prediction_scoresA = out.decoder_outputsA.lm_logits
            predictionsA = F.log_softmax(prediction_scoresA, dim=2)
            prediction_scoresB = out.decoder_outputsB.lm_logits
            predictionsB = F.log_softmax(prediction_scoresB, dim=2)
            prediction_scoresE = out.decoder_outputsE.lm_logits
            predictionsE = F.log_softmax(prediction_scoresE, dim=2)
            lossa = compute_loss(
                predictionsA[alpha_observed_mask],
                alpha_input[alpha_observed_mask],
                criterion,
            )
            lossb = compute_loss(
                predictionsB[beta_observed_mask],
                beta_input[beta_observed_mask],
                criterion,
            )
            losse = compute_loss(
                predictionsE[peptide_observed_mask],
                peptide_input[peptide_observed_mask],
                criterion,
            )
            mlm_lossA = MLM_Loss(
                model.encoderA,
                model.MLMHeadA,
                masker,
                alpha_input,
                alpha_mask,
                alpha_observed_mask,
            )
            mlm_lossB = MLM_Loss(
                model.encoderB,
                model.MLMHeadB,
                masker,
                beta_input,
                beta_mask,
                beta_observed_mask,
            )

            mlm_lossE = MLM_Loss(
                model.encoderE,
                model.MLMHeadE,
                masker,
                peptide_input,
                peptide_mask,
                peptide_observed_mask,
            )

            loss = mlm_lossA + mlm_lossB + mlm_lossE + lossa + lossb + losse

            # print(model.decoder.bert.pooler.dense.weight)
            count_A += sum(alpha_observed_mask)
            count_B += sum(beta_observed_mask)
            count_E += sum(peptide_observed_mask)
            epoch_lm_lossA += lossa
            epoch_lm_lossB += lossb
            epoch_lm_lossE += losse
            epoch_mlm_lossA += mlm_lossA
            epoch_mlm_lossB += mlm_lossB
            epoch_mlm_lossE += mlm_lossE

        epoch_lm_lossA /= count_A
        epoch_lm_lossB /= count_B
        epoch_lm_lossE /= count_E
        epoch_mlm_lossA /= count_A
        epoch_mlm_lossB /= count_B
        epoch_mlm_lossE /= count_E

        return (
            epoch_lm_lossA,
            epoch_lm_lossB,
            epoch_lm_lossE,
            epoch_mlm_lossA,
            epoch_mlm_lossB,
            epoch_mlm_lossE,
        )


class MyMasking:
    def __init__(self, tokenizer, mlm_probability: float = 0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def torch_mask_tokens(
        self, inputs: Any, special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        ).to(inputs.device)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


def LLLoss_raw(predictions, targets, ignore_index):
    """Compute our custom loss"""
    criterion = nn.NLLLoss(ignore_index=ignore_index, reduction="none")
    if len(targets) > 0:
        predictions = predictions[:, :-1, :].contiguous()
        targets = targets[:, 1:]
        bs = targets.shape[0]

        rearranged_output = predictions.view(
            predictions.shape[0] * predictions.shape[1], -1
        )
        rearranged_target = targets.contiguous().view(-1)

        loss = (
            criterion(rearranged_output, rearranged_target).reshape(bs, -1).sum(dim=1)
        )
    else:
        loss = torch.zeros(1)
    return loss


def unsupervised_auc(model, test_dataloader, ignore_index):
    model.eval()
    with torch.no_grad():
        clf_scorea = []
        clf_scoreb = []
        clf_scoree = []
        Boolbinders = []
        for i, (peptide, alpha, beta, binder, mhc) in enumerate(test_dataloader):
            peptide_input = peptide["input_ids"]
            peptide_mask = peptide["attention_mask"]
            peptide_tokentype = peptide["token_type_ids"]
            alpha_input = alpha["input_ids"]
            alpha_mask = alpha["attention_mask"]
            alpha_tokentype = alpha["token_type_ids"]
            beta_input = beta["input_ids"]
            beta_mask = beta["attention_mask"]
            beta_tokentype = beta["token_type_ids"]

            binder = binder
            positive_mask = binder == 1
            alpha_observed_mask = alpha_input.clone().detach()[:, 1] != 4
            beta_observed_mask = beta_input.clone().detach()[:, 1] != 4
            peptide_observed_mask = peptide_input.clone().detach()[:, 1] != 4

            clf_label = binder.clone()
            labels = clf_label

            out = model(
                input_ids=(alpha_input, beta_input, peptide_input),
                attention_mask=(alpha_mask, beta_mask, peptide_mask),
                labels=labels,
                mhc=mhc,
            )

            def softm(x):
                x = torch.exp(x)
                su = torch.sum(x, dim=1)
                x = x / su
                return x

            Boolbinders += [(binder[i] == 1).cpu().item() for i in range(len(binder))]
            lm_lossA = out.decoder_outputsA.lm_loss
            lm_lossB = out.decoder_outputsB.lm_loss
            lm_lossE = out.decoder_outputsE.lm_loss

            prediction_scoresA = out.decoder_outputsA.lm_logits
            predictionsA = F.log_softmax(prediction_scoresA, dim=2)
            prediction_scoresB = out.decoder_outputsB.lm_logits
            predictionsB = F.log_softmax(prediction_scoresB, dim=2)
            prediction_scoresE = out.decoder_outputsE.lm_logits
            predictionsE = F.log_softmax(prediction_scoresE, dim=2)

            lossa = LLLoss_raw(predictionsA, alpha_input, ignore_index)
            lossb = LLLoss_raw(predictionsB, beta_input, ignore_index)
            losse = LLLoss_raw(predictionsE, peptide_input, ignore_index)
            clf_scorea += [-1 * lossa[i].cpu().item() for i in range(len(lossa))]
            clf_scoreb += [-1 * lossb[i].cpu().item() for i in range(len(lossb))]
            clf_scoree += [-1 * losse[i].cpu().item() for i in range(len(losse))]

        auca = roc_auc_score(Boolbinders, clf_scorea)
        aucb = roc_auc_score(Boolbinders, clf_scoreb)
        auce = roc_auc_score(Boolbinders, clf_scoree)
        return auca, aucb, auce


def get_mi(model, dataset, mask_mhc=True, mask_peptide=True, mask_paired=False):
    """_summary_
    Compute the AUC of the model on the dataset using the Mutual Information on the alpha and beta chain.
    when masking the MHC, the peptide or the paired alpha/beta chain.

    Args:
        model (_type_): Tulip model
        dataset (_type_): TCRdataset model with positive and negative models
        mask_mhc (bool, optional): either to include mhc in the MI computation. Defaults to True.
        mask_peptide (bool, optional): either to include peptide in the MI computation. Defaults to True.
        mask_paired (bool, optional): either to include the other tcr chain in the MI computation. . Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    ignore_index = dataset.tokenizer.pad_token_id
    clf_scoree, clf_scorea, clf_scoreb = get_logproba(dataset, model, ignore_index)
    dataset2 = copy.deepcopy(dataset)
    if mask_peptide:
        dataset2.peptide = ["<MIS>"] * len(dataset2.peptide)
    if mask_mhc:
        dataset2.MHC = ["<MIS>"] * len(dataset2.MHC)
    if mask_paired:
        dataset2.alpha = ["<MIS>"] * len(dataset2.alpha)
    _, _, clf_scorebmi = get_logproba(dataset2, model, ignore_index)
    clf_scorebmi = np.array(clf_scoreb) - np.array(clf_scorebmi)

    dataset2 = copy.deepcopy(dataset)
    if mask_peptide:
        dataset2.peptide = ["<MIS>"] * len(dataset2.peptide)
    if mask_mhc:
        dataset2.MHC = ["<MIS>"] * len(dataset2.MHC)
    if mask_paired:
        dataset2.beta = ["<MIS>"] * len(dataset2.beta)
    _, clf_scoreami, _ = get_logproba(dataset2, model, ignore_index)
    clf_scoreami = np.array(clf_scorea) - np.array(clf_scoreami)
    return clf_scoreami, clf_scorebmi


import numpy as np


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def get_logscore(dataset, model, ignore_index):
    dataloaderPetideSpecific = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=100,
        shuffle=False,
        collate_fn=dataset.all2allmhc_collate_function,
    )
    model.eval()
    with torch.no_grad():

        clf_scoree = []
        Boolbinders = []
        for i, (peptide, alpha, beta, binder, mhc) in enumerate(
            dataloaderPetideSpecific
        ):
            peptide_input = peptide["input_ids"]
            peptide_mask = peptide["attention_mask"]
            peptide_tokentype = peptide["token_type_ids"]
            alpha_input = alpha["input_ids"]
            alpha_mask = alpha["attention_mask"]
            alpha_tokentype = alpha["token_type_ids"]
            beta_input = beta["input_ids"]
            beta_mask = beta["attention_mask"]
            beta_tokentype = beta["token_type_ids"]

            binder = binder

            clf_label = binder.clone()
            labels = clf_label

            out = model(
                input_ids=(alpha_input, beta_input, peptide_input),
                attention_mask=(alpha_mask, beta_mask, peptide_mask),
                labels=labels,
                mhc=mhc,
            )

            def softm(x):
                x = torch.exp(x)
                su = torch.sum(x, dim=1)
                x = x / su
                return x

            prediction_scoresE = out.decoder_outputsE.lm_logits
            prediction_scoresA = out.decoder_outputsA.lm_logits
            prediction_scoresB = out.decoder_outputsB.lm_logits
            predictionsE = F.log_softmax(prediction_scoresE, dim=2)
            predictionsB = F.log_softmax(prediction_scoresB, dim=2)
            predictionsA = F.log_softmax(prediction_scoresA, dim=2)

            losse = LLLoss_raw(predictionsE, peptide_input, ignore_index)
            clf_scoree += [losse[i].cpu().item() for i in range(len(losse))]
        return clf_scoree


def get_logproba(dataset, model, ignore_index):
    dataloaderPetideSpecific = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=100,
        shuffle=False,
        collate_fn=dataset.all2allmhc_collate_function,
    )
    model.eval()
    with torch.no_grad():

        clf_scoree, clf_scorea, clf_scoreb = [], [], []
        Boolbinders = []
        for i, (peptide, alpha, beta, binder, mhc) in enumerate(
            dataloaderPetideSpecific
        ):
            peptide_input = peptide["input_ids"]
            peptide_mask = peptide["attention_mask"]
            peptide_tokentype = peptide["token_type_ids"]
            alpha_input = alpha["input_ids"]
            alpha_mask = alpha["attention_mask"]
            alpha_tokentype = alpha["token_type_ids"]
            beta_input = beta["input_ids"]
            beta_mask = beta["attention_mask"]
            beta_tokentype = beta["token_type_ids"]

            binder = binder

            clf_label = binder.clone()
            labels = clf_label

            out = model(
                input_ids=(alpha_input, beta_input, peptide_input),
                attention_mask=(alpha_mask, beta_mask, peptide_mask),
                labels=labels,
                mhc=mhc,
            )

            def softm(x):
                x = torch.exp(x)
                su = torch.sum(x, dim=1)
                x = x / su
                return x

            prediction_scoresE = out.decoder_outputsE.lm_logits
            prediction_scoresA = out.decoder_outputsA.lm_logits
            prediction_scoresB = out.decoder_outputsB.lm_logits
            predictionsE = F.log_softmax(prediction_scoresE, dim=2)
            predictionsB = F.log_softmax(prediction_scoresB, dim=2)
            predictionsA = F.log_softmax(prediction_scoresA, dim=2)

            losse = LLLoss_raw(predictionsE, peptide_input, ignore_index)
            lossa = LLLoss_raw(predictionsA, alpha_input, ignore_index)
            lossb = LLLoss_raw(predictionsB, beta_input, ignore_index)
            clf_scoree += [-1 * losse[i].cpu().item() for i in range(len(losse))]
            clf_scorea += [-1 * lossa[i].cpu().item() for i in range(len(lossa))]
            clf_scoreb += [-1 * lossb[i].cpu().item() for i in range(len(lossb))]
        return clf_scoree, clf_scorea, clf_scoreb


def get_auc_mi(model, dataset, mask_mhc=True, mask_peptide=True, mask_paired=False):
    """_summary_
    Compute the AUC of the model on the dataset using the Mutual Information on the alpha and beta chain.
    when masking the MHC, the peptide or the paired alpha/beta chain.

    Args:
        model (_type_): Tulip model
        dataset (_type_): TCRdataset model with positive and negative models
        mask_mhc (bool, optional): either to include mhc in the MI computation. Defaults to True.
        mask_peptide (bool, optional): either to include peptide in the MI computation. Defaults to True.
        mask_paired (bool, optional): either to include the other tcr chain in the MI computation. . Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    ignore_index = dataset.tokenizer.pad_token_id
    clf_scoree, clf_scorea, clf_scoreb = get_logproba(dataset, model, ignore_index)
    dataset2 = copy.deepcopy(dataset)
    if mask_peptide:
        dataset2.peptide = ["<MIS>"] * len(dataset2.peptide)
    if mask_mhc:
        dataset2.MHC = ["<MIS>"] * len(dataset2.MHC)
    if mask_paired:
        dataset2.alpha = ["<MIS>"] * len(dataset2.alpha)
    _, _, clf_scorebmi = get_logproba(dataset2, model, ignore_index)
    aucbmi = roc_auc_score(
        dataset.binder, np.array(clf_scoreb) - np.array(clf_scorebmi)
    )

    dataset2 = copy.deepcopy(dataset)
    if mask_peptide:
        dataset2.peptide = ["<MIS>"] * len(dataset2.peptide)
    if mask_mhc:
        dataset2.MHC = ["<MIS>"] * len(dataset2.MHC)
    if mask_paired:
        dataset2.beta = ["<MIS>"] * len(dataset2.beta)
    _, clf_scoreami, _ = get_logproba(dataset2, model, ignore_index)
    aucami = roc_auc_score(
        dataset.binder, np.array(clf_scorea) - np.array(clf_scoreami)
    )
    return aucami, aucbmi


### SAMPLING FUNCTIONS


def load_model_output(
    tokenizer,
    mhctok,
    peptide,
    alpha_model=None,
    beta_model=None,
    alpha_to_fill=None,
    beta_to_fill=None,
    device="cpu",
    mhc="HLA-A*02:01",
):
    df2 = pd.DataFrame(columns=["CDR3b", "CDR3a", "peptide", "MHC", "binder"])
    print("coucou")
    assert alpha_model is not None or alpha_to_fill is not None, "alpha missing"
    assert beta_model is not None or beta_to_fill is not None, "beta missing"
    if alpha_model is not None:
        alpha_seq = tokenizer.batch_decode(alpha_model, skip_special_tokens=True)
        alpha_seq = [s.replace(" ", "") for s in alpha_seq]
        alpha_seq = [x if x != "" else "A" for x in alpha_seq]
        batchsize_a = len(alpha_seq)
    if beta_model is not None:
        beta_seq = tokenizer.batch_decode(beta_model, skip_special_tokens=True)
        beta_seq = [s.replace(" ", "") for s in beta_seq]
        beta_seq = [x if x != "" else "A" for x in beta_seq]
        batchsize_b = len(beta_seq)
    if alpha_model is None:
        batchsize_a = batchsize_b
        alpha_seq = [alpha_to_fill] * batchsize_a
    if beta_model is None:
        batchsize_b = batchsize_a
        beta_seq = [beta_to_fill] * batchsize_b

    assert batchsize_a == batchsize_b, "batchsize mismatch"
    dataset = TCRDataset.empty_init(tokenizer, device, mhctok)
    for i in range(batchsize_a):
        df2 = pd.concat(
            [
                df2,
                pd.DataFrame(
                    [
                        {
                            "CDR3b": beta_seq[i],
                            "CDR3a": alpha_seq[i],
                            "MHC": mhc,
                            "peptide": peptide,
                            "binder": 1,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        dataset.append(
            MHC=mhc, alpha=alpha_seq[i], beta=beta_seq[i], peptide=peptide, binder=1
        )

    return dataset


def sample_chain_de_novo(
    model,
    tokenizer,
    mhctok,
    starting_batch,
    peptide_str,
    num_return_sequences=1,
    mode="greedy",
    togenerate="B",
    temperature=1.0,
):

    peptide, alpha, beta, binder, mhc = starting_batch
    peptide_input = peptide["input_ids"]
    peptide_mask = peptide["attention_mask"]
    peptide_tokentype = peptide["token_type_ids"]
    alpha_input = alpha["input_ids"]
    alpha_mask = alpha["attention_mask"]
    alpha_tokentype = alpha["token_type_ids"]
    beta_input = beta["input_ids"]
    beta_mask = beta["attention_mask"]
    beta_tokentype = beta["token_type_ids"]

    model_kwargs = {}
    model_kwargs["input_ids"] = (alpha_input, beta_input, peptide_input)
    model_kwargs["attention_mask"] = (alpha_mask, beta_mask, peptide_mask)
    model_kwargs["mhc"] = mhc
    model_kwargs["togenerate"] = togenerate
    model_kwargs["use_cache"] = False

    model_kwargs_copy = copy.deepcopy(model_kwargs)

    if mode == "greedy":
        generate_ids = model.generate(
            inputs=None,  #### Is inputs filtered wtf
            max_length=40,
            do_sample=False,
            early_stopping=False,
            num_beams=1,
            temperature=1.0,
            bos_token_id=tokenizer.cls_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            suppress_tokens=[
                tokenizer.pad_token_id,
                tokenizer.cls_token_id,
                4,
                28,
                25,
                26,
                29,
                27,
                0,
            ],
            **model_kwargs_copy,
        )
        print(generate_ids)

    elif mode == "sampling":
        generate_ids = model.generate(
            inputs=None,  #### Is inputs filtered wtf
            max_length=40,
            do_sample=True,
            temperature=temperature,
            bos_token_id=tokenizer.cls_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            suppress_tokens=[
                tokenizer.pad_token_id,
                tokenizer.cls_token_id,
                4,
                28,
                25,
                26,
                29,
                27,
                0,
            ],
            **model_kwargs_copy,
        )

    elif mode == "beamsearch":
        generate_ids = model.generate(
            inputs=None,  #### Is inputs filtered wtf
            max_length=40,
            do_sample=False,
            early_stopping=False,
            num_beams=10,
            temperature=1.0,
            top_k=None,
            top_p=None,
            bos_token_id=tokenizer.cls_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=num_return_sequences,
            suppress_tokens=[
                tokenizer.pad_token_id,
                tokenizer.cls_token_id,
                4,
                28,
                25,
                26,
                27,
                29,
                0,
            ],
            **model_kwargs_copy,
        )

    else:
        raise ValueError("mode not recognized")

    if togenerate == "B":
        print(generate_ids)
        alpha_start = tokenizer.decode(
            alpha_input[0], skip_special_tokens=True
        ).replace(" ", "")
        beta_start = tokenizer.decode(beta_input[0], skip_special_tokens=True).replace(
            " ", ""
        )
        dts = load_model_output(
            tokenizer,
            mhctok,
            peptide_str,
            alpha_model=alpha_input,
            beta_model=generate_ids,
            alpha_to_fill=None,
            beta_to_fill=None,
            device=model.device,
        )

    elif togenerate == "A":
        alpha_start = tokenizer.decode(
            alpha_input[0], skip_special_tokens=True
        ).replace(" ", "")
        beta_start = tokenizer.decode(beta_input[0], skip_special_tokens=True).replace(
            " ", ""
        )
        dts = load_model_output(
            tokenizer,
            mhctok,
            peptide_str,
            alpha_model=generate_ids,
            beta_model=beta_input,
            alpha_to_fill=None,
            beta_to_fill=None,
            device=model.device,
        )

    return dts, generate_ids


def sample_tcr_denovo(
    model,
    peptide,
    tokenizer,
    mhctok,
    n_recycle=3,
    num_return_sequences=1,
    mode="sampling",
    temperature=1.0,
):
    out_dts = TCRDataset.empty_init(tokenizer, model.device, mhctok=mhctok)
    with torch.inference_mode():
        # for j in range(num_return_sequences):
        starting_batch = get_starting_batch(
            peptide, tokenizer, mhctok, model.device, size=num_return_sequences
        )

        for i in range(n_recycle):
            dts, _ = sample_chain_de_novo(
                model,
                tokenizer,
                mhctok,
                starting_batch,
                peptide,
                num_return_sequences=1,
                mode=mode,
                togenerate="B",
                temperature=temperature,
            )
            print(dts.alpha)
            # [print('alpha', alpha, 'beta',beta, 'peptide',peptide, mhc) for alpha, beta, peptide, mhc in zip(dts.alpha, dts.beta, dts.peptide, dts.MHC)]

            dl = data.DataLoader(
                dts,
                batch_size=len(dts),
                shuffle=False,
                collate_fn=dts.all2allmhc_collate_function,
            )
            starting_batch = next(iter(dl))
            dts, _ = sample_chain_de_novo(
                model,
                tokenizer,
                mhctok,
                starting_batch,
                peptide,
                num_return_sequences=1,
                mode=mode,
                togenerate="A",
                temperature=temperature,
            )
            # [print('alpha', alpha, 'beta',beta, 'peptide',peptide, mhc) for alpha, beta, peptide, mhc in zip(dts.alpha, dts.beta, dts.peptide, dts.MHC)]

            dl = data.DataLoader(
                dts,
                batch_size=len(dts),
                shuffle=False,
                collate_fn=dts.all2allmhc_collate_function,
            )
            starting_batch = next(iter(dl))
        out_dts.concatenate(dts)
    return out_dts


def sample_tcr_denovo_from_chain(
    model,
    peptide,
    tokenizer,
    mhctok,
    datainit,
    n_recycle=1,
    num_return_sequences=1,
    mode="sampling",
    temperature=1.0,
):
    out_dts = TCRDataset.empty_init(tokenizer, model.device, mhctok=mhctok)
    with torch.inference_mode():
        # for j in range(num_return_sequences):
        starting_batch = get_starting_batch_from_chain(peptide, datainit, chain="alpha")

        for i in range(n_recycle):
            dts, _ = sample_chain_de_novo(
                model,
                tokenizer,
                mhctok,
                starting_batch,
                peptide,
                num_return_sequences=1,
                mode=mode,
                togenerate="B",
                temperature=temperature,
            )
            print(dts.alpha)
            # [print('alpha', alpha, 'beta',beta, 'peptide',peptide, mhc) for alpha, beta, peptide, mhc in zip(dts.alpha, dts.beta, dts.peptide, dts.MHC)]

            dl = data.DataLoader(
                dts,
                batch_size=len(dts),
                shuffle=False,
                collate_fn=dts.all2allmhc_collate_function,
            )
            starting_batch = next(iter(dl))
            dts, _ = sample_chain_de_novo(
                model,
                tokenizer,
                mhctok,
                starting_batch,
                peptide,
                num_return_sequences=1,
                mode=mode,
                togenerate="A",
                temperature=temperature,
            )
            # [print('alpha', alpha, 'beta',beta, 'peptide',peptide, mhc) for alpha, beta, peptide, mhc in zip(dts.alpha, dts.beta, dts.peptide, dts.MHC)]

            dl = data.DataLoader(
                dts,
                batch_size=len(dts),
                shuffle=False,
                collate_fn=dts.all2allmhc_collate_function,
            )
            starting_batch = next(iter(dl))
        out_dts.concatenate(dts)
    return out_dts


def get_starting_batch(peptide, tokenizer, mhctok, device, size=1, MHC="HLA-A*02:01"):
    dataset = TCRDataset.empty_init(tokenizer, device, mhctok=mhctok)
    for i in range(size):
        dataset.append(peptide=peptide, MHC=MHC)

    dl = data.DataLoader(
        dataset,
        batch_size=size,
        shuffle=False,
        collate_fn=dataset.all2allmhc_collate_function,
    )
    return next(iter(dl))


def get_starting_batch_from_chain(peptide, datainit, chain="alpha", MHC="HLA-A*02:01"):
    dataset = datainit.select_peptide(peptide)
    dataset = dataset.select_chain(chain)
    dataset.MHC = ["HLA-A*02:01"] * len(dataset)
    if chain == "alpha":
        dataset.beta = ["<MIS>"] * len(dataset)
    if chain == "beta":
        dataset.alpha = ["<MIS>"] * len(dataset)
    dl = data.DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        collate_fn=dataset.all2allmhc_collate_function,
    )
    return next(iter(dl))
