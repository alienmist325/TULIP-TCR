import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig
from sklearn.metrics import roc_auc_score, PrecisionRecallDisplay

from transformers.models.encoder_decoder.configuration_encoder_decoder import (
    EncoderDecoderConfig,
)
from src.multiTrans import (
    TulipPetal,
    TCRDataset,
    BertLastPooler,
    unsupervised_auc,
    train_unsupervised,
    eval_unsupervised,
    MyMasking,
    Tulip,
    get_logscore,
    get_mi,
    get_auc_mi,
)

import argparse
import os
from datetime import datetime

torch.manual_seed(0)


def get_input_identifier(path: str):
    """
    Asssumes path = "{arbitrary prefix}_{identifier}.csv"
    """
    path = path[0:-4]
    parts = path.split("_")
    del parts[0]
    return "_".join(parts)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_dir",
        default=None,
        type=str,
        required=True,
        help="The test data dir. Should contain the .fasta files (or other data files) for the task.",
    )
    parser.add_argument(
        "--modelconfig",
        default="configs/shallow.config.json",
        type=str,
        help="path to json including the config of the model",
    )
    parser.add_argument(
        "--load",
        default="model_weights/pytorch_model.bin",
        type=str,
        help="path to the model pretrained to load",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help="path to save results",
    )
    parser.add_argument(
        "--batch_size",
        default=512,
        type=int,
        help="batch_size",
    )

    args = parser.parse_args()

    starttime = datetime.now()

    with open(args.modelconfig, "r") as read_file:
        print("loading hyperparameter")
        modelconfig = json.load(read_file)

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_path = args.test_dir
    # train_path = args.train_dir

    tokenizer = AutoTokenizer.from_pretrained("aatok/")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({"sep_token": "<MIS>"})

    if tokenizer.cls_token is None:
        tokenizer.add_special_tokens({"cls_token": "<CLS>"})

    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<EOS>"})

    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<MASK>"})

    from tokenizers.processors import TemplateProcessing

    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single="<CLS> $A <EOS>",
        pair="<CLS> $A <MIS> $B:1 <EOS>:1",
        special_tokens=[
            ("<EOS>", 2),
            ("<CLS>", 3),
            ("<MIS>", 4),
        ],
    )

    mhctok = AutoTokenizer.from_pretrained("mhctok/")
    vocabsize = len(tokenizer._tokenizer.get_vocab())
    mhcvocabsize = len(mhctok._tokenizer.get_vocab())
    print(mhcvocabsize)
    print("Loading models ..")
    # vocabsize = encparams["vocab_size"]

    max_length = 50
    encoder_config = BertConfig(
        vocab_size=vocabsize,
        max_position_embeddings=max_length,  # this shuold be some large value
        num_attention_heads=modelconfig["num_attn_heads"],
        num_hidden_layers=modelconfig["num_hidden_layers"],
        hidden_size=modelconfig["hidden_size"],
        type_vocab_size=1,
        pad_token_id=tokenizer.pad_token_id,
    )

    encoder_config.mhc_vocab_size = mhcvocabsize

    encoderA = BertModel(config=encoder_config)
    encoderB = BertModel(config=encoder_config)
    encoderE = BertModel(config=encoder_config)

    max_length = 100
    max_length = 50
    decoder_config = BertConfig(
        vocab_size=vocabsize,
        max_position_embeddings=max_length,  # this shuold be some large value
        num_attention_heads=modelconfig["num_attn_heads"],
        num_hidden_layers=modelconfig["num_hidden_layers"],
        hidden_size=modelconfig["hidden_size"],
        type_vocab_size=1,
        is_decoder=True,
        pad_token_id=tokenizer.pad_token_id,
    )  # Very Important

    decoder_config.add_cross_attention = True

    decoderA = TulipPetal(config=decoder_config)  # BertForMaskedLM
    decoderA.pooler = BertLastPooler(config=decoder_config)
    decoderB = TulipPetal(config=decoder_config)  # BertForMaskedLM
    decoderB.pooler = BertLastPooler(config=decoder_config)
    decoderE = TulipPetal(config=decoder_config)  # BertForMaskedLM
    decoderE.pooler = BertLastPooler(config=decoder_config)

    # Define encoder decoder model

    model = Tulip(
        encoderA=encoderA,
        encoderB=encoderB,
        encoderE=encoderE,
        decoderA=decoderA,
        decoderB=decoderB,
        decoderE=decoderE,
    )
    if torch.cuda.is_available():
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint)

    else:
        checkpoint = torch.load(args.load, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint)

    model.to(device)
    target_peptidesFinal = pd.read_csv(test_path)["peptide"].unique()

    prd_choice = input("Would you like to print all the graphs?")

    if prd_choice == "y":
        plot_prd = True
    else:
        plot_prd = False

    auc_choice = input("Would you like to compute AUCs?")

    if auc_choice == "y":
        compute_auc = True
    else:
        compute_auc = False

    mi_choice = input("Would you like to compute the mi?")

    if mi_choice == "y":
        compute_mi = True
    else:
        compute_mi = False

    for target_peptide in target_peptidesFinal:
        results = pd.DataFrame(
            columns=["CDR3a", "CDR3b", "peptide", "score", "rank", "binder"]
        )
        datasetPetideSpecific = TCRDataset(
            test_path, tokenizer, device, target_peptide=target_peptide, mhctok=mhctok
        )
        print(target_peptide)
        scores = -1 * np.array(
            get_logscore(
                datasetPetideSpecific, model, ignore_index=tokenizer.pad_token_id
            )
        )

        header = "receptor_number,CDR3a,CDR3b,peptide,score,rank,binder\n"

        ranks = np.argsort(np.argsort(scores))
        results["CDR3a"] = datasetPetideSpecific.alpha
        results["CDR3b"] = datasetPetideSpecific.beta
        results["peptide"] = target_peptide
        results["rank"] = ranks
        results["score"] = scores
        results["binder"] = datasetPetideSpecific.binder

        if compute_mi:
            mi_scores_a, mi_scores_b = np.array(get_mi(model, datasetPetideSpecific))
            mi_auc_scores_a, mi_auc_scores_b = np.array(
                get_auc_mi(model, datasetPetideSpecific)
            )
            print(mi_auc_scores_a)
            print(mi_auc_scores_b)
            results["mi_scores_a"] = mi_scores_a
            results["mi_scores_b"] = mi_scores_b

            header = "receptor_number,CDR3a,CDR3b,peptide,score,rank,binder,mi_scores_a,mi_scores_b\n"

        # results.to_csv(args.output + target_peptide + ".csv")

        number = len(ranks)

        current_datetime = starttime.strftime("%Y%m%d-%H%M%S")
        input_identifier = get_input_identifier(test_path)

        output_path = (
            args.output
            + "/output_"
            + input_identifier
            + "-"
            + current_datetime
            + ".csv"
        )
        if not os.path.isfile(output_path):
            with open(output_path, "w") as file:
                file.write(header)

        results.to_csv(output_path, mode="a", header=False)

        if compute_auc:
            dl = torch.utils.data.DataLoader(
                dataset=datasetPetideSpecific,
                batch_size=1,
                shuffle=False,
                collate_fn=datasetPetideSpecific.all2allmhc_collate_function,
            )
            # print(unsupervised_auc(model,dl, tokenizer.pad_token_id))
            auce = roc_auc_score(datasetPetideSpecific.binder, ranks)
            aucs = pd.DataFrame(columns=["peptide", "auc"])
            aucs["peptide"] = [target_peptide]
            aucs["auc"] = [auce]

            output_path = (
                args.output
                + "/output_"
                + input_identifier
                + "_auc-"
                + current_datetime
                + ".csv"
            )
            if not os.path.isfile(output_path):
                with open(output_path, "w") as file:
                    file.write("index,peptide,auc\n")

            aucs.to_csv(output_path, mode="a", header=False)

        if plot_prd:
            import matplotlib.pyplot as plt

            plt.rcParams["figure.figsize"] = (10, 8)
            plt.rcParams["font.size"] = 25
            plt.rcParams["lines.linewidth"] = 4
            plt.rcParams["xtick.labelbottom"] = True

            name = f"{target_peptide}-{number}"
            display = PrecisionRecallDisplay.from_predictions(
                datasetPetideSpecific.binder,
                ranks,
                name=None,
                plot_chance_level=True,
            )
            prd_output = args.output + f"/graphs/prd-{number}-{target_peptide}.png"

            plt.xlabel("Recall", labelpad=20)
            plt.ylabel("Precision", labelpad=20)
            plt.title(f"{target_peptide} against {number} TCRs")
            plt.savefig(prd_output, bbox_inches="tight")
            # print(auce)


if __name__ == "__main__":
    main()
