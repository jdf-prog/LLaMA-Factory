# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, PreTrainedModel
from trl import DPOTrainer
from trl.trainer import disable_dropout_in_model

from ...extras.constants import IGNORE_INDEX
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimzer, create_custom_scheduler, get_batch_logps
from .utils import get_diff_label_indices


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomVAPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.finetuning_args = finetuning_args
        self.f_divergence_type = "reverse_kl"
        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # dpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma

        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.callback_handler.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes ORPO's odds ratio (OR) loss for batched log probabilities of the policy model.
        """
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + self.beta * odds_ratio_loss
        return orpo_loss

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes SimPO loss for batched log probabilities of the policy model.
        """
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.beta * logits)
        simpo_loss += chosen_logps
        return simpo_loss
    
    def simpo_sft_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes SimPO loss for batched log probabilities of the policy model.
        """
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.beta * logits)
        simpo_loss += chosen_logps
        return simpo_loss

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes loss for preference learning.
        """
        if not self.finetuning_args.use_ref_model:
            if self.loss_type == "orpo":
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
            elif self.loss_type == "simpo_sft":
                losses = self.simpo_sft_loss(policy_chosen_logps, policy_rejected_logps)
            else:
                raise NotImplementedError("Unknown loss type: {}.".format(self.loss_type))

            chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
            rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        else:
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
            )

        return losses, chosen_rewards, rejected_rewards

    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

        Otherwise the average log probabilities.
        """
        if self.finetuning_args.use_ref_model:
            batch = {k: v.detach().clone() for k, v in batch.items()}  # avoid error

        all_logits: "torch.Tensor" = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)

        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length

    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Computes log probabilities of the reference model.
        """
        if not self.finetuning_args.use_ref_model:
            return None, None

        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        with torch.no_grad(), ref_context:
            reference_chosen_logps, reference_rejected_logps, *_ = self.concatenated_forward(ref_model, batch)

        return reference_chosen_logps, reference_rejected_logps

    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        r"""
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.
        """
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_logps_avg,
        ) = self.concatenated_forward(model, batch)

        reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)
        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        sft_loss = -policy_chosen_logps_avg
        if self.ftx_gamma > 1e-6:
            losses += self.ftx_gamma * sft_loss

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics["{}rewards/chosen".format(prefix)] = chosen_rewards.mean().cpu()
        metrics["{}rewards/rejected".format(prefix)] = rejected_rewards.mean().cpu()
        metrics["{}rewards/accuracies".format(prefix)] = reward_accuracies.mean().cpu()
        metrics["{}rewards/margins".format(prefix)] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics["{}logps/rejected".format(prefix)] = policy_rejected_logps.detach().mean().cpu()
        metrics["{}logps/chosen".format(prefix)] = policy_chosen_logps.detach().mean().cpu()
        metrics["{}logits/rejected".format(prefix)] = policy_rejected_logits.detach().mean().cpu()
        metrics["{}logits/chosen".format(prefix)] = policy_chosen_logits.detach().mean().cpu()
        if self.loss_type == "orpo":
            metrics["{}sft_loss".format(prefix)] = sft_loss.detach().mean().cpu()
            metrics["{}odds_ratio_loss".format(prefix)] = ((losses - sft_loss) / self.beta).detach().mean().cpu()

        return losses.mean(), metrics

    # def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
    #     """Tokenize a single row from a DPO specific dataset.

    #     At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
    #     in case the prompt + chosen or prompt + rejected responses is/are too long. First
    #         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

    #     We also create the labels for the chosen/rejected responses, which are of length equal to
    #         the sum of the length of the prompt and the chosen/rejected response, with
    #         label_pad_token_id  for the prompt tokens.
    #     """
    #     batch = {}
    #     prompt = feature["prompt"]
    #     chosen = feature["chosen"]
    #     rejected = feature["rejected"]
    #     images = feature.get("images")

    #     if not self.is_encoder_decoder:
    #         # Check issues below for more details
    #         #  1. https://github.com/huggingface/trl/issues/907
    #         #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
    #         #  3. https://github.com/LianjiaTech/BELLE/issues/337

    #         if not isinstance(prompt, str):
    #             raise ValueError(f"prompt should be an str but got {type(prompt)}")
    #         if self.is_vision_model:
    #             prompt_tokens = self.processor(prompt, images=images, add_special_tokens=False)
    #             prompt_tokens = {k: v[0] for k, v in prompt_tokens.items()}  # Unbatch, not done when using idefics
    #         else:
    #             prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)

    #         prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

    #         if not isinstance(chosen, str):
    #             raise ValueError(f"chosen should be an str but got {type(chosen)}")

    #         chosen_tokens = self.build_tokenized_answer(prompt, chosen, images)

    #         if not isinstance(rejected, str):
    #             raise ValueError(f"rejected should be an str but got {type(rejected)}")
    #         rejected_tokens = self.build_tokenized_answer(prompt, rejected, images)

    #         # Last prompt token might get merged by tokenizer and
    #         # it should not be included for generation if that happens
    #         prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

    #         chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
    #         rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
    #         prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

    #         for k, v in prompt_tokens.items():
    #             prompt_tokens[k] = v[:prompt_len_input_ids]

    #         # Make sure prompts only have one different token at most an
    #         # and length only differs by 1 at most
    #         num_diff_tokens = sum(
    #             [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
    #         )
    #         num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
    #         if num_diff_tokens > 1 or num_diff_len > 1:
    #             raise ValueError(
    #                 "Chosen and rejected prompt_input_ids might only differ on the "
    #                 "last token due to tokenizer merge ops."
    #             )

    #         # add BOS token to head of prompt. Avoid adding if it's already there
    #         bos_token_id = self.tokenizer.bos_token_id
    #         if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
    #             prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
    #             prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
    #         if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
    #             chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
    #             chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
    #         if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
    #             rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
    #             rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

    #         # add EOS token to end of answer. Avoid adding if it's already there
    #         eos_token_id = self.tokenizer.eos_token_id
    #         if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
    #             chosen_tokens["input_ids"].append(eos_token_id)
    #             chosen_tokens["attention_mask"].append(1)
    #         if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
    #             rejected_tokens["input_ids"].append(eos_token_id)
    #             rejected_tokens["attention_mask"].append(1)

    #         longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

    #         # if combined sequence is too long, truncate the prompt
    #         for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
    #             if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
    #                 if self.truncation_mode == "keep_start":
    #                     for k in ["prompt_input_ids", "prompt_attention_mask"]:
    #                         answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
    #                 elif self.truncation_mode == "keep_end":
    #                     for k in ["prompt_input_ids", "prompt_attention_mask"]:
    #                         answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
    #                 else:
    #                     raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

    #         # if that's still too long, truncate the response
    #         for answer_tokens in [chosen_tokens, rejected_tokens]:
    #             if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
    #                 for k in ["input_ids", "attention_mask"]:
    #                     answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

    #         # Create labels
    #         chosen_sequence_tokens = {
    #             k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
    #         }
    #         rejected_sequence_tokens = {
    #             k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
    #         }
    #         # same shape as input_ids, but with pad_token_id for prompt tokens
    #         chosen_sequence_tokens["labels"] = torch.tensor(
    #             [self.label_pad_token_id] * len(chosen_sequence_tokens["input_ids"])
    #         )
    #         rejected_sequence_tokens["labels"] = torch.tensor(
    #             [self.label_pad_token_id] * len(rejected_sequence_tokens["input_ids"])
    #         )
    #         chosen_label_indices, rejected_label_indices = get_diff_label_indices(
    #             chosen_sequence_tokens["labels"][len(chosen_tokens["prompt_input_ids"]) :],
    #             rejected_sequence_tokens["labels"][len(rejected_tokens["prompt_input_ids"]) :],
    #         )
    #         chosen_sequence_tokens["labels"][len(chosen_tokens["prompt_input_ids"]) :][chosen_label_indices] = \
    #             chosen_sequence_tokens["input_ids"][len(chosen_tokens["prompt_input_ids"]) :][chosen_label_indices]
    #         rejected_sequence_tokens["labels"][len(rejected_tokens["prompt_input_ids"]) :][rejected_label_indices] = \
    #             rejected_sequence_tokens["input_ids"][len(rejected_tokens["prompt_input_ids"]) :][rejected_label_indices]
            
    #         # decode labels and input_ids to verify, replace label_pad_token_id with 0 so it can be decoded
    #         _to_debug_chosen_labels = torch.where(
    #             chosen_sequence_tokens["labels"] == self.label_pad_token_id, torch.tensor(0), chosen_sequence_tokens["labels"]
    #         )
    #         _to_debug_rejected_labels = torch.where(
    #             rejected_sequence_tokens["labels"] == self.label_pad_token_id, torch.tensor(0), rejected_sequence_tokens["labels"]
    #         )
    #         print(f"Prompt: {self.tokenizer.decode(prompt_tokens['prompt_input_ids'])}")
    #         print(f"Chosen: {self.tokenizer.decode(chosen_sequence_tokens['input_ids'])}")
    #         print(f"Chosen Labels: {self.tokenizer.decode(_to_debug_chosen_labels)}")
    #         print(f"Rejected: {self.tokenizer.decode(rejected_sequence_tokens['input_ids'])}")
    #         print(f"Rejected Labels: {self.tokenizer.decode(_to_debug_rejected_labels)}")


    #         for k, toks in {
    #             "chosen_": chosen_sequence_tokens,
    #             "rejected_": rejected_sequence_tokens,
    #             "": prompt_tokens,
    #         }.items():
    #             for type_key, tokens in toks.items():
    #                 if type_key == "token_type_ids":
    #                     continue
    #                 batch[f"{k}{type_key}"] = tokens

    #     else:
    #         chosen_tokens = self.tokenizer(
    #             chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
    #         )
    #         rejected_tokens = self.tokenizer(
    #             rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True
    #         )
    #         prompt_tokens = self.tokenizer(
    #             prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
    #         )

    #         batch["chosen_labels"] = chosen_tokens["input_ids"]
    #         batch["rejected_labels"] = rejected_tokens["input_ids"]

    #         chosen_label_indices, rejected_label_indices = get_diff_label_indices(
    #             batch["chosen_labels"], batch["rejected_labels"]
    #         )
    #         batch["chosen_labels"][chosen_label_indices] = self.label_pad_token_id
    #         batch["rejected_labels"][rejected_label_indices] = self.label_pad_token_id

    #         batch["prompt_input_ids"] = prompt_tokens["input_ids"]
    #         batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

    #         if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
    #             batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
    #                 labels=torch.tensor(batch["rejected_labels"])
    #             )
    #             batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
    #                 labels=torch.tensor(batch["chosen_labels"])
    #             )

    #     return batch