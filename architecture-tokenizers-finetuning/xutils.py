import json
from typing import List, Dict, Optional, Set, Union

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessor,
    LogitsProcessorList,
)


class AllowOnlyTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids: Set[int]):
        self.allowed_token_ids = set(int(t) for t in allowed_token_ids)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        vocab_size = scores.shape[-1]
        device = scores.device

        allowed = [
            token_id
            for token_id in self.allowed_token_ids
            if 0 <= token_id < vocab_size
        ]

        if not allowed:
            raise ValueError("No valid allowed token ids for this model vocabulary.")

        mask = torch.full(
            (vocab_size,),
            -float("inf"),
            device=device,
            dtype=scores.dtype,
        )

        mask[torch.tensor(allowed, device=device, dtype=torch.long)] = 0.0
        return scores + mask


class HFChatGenerator:
    def __init__(
        self,
        model_id: str,
        allowed_token_json_path: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype: Union[str, torch.dtype] = "auto",
        check_model_id: bool = True,
    ):
        self.model_id = model_id
        self.allowed_token_json_path = allowed_token_json_path
        self.check_model_id = check_model_id

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        self.model.eval()

        self.allowed_token_ids = None
        if allowed_token_json_path is not None:
            self.allowed_token_ids = self._load_allowed_token_ids(
                allowed_token_json_path
            )

    def _load_allowed_token_ids(self, allowed_token_json_path: str) -> Set[int]:
        with open(allowed_token_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            if (
                self.check_model_id
                and "model_id" in data
                and data["model_id"] != self.model_id
            ):
                raise ValueError(
                    f"JSON file is for model_id={data['model_id']!r}, "
                    f"but current model is {self.model_id!r}."
                )

            allowed = set(int(t) for t in data["allowed_token_ids"])

        elif isinstance(data, list):
            allowed = set(int(t) for t in data)

        else:
            raise ValueError(
                "Allowed token JSON must be either a list of ids "
                "or a dict with key 'allowed_token_ids'."
            )

        # Add regular special tokens of the tokenizer.
        for token_id in self.tokenizer.all_special_ids:
            if token_id is not None:
                allowed.add(int(token_id))

        for token_id in [
            self.tokenizer.eos_token_id,
            self.tokenizer.bos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
        ]:
            if token_id is not None:
                allowed.add(int(token_id))

        return allowed

    def _build_inputs(
        self,
        messages: List[Dict[str, str]],
        debug_template: bool = False,
    ):
        """
        This is where the tokenizer's chat template is used.
        """

        if debug_template:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            print("=== CHAT TEMPLATE PROMPT ===")
            print(prompt_text)
            print("============================")

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def _decode_new_tokens(
        self,
        output_ids: torch.LongTensor,
        input_len: int,
        skip_special_tokens: bool = True,
    ) -> str:
        new_tokens = output_ids[0, input_len:]
        return self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=skip_special_tokens,
        )

    def _base_generation_kwargs(
        self,
        inputs,
        max_new_tokens: int,
        do_sample: bool,
        temperature: Optional[float],
    ) -> dict:
        kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        if do_sample:
            kwargs["do_sample"] = True
            if temperature is not None:
                kwargs["temperature"] = temperature
        else:
            kwargs["do_sample"] = False

        return kwargs

    @torch.no_grad()
    def generate_unconstrained(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        debug_template: bool = False,
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Regular generation, but still using tokenizer.apply_chat_template.
        No constrained decoding.
        """

        inputs = self._build_inputs(
            messages=messages,
            debug_template=debug_template,
        )

        generation_kwargs = self._base_generation_kwargs(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )

        output_ids = self.model.generate(**generation_kwargs)

        return self._decode_new_tokens(
            output_ids=output_ids,
            input_len=inputs["input_ids"].shape[-1],
            skip_special_tokens=skip_special_tokens,
        )

    @torch.no_grad()
    def generate_constrained(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        debug_template: bool = False,
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Constrained generation using tokenizer.apply_chat_template
        and AllowOnlyTokensLogitsProcessor.
        """

        if self.allowed_token_ids is None:
            raise ValueError(
                "allowed_token_json_path was not provided. "
                "Cannot run constrained generation."
            )

        inputs = self._build_inputs(
            messages=messages,
            debug_template=debug_template,
        )

        generation_kwargs = self._base_generation_kwargs(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )

        generation_kwargs["logits_processor"] = LogitsProcessorList([
            AllowOnlyTokensLogitsProcessor(self.allowed_token_ids)
        ])

        output_ids = self.model.generate(**generation_kwargs)

        return self._decode_new_tokens(
            output_ids=output_ids,
            input_len=inputs["input_ids"].shape[-1],
            skip_special_tokens=skip_special_tokens,
        )

    def print_device_info(self):
        import torch

        print("CUDA available:", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}:", torch.cuda.get_device_name(i))

        print("\nmodel.device:")
        print(self.model.device)

        print("\nhf_device_map:")
        if hasattr(self.model, "hf_device_map"):
            print(self.model.hf_device_map)
        else:
            print("No hf_device_map found")

        if torch.cuda.is_available():
            print("\nGPU memory:")
            print("Allocated GB:", torch.cuda.memory_allocated() / 1024 ** 3)
            print("Reserved GB:", torch.cuda.memory_reserved() / 1024 ** 3)