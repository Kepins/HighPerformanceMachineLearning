import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.profiler import profile, record_function, ProfilerActivity
import time
import numpy as np


def n_token_prompt(tokenizer, n, batch_size, device):
    token_id = 18637  # "Hey"
    input_ids = torch.full((batch_size, n), token_id, device=device)
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


def profile_inference(model, tokenizer, device, batch_size=1):
    model.to(device)
    model.eval()
    inputs = n_token_prompt(tokenizer, n=20, batch_size=batch_size, device=device)

    print(f"\nRunning on {device}...")

    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device == "cuda" else [ProfilerActivity.CPU],
            record_shapes=True,
            with_stack=True,
            with_flops=True,
            profile_memory=True
    ) as prof:
        with torch.no_grad():
            with record_function("model_generate"):
                start = time.time()
                output = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    min_new_tokens=20,
                    max_new_tokens=20,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                torch.cuda.synchronize() if device == "cuda" else None
                end = time.time()

    print(f"Inference time on {device}: {end - start:.4f} seconds")
    print(prof.key_averages().table(sort_by="self_cuda_time_total" if device == "cuda" else "self_cpu_time_total",
                                    row_limit=20))


def task_1(model):
    """
    1.
    12*n_layers*d_model^2 = 12 * 22 * 2048**2 = 1107296256 ~ 1,1B

    Per layer
    Num parameters in projection = 2*hidden*hidden_intermediate=23068672 (nie 4x,a ~2,5x)

    K,V mają (256, 2048) a nie jak we wzorze założone 2048, 2048
    """
    number_of_parameters_by_hand = 12 * 22 * 2048 ** 2
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{number_of_parameters_by_hand=}")
    print(f"{trainable_parameters=}")

    total_params = 0
    for name, param in model.named_parameters():
        count = param.numel()
        print(f"{name:<60} {str(tuple(param.shape)):>20} --> {count}")
        total_params += count
    print("=" * 80)
    print(f"Total Parameters: {total_params:,}")


def throughput_inference(model, tokenizer, device, batch_size):
    model.to(device)
    model.eval()
    inputs = n_token_prompt(tokenizer, n=20, batch_size=batch_size, device=device)

    print(f"\nRunning on {device} with batch size {batch_size}...")

    with torch.no_grad():
        start = time.time()
        output = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            min_new_tokens=20,
            max_new_tokens=20,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.time()

    inference_time = end - start
    total_tokens = batch_size * 20
    throughput = total_tokens / inference_time

    print(f"Inference time: {inference_time:.4f} seconds")
    print(f"Throughput: {throughput:.2f} tokens/second")


def profile_execution_time(model, tokenizer, device, batch_size, prompt_length=20, use_cache=True):
    model.to(device)
    model.eval()
    inputs = n_token_prompt(tokenizer, n=prompt_length, batch_size=batch_size, device=device)

    # Not timed
    with torch.no_grad():
        output = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            use_cache=use_cache,
            min_new_tokens=10,
            max_new_tokens=10,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if device == "cuda":
            torch.cuda.synchronize()
    inference_times = []
    for _ in range(3):
        with torch.no_grad():
            start = time.time()
            output = model.generate(
                inputs['input_ids'],
                use_cache=use_cache,
                attention_mask=inputs['attention_mask'],
                min_new_tokens=10,
                max_new_tokens=10,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.time()
        inference_times.append(end - start)
    mean = np.mean(inference_times)
    std = np.std(inference_times)
    print(f"{mean=}")
    print(f"{std=}")


if __name__ == "__main__":
    model_path = "./TinyLlama/TinyLlama-1.1B-Chat-v1.0/"
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path)

    # inputs = n_token_prompt(tokenizer=tokenizer, n=20)
    # output = model.generate(
    #     inputs['input_ids'],
    #     attention_mask=inputs['attention_mask'],
    #     min_new_tokens=20,
    #     max_new_tokens=20,
    #     num_return_sequences=1,
    #     do_sample=True,
    #     temperature=0.7,
    #     top_k=50,
    #     top_p=0.95,
    #     pad_token_id=tokenizer.pad_token_id,
    #     eos_token_id=tokenizer.eos_token_id,
    # )

    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # task_1(model=model)

    ###
    # Lab1 - task2
    ###

    # print("-" * 10 + "WARMUP" + "-" * 10)
    # profile_inference(model, tokenizer, "cpu", batch_size=1)
    # # Profile on GPU (if available)
    # if torch.cuda.is_available():
    #     profile_inference(model, tokenizer, "cuda", batch_size=1)

    # print("-" * 10 + "AFTER WARMUP" + "-" * 10)
    # profile_inference(model, tokenizer, "cpu", batch_size=1)
    # # Profile on GPU (if available)
    # if torch.cuda.is_available():
    #     profile_inference(model, tokenizer, "cuda", batch_size=1)

    # """
    # Większość czasu dla aten:mm
    # """

    # # Profile on CPU
    # for batch_size in [1, 2, 4, 8]:
    #     throughput_inference(model, tokenizer, device="cpu", batch_size=batch_size)

    # # Profile on GPU (if available)
    # if torch.cuda.is_available():
    #     for batch_size in [1, 2, 4, 8]:
    #         throughput_inference(model, tokenizer, device="cuda", batch_size=batch_size)

    ###
    # Lab2 - task0
    ###
    print("----- Eager Mode Inference Profiling - CPU -----")
    profile_execution_time(model, tokenizer, device="cpu", batch_size=1)
    if torch.cuda.is_available():
        print("----- Eager Mode Inference Profiling - CUDA -----")
        profile_execution_time(model, tokenizer, device="cuda", batch_size=1)

    ###
    # Lab2 - task1
    ###
    compiled_model = torch.compile(model)
    print("----- Compiled Mode Inference Profiling - CPU -----")
    profile_execution_time(compiled_model, tokenizer, device="cpu", batch_size=1)
    if torch.cuda.is_available():
        print("----- Compiled Mode Inference Profiling - CUDA -----")
        profile_execution_time(compiled_model, tokenizer, device="cuda", batch_size=1)

    ###
    # Lab2 - task2
    ###
    print("----- No KV Cache - Eager Mode Inference Profiling - CPU -----")
    profile_execution_time(model, tokenizer, device="cpu", batch_size=1, prompt_length=200, use_cache=False)
    if torch.cuda.is_available():
        print("----- No KV Cache - Eager Mode Inference Profiling - CUDA -----")
        profile_execution_time(model, tokenizer, device="cuda", batch_size=1, prompt_length=200, use_cache=False)

    print("----- No KV Cache - Compiled Mode Inference Profiling - CPU -----")
    profile_execution_time(compiled_model, tokenizer, device="cpu", batch_size=1, prompt_length=200, use_cache=False)
    if torch.cuda.is_available():
        print("----- No KV Cache - Compiled Mode Inference Profiling - CUDA -----")
        profile_execution_time(compiled_model, tokenizer, device="cuda", batch_size=1, prompt_length=200,
                               use_cache=False)
