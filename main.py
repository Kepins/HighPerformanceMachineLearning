import torch
from transformers import LlamaForCausalLM, LlamaTokenizer


def n_token_prompt(tokenizer, n):
    # prompt_text = "Hey! What is life?"
    # return tokenizer(prompt_text, return_tensors="pt")
    token_id = 18637  # Hey

    input_ids = torch.tensor([[token_id] * n])
    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


if __name__ == "__main__":
    model_path = "./TinyLlama/TinyLlama-1.1B-Chat-v1.0/"
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path)

    inputs = n_token_prompt(tokenizer=tokenizer, n=20)

    print(inputs)

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

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"Generated Text: {generated_text}")