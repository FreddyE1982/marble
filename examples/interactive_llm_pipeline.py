import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marble_main import insert_into_torch_model, MARBLE
from config_loader import load_config
from pipeline import Pipeline


model = None
tokenizer = None
marble = None
cfg = None


def load_llm(llm_name: str) -> None:
    global model, tokenizer
    model = AutoModelForCausalLM.from_pretrained(llm_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(llm_name)


def init_marble(cfg_path: str) -> None:
    global marble, cfg
    cfg = load_config(cfg_path)
    marble = MARBLE(cfg["core"])


def inject_layer(position: str, mode: str = "intermediate") -> None:
    global model, marble
    if model is None or marble is None:
        raise RuntimeError("Model and MARBLE must be initialised first")
    model, _ = insert_into_torch_model(model, marble=marble, position=position, mode=mode)


def interactive_chat() -> None:
    global model, tokenizer, marble
    if model is None or tokenizer is None or marble is None:
        raise RuntimeError("Pipeline not initialised")
    print("\nMARBLE-Enhanced Chat Interface Ready. Type 'exit' to quit.\n")
    history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        history.append({"role": "user", "content": user_input})
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in history]) + "\nassistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True)
        response = (
            tokenizer.decode(outputs[0], skip_special_tokens=True)
            .split("assistant:")[-1]
            .strip()
        )
        print("MARBLE LLM:", response)
        history.append({"role": "assistant", "content": response})
        marble.brain.observe_input_output(user_input, response)
        marble.brain.consolidate()


def main() -> None:
    llm_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    pipe = Pipeline()
    pipe.add_step("load_llm", module=__name__, params={"llm_name": llm_name})
    pipe.add_step("init_marble", module=__name__, params={"cfg_path": "config.yaml"})
    pipe.add_step(
        "inject_layer",
        module=__name__,
        params={"position": "model.layers.10", "mode": "intermediate"},
    )
    pipe.add_step("interactive_chat", module=__name__)
    pipe.execute()


if __name__ == "__main__":
    main()
