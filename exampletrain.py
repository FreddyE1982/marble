
# -----------------------------------------------------
# 8. External main part: Preprocessing, training loop, inference, benchmarking, and dreaming.
# -----------------------------------------------------
if __name__ == '__main__':
    # Core parameters
    params = {
        'xmin': -2.0,
        'xmax': 1.0,
        'ymin': -1.5,
        'ymax': 1.5,
        'width': 30,
        'height': 30,
        'max_iter': 50,
        'vram_limit_mb': 0.5,
        'ram_limit_mb': 1.0,
        'disk_limit_mb': 10
    }
    if not torch.cuda.is_available():
        params['ram_limit_mb'] += params.get('vram_limit_mb', 0)
        params['vram_limit_mb'] = 0
    formula = "log(1+T)/log(1+I)"
    from diffusers import StableDiffusionPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.bfloat16
    ).to(device)
    # Instantiate the MARBLE system with the option to initialize from weights.
    marble_system = MARBLE(params, formula=formula, formula_num_neurons=100, converter_model=pipe.text_encoder, init_from_weights=True)
    core = marble_system.get_core()
    print(f"Core contains {len(core.neurons)} neurons and {len(core.synapses)} synapses.")
    
    from datasets import load_dataset
    dataset = load_dataset("laion-aesthetics-v2-5plus", split="train")
    subset_size = 10000
    if len(dataset) > subset_size:
        dataset = dataset.select(range(subset_size))
    
    # Preprocessing: For each sample, process the text prompt via the text encoder,
    # and use the colored image to compute a target scalar (sum of channel means).
    def preprocess(sample):
        caption = sample["caption"].strip()
        inputs = pipe.tokenizer(caption, return_tensors="pt")
        with torch.no_grad():
            text_embedding = pipe.text_encoder(**inputs).last_hidden_state
        input_scalar = float(text_embedding.mean().item())
        img = sample["image"]
        img_arr = np.array(img).astype(np.float32)
        if img_arr.ndim == 3 and img_arr.shape[2] == 3:
            mean_R = img_arr[:, :, 0].mean()
            mean_G = img_arr[:, :, 1].mean()
            mean_B = img_arr[:, :, 2].mean()
            target_scalar = mean_R + mean_G + mean_B
        else:
            target_scalar = img_arr.mean()
        return input_scalar, target_scalar

    train_examples = []
    val_examples = []
    for i, sample in enumerate(dataset):
        inp, tgt = preprocess(sample)
        if i % 10 == 0:
            val_examples.append((inp, tgt))
        else:
            train_examples.append((inp, tgt))
    print(f"Training examples: {len(train_examples)}, Validation examples: {len(val_examples)}")
    
    # Start auto-firing (continuous inference in the background)
    marble_system.get_brain().start_auto_firing()
    # Start dreaming (offline consolidation)
    marble_system.get_brain().start_dreaming(num_cycles=5, interval=10)
    
    # Training loop with live metrics
    num_epochs = 5
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", ncols=100)
    for epoch in epoch_pbar:
        marble_system.get_brain().train(train_examples, epochs=1, validation_examples=val_examples)
        current_val_loss = marble_system.get_brain().validate(val_examples)
        global_acts = marble_system.get_neuronenblitz().global_activation_count
        vram_usage = core.get_usage_by_tier('vram')
        epoch_pbar.set_postfix({
            "MeanValLoss": f"{current_val_loss:.4f}",
            "GlobalActs": global_acts,
            "VRAM(MB)": f"{vram_usage:.2f}"
        })
    epoch_pbar.close()
    
    marble_system.get_brain().stop_auto_firing()
    marble_system.get_brain().stop_dreaming()
    print("\nTraining completed.")
    
    benchmark_manager = marble_system.get_benchmark_manager()
    dummy_input = random.uniform(0.0, 1.0)
    benchmark_manager.compare(val_examples, dummy_input)
    
    prompt_text = "A futuristic cityscape at sunset with neon lights."
    inputs = pipe.tokenizer(prompt_text, return_tensors="pt")
    with torch.no_grad():
        text_embedding = pipe.text_encoder(**inputs).last_hidden_state
    input_scalar = float(text_embedding.mean().item())
    output_scalar, path = marble_system.get_neuronenblitz().dynamic_wander(input_scalar)
    norm = (output_scalar - math.floor(output_scalar))
    color_val = int(norm * 255)
    image_array = np.full((128, 128, 3), fill_value=color_val, dtype=np.uint8)
    generated_image = Image.fromarray(image_array)
    generated_image.save("generated_image.png")
    print("Inference completed. The generated image has been saved as 'generated_image.png'.")
