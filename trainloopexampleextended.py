# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:47:48 2025

@author: frede
"""

# ----------------------------
# 8. Beispielhafter Trainingsloop inkl. Inferenz (kompletter Prozess)
# ----------------------------
if __name__ == '__main__':
    # 1. Parameter und Core-Erstellung (mit thematisch passender Formel)
    # Wir modellieren das Verhältnis von Textprompt-Menge (T) zu Bildinhalt (I) mit:
    #    Φ = log(1+T) / log(1+I)
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
    formula = "log(1+T)/log(1+I)"
    initial_core = Core(params, formula=formula, formula_num_neurons=100)
    print(f"Initialer Kern: {len(initial_core.neurons)} Neuronen, {len(initial_core.synapses)} Synapsen.")

    # 2. Laden des echten Hugging Face Stable Diffusion 3.5 Large Modells
    # Wir laden den kompletten Text-to-Image-Prozess – hier exemplarisch den Text-Encoder,
    # gehen aber davon aus, dass der gesamte Prozess (Text-Encodierung + Bildgenerierung) durch MARBLE trainiert wird.
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    # Konvertiere das Modell in ein MARBLE-Modell (unabhängig von zusätzlichen Eigenschaften)
    converted_core = MarbleConverter.convert(pipe.text_encoder, mode='sequential', core_params=params)
    core = converted_core
    print(f"Konvertierter Kern: {len(core.neurons)} Neuronen, {len(core.synapses)} Synapsen.")

    # 3. Initialisierung der MARBLE-Komponenten (Brain, Neuronenblitz, DataLoader)
    nb = Neuronenblitz(core, wander_steps=7, learning_rate=0.01)
    dl = DataLoader()
    # Setze das Auto-Firing-Intervall (in ms)
    brain = Brain(core, nb, dl, save_threshold=0.05, max_saved_models=5, firing_interval_ms=500)

    # 4. Laden eines echten Datensatzes von Hugging Face:  
    # Hier verwenden wir den LAION-Aesthetics-v2-5plus Datensatz als Beispiel.
    from datasets import load_dataset
    dataset = load_dataset("laion-aesthetics-v2-5plus", split="train")
    
    # 5. Auswahl einer Teilmenge (subset_size ist ein konfigurierbarer Parameter, hier z.B. 10.000 Records)
    subset_size = 10000
    if len(dataset) > subset_size:
        dataset = dataset.select(range(subset_size))
    
    # 6. Vorbereitung der Trainings- und Validierungsbeispiele:
    # Annahme: Jeder Datensatz-Eintrag enthält "caption" (Text) und "image" (als PIL-Image)
    from io import BytesIO
    def preprocess(sample):
        # Wandle die Caption in UTF-8 Bytes um
        caption_bytes = sample["caption"].strip().encode("utf-8")
        # Wandle das Bild (PIL-Image) in PNG-Binärdaten um
        buffer = BytesIO()
        sample["image"].save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        return caption_bytes, image_bytes

    train_examples = []
    val_examples = []
    for i, sample in enumerate(dataset):
        inp, tgt = preprocess(sample)
        # Trainingsziel: Hier soll das generierte Bild möglichst den echten Bilddaten (tgt) entsprechen.
        target_bytes = tgt  # Für echte Loss-Berechnung nutzen wir das Originalbild.
        if i % 10 == 0:
            val_examples.append((inp, target_bytes))
        else:
            train_examples.append((inp, target_bytes))
    print(f"Trainingsbeispiele: {len(train_examples)}, Validierungsbeispiele: {len(val_examples)}")

    # 7. Definition einer Funktion, die aus dem aktuellen Core ein Bild generiert,
    # passend zur Zielbildgröße eines gegebenen Validierungssamples.
    from PIL import Image
    def generate_image(core, target_shape):
        # target_shape sollte (height, width, channels) sein
        required_neurons = target_shape[0] * target_shape[1] * target_shape[2]
        if len(core.neurons) < required_neurons:
            additional = required_neurons - len(core.neurons)
            core.expand(num_new_neurons=additional, num_new_synapses=0)
            print(f"Core erweitert auf {len(core.neurons)} Neuronen für Bildgenerierung.")
        neuron_values = np.array([n.value for n in core.neurons[-required_neurons:]])
        image_array = neuron_values.reshape(target_shape)
        min_val, max_val = image_array.min(), image_array.max()
        if max_val - min_val > 0:
            image_array = (image_array - min_val) / (max_val - min_val) * 255.0
        else:
            image_array = np.full(target_shape, 127.0)
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        return Image.fromarray(image_array)

    # 8. Starte das Auto-Firing, das alle 500 ms unabhängig vom Trainingsloop ein Bild generiert und Metriken anzeigt.
    brain.start_auto_firing()

    # 9. Trainingsloop mit Live-Metriken via tqdm:
    num_epochs = 5
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", ncols=100)
    for epoch in epoch_pbar:
        brain.train(train_examples, epochs=1, validation_examples=val_examples)
        losses = []
        for inp_bytes, target_bytes in val_examples:
            encoded_prompt = dl.encode(inp_bytes)
            input_scalar = float(np.mean(encoded_prompt))
            nb.dynamic_wander(input_scalar)
            target_img = Image.open(BytesIO(target_bytes))
            target_shape = (target_img.height, target_img.width, 3)
            generated_img = generate_image(core, target_shape)
            target_arr = np.array(target_img.convert("RGB")).astype(np.float32)
            gen_arr = np.array(generated_img).astype(np.float32)
            loss = np.mean((target_arr - gen_arr) ** 2)
            losses.append(loss)
        mean_val_loss = np.mean(losses) if losses else 0
        epoch_pbar.set_postfix({
            "MeanValLoss": f"{mean_val_loss:.4f}",
            "GlobalActs": f"{nb.global_activation_count}",
            "VRAM(MB)": f"{core.get_usage_by_tier('vram'):.2f}"
        })
    epoch_pbar.close()

    # 10. Stoppe das Auto-Firing nach Abschluss des Trainings.
    brain.stop_auto_firing()
    print("\nTraining abgeschlossen.")

    # 11. Inferenzbeispiel: Jetzt soll ein neuer Textprompt verarbeitet werden,
    # und das MARBLE-Modell generiert ein Bild, das live angezeigt wird.
    prompt_text = "A futuristic cityscape at sunset with neon lights."
    prompt_bytes = prompt_text.strip().encode("utf-8")
    encoded_prompt = dl.encode(prompt_bytes)
    input_scalar = float(np.mean(encoded_prompt))
    nb.dynamic_wander(input_scalar)
    _, target_bytes = val_examples[0]
    target_img = Image.open(BytesIO(target_bytes))
    target_shape = (target_img.height, target_img.width, 3)
    final_generated_img = generate_image(core, target_shape)
    final_generated_img.save("final_generated_image.png")
    print("Inferenz abgeschlossen. Das generierte Bild wurde als 'final_generated_image.png' gespeichert.")
