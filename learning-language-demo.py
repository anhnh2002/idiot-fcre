import gradio as gr
from openai import OpenAI
import os
import tiktoken

# Set your OpenAI API key
client = OpenAI(
    base_url= 'https://api.together.ai/',
    api_key='1b3906427c277704b4db97954af844ea67e6e54839569c85d42b947e832246d0'
)

# Available models configuration with pricing
AVAILABLE_MODELS = {
    "Small-1 0.18$/1M tokens": {
        "name": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K",
        "price": 0.18
    },
    "Small-2 0.3$/1M tokens": {
        "name": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "price": 0.3
    },
    "Small-3 0.3$/1M tokens": {
        "name": "google/gemma-2-9b-it",
        "price": 0.3
    },
    "Medium-1 0.88$/1M tokens": {
        "name": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "price": 0.88
    },
    "Medium-2 1.2$/1M tokens": {
        "name": "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "price": 1.2
    },
    "Medium-3 0.88$/1M tokens": {
        "name": "meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
        "price": 0.88
    },
    "Medium-4 0.88$/1M tokens": {
        "name": "google/gemma-2-27b-it",
        "price": 0.88
    },
    "Medium-5 0.8$/1M tokens": {
        "name": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "price": 0.8
    },
    "Large-1 3.5$/1M tokens": {
        "name": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "price": 3.5
    }
}

def estimate_tokens(text):
    # This is a rough estimate - actual token count may vary
    return len(text.split()) * 1.3

def calculate_cost(token_count, model_name):
    price_per_million = AVAILABLE_MODELS[model_name]["price"]
    return (token_count / 1000000) * price_per_million

def process_with_model(prompt, model_name):
    try:
        # Estimate tokens and cost
        token_count = estimate_tokens(prompt)
        estimated_cost = calculate_cost(token_count, model_name)
        
        response = client.chat.completions.create(
            model=AVAILABLE_MODELS[model_name]["name"],
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            top_p=0.5
        )
        
        response_text = response.choices[0].message.content.strip()

        if response_text[0] == '"':
            response_text = response_text[1:]
        if response_text[-1] == '"':
            response_text = response_text[:-1]

        response_tokens = estimate_tokens(response_text)
        total_tokens = token_count + response_tokens
        total_cost = calculate_cost(total_tokens, model_name)
        
        return response_text, f"Estimated tokens: {int(total_tokens)}\nEstimated cost: ${total_cost:.6f}"
    except Exception as e:
        return f"Error: {str(e)}", "Error calculating cost"

def translate_text(text, target_lang, model_name):
    prompt = f"Translate the following text to {target_lang}:\n\"\"\"{text}\n\"\"\"\nOnly return the translated text without any additional information."
    return process_with_model(prompt, model_name)

def write_content(prompt, style, model_name):
    full_prompt = f"Write content about:\n\"\"\"{prompt}\n\"\"\"\nStyle: {style}\n\nOnly return the written content without any additional information."
    return process_with_model(full_prompt, model_name)

def grammar_check(text, model_name):
    prompt = f"Check and correct any grammar or spelling mistakes in the following text. Provide the corrected version and explain the corrections:\n\"\"\"{text}\n\"\"\"\nOnly return the corrected text, detailed corrections, and explanations without any additional information."
    return process_with_model(prompt, model_name)

def paraphrase(text, style, model_name):
    prompt = f"Paraphrase the following text in a {style} style:\n\"\"\"{text}\n\"\"\"\nOnly return the paraphrased text without any additional information."
    return process_with_model(prompt, model_name)

# Create the Gradio interface
with gr.Blocks(title="Learning Languages Demo") as demo:
    gr.Markdown("# Learning Languages Demo")
    
    # Common model selector
    model_selector = gr.Dropdown(
        choices=list(AVAILABLE_MODELS.keys()),
        value=list(AVAILABLE_MODELS.keys())[0],
        label="Select Model",
        info="Choose the model to use for processing"
    )
    
    with gr.Tab("Translation"):
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(label="Text to be translated", lines=5)
                target_lang = gr.Dropdown(
                    choices=["English", "Vietnamese", "French", "German", "Spanish", "Chinese", "Japanese", "Korean"],
                    label="Target language",
                    value="English"
                )
                translate_btn = gr.Button("Dịch")
            with gr.Column():
                output_translation = gr.Textbox(label="Result", lines=5)
                cost_info_translation = gr.Textbox(label="Estimated cost", lines=2)
        
        translate_btn.click(
            translate_text,
            inputs=[input_text, target_lang, model_selector],
            outputs=[output_translation, cost_info_translation]
        )
    
    with gr.Tab("Content Writing"):
        with gr.Row():
            with gr.Column():
                content_prompt = gr.Textbox(label="Required content", lines=3)
                content_style = gr.Dropdown(
                    choices=["Academic", "Professional", "Casual", "Creative"],
                    label="Style",
                    value="Professional"
                )
                write_btn = gr.Button("Write")
            with gr.Column():
                output_content = gr.Textbox(label="Result", lines=8)
                cost_info_content = gr.Textbox(label="Estimated cost", lines=2)
        
        write_btn.click(
            write_content,
            inputs=[content_prompt, content_style, model_selector],
            outputs=[output_content, cost_info_content]
        )
    
    with gr.Tab("Grammar Checking"):
        with gr.Row():
            with gr.Column():
                grammar_text = gr.Textbox(label="Text to check", lines=5)
                check_btn = gr.Button("Check")
            with gr.Column():
                grammar_output = gr.Textbox(label="Result", lines=5)
                cost_info_grammar = gr.Textbox(label="Estimated cost", lines=2)
        
        check_btn.click(
            grammar_check,
            inputs=[grammar_text, model_selector],
            outputs=[grammar_output, cost_info_grammar]
        )
    
    with gr.Tab("ParaPhrasing"):
        with gr.Row():
            with gr.Column():
                paraphrase_text = gr.Textbox(label="Text to paraphase", lines=5)
                paraphrase_style = gr.Dropdown(
                    choices=["Simple", "Professional", "Academic", "Casual"],
                    label="Style",
                    value="Professional"
                )
                paraphrase_btn = gr.Button("Paraphase")
            with gr.Column():
                paraphrase_output = gr.Textbox(label="Result", lines=5)
                cost_info_paraphrase = gr.Textbox(label="Estimated cost", lines=2)
        
        paraphrase_btn.click(
            paraphrase,
            inputs=[paraphrase_text, paraphrase_style, model_selector],
            outputs=[paraphrase_output, cost_info_paraphrase]
        )

# Launch the demo
if __name__ == "__main__":
    demo.launch(share=True)