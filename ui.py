import gradio as gr
import torch
import os
import sys

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import GPT2Simple
from tokenizers import Tokenizer

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None

def load_model():
    """Load the model and tokenizer once at startup"""
    global model, tokenizer, device
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load tokenizer
        tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
        print("Tokenizer loaded successfully")
        
        # Load model
        model = GPT2Simple(vocab_size=tokenizer.get_vocab_size()).to(device)
        
        # Check if model weights exist
        if os.path.exists("kidgpt.pt"):
            model.load_state_dict(torch.load("kidgpt.pt", map_location=device))
            print("Model weights loaded successfully")
        else:
            print("Warning: kidgpt.pt not found. Using untrained model.")
        
        model.eval()
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def generate_text(prompt_ids, max_length=100, temperature=1.0, top_k=50):
    """Generate text using the model"""
    global model, device
    
    if model is None:
        return "Error: Model not loaded"
    
    try:
        with torch.no_grad():
            # Start with the prompt
            generated_ids = prompt_ids.clone()
            
            for _ in range(max_length):
                # Get model predictions
                logits = model(generated_ids.unsqueeze(0).to(device))[0, -1, :]
                
                # Apply temperature
                logits = logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = torch.full_like(logits, float('-inf'))
                    logits[top_k_indices] = top_k_logits
                
                # Sample from the distribution
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token.cpu()])
                
                # Stop if we generate an end token or reach max length
                if next_token.item() == tokenizer.token_to_id("<|endoftext|>"):
                    break
                    
        return generated_ids
        
    except Exception as e:
        print(f"Error during generation: {e}")
        return prompt_ids

def generate(text, grade, max_length=100, temperature=1.0):
    """Main generation function for the Gradio interface"""
    global tokenizer
    
    if tokenizer is None:
        return "Error: Model not loaded. Please check if the model files exist."
    
    try:
        # Create the prompt with grade
        prompt = f"<s_grade={grade}> {text}"
        
        # Encode the prompt
        prompt_ids = torch.tensor(tokenizer.encode(prompt).ids)
        
        # Generate text
        generated_ids = generate_text(prompt_ids, max_length, temperature)
        
        # Decode the generated text
        generated_text = tokenizer.decode(generated_ids.tolist())
        
        # Remove the original prompt from the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text if generated_text else "No text generated"
        
    except Exception as e:
        return f"Error during generation: {str(e)}"

# Load the model when the script starts
model_loaded = load_model()

# Create the Gradio interface
iface = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Input Text", placeholder="Enter your question or prompt here..."),
        gr.Slider(minimum=1, maximum=8, value=3, step=1, label="Grade Level"),
        gr.Slider(minimum=10, maximum=200, value=100, step=10, label="Max Length"),
        gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Temperature")
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="KidsGPT - AI Text Generation for Children",
    description="Generate age-appropriate text responses. Adjust the grade level (1-8) to control complexity.",
    examples=[
        ["What is gravity?", 3],
        ["Tell me a story about a cat", 2],
        ["How do plants grow?", 4],
        ["What is the solar system?", 5]
    ]
)

# Launch the interface
if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=True,             # Keep share for public URL if needed
        debug=False
    )
