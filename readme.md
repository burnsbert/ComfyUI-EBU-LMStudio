# ComfyUI-EBU-LMStudio

A collection of custom nodes for ComfyUI that provide integration with LM Studio. These nodes allow you to load, unload, and interact with LLMs through the LM Studio command-line interface. This can be used to facilitate the automatic creation of descriptive prompts using a LLM, which is helpful for modern image models like Flux that respond well to detailed prompts in natural language.

## Features

- Load and manage LLM models directly from ComfyUI
- Fuzzy model name searching for easy model selection
- Configurable context length and parameters
- Optional image model unloading to free up VRAM
- Simple prompt submission interface
- Option for UTF-8 safe LLM text responses
- Detailed error reporting and status messages

## Installation

1. Copy this directory to your `ComfyUI/custom_nodes/` directory
2. Install LM Studio and make sure 'lms' command line tools are working and that 'Developer Mode' is enabled to run LM Studio as a local server.
3. Make sure to have at least one LLM installed inside of LM Studio.
4. Test this by running `lms ls --detailed` from the command line. You should see a list of all installed LLMs. If this doesn't work you may need to modify your PATH.
5. Restart ComfyUI

## LM Studio
- LM Studio is available at https://lmstudio.ai/
- Information about LM Studio command line tools is available at https://lmstudio.ai/blog/lms

## Nodes

### EBU LM Studio Load Model

Unloads any loaded LLMs and then loads an LLM model into LM Studio with the following parameters:

- `input_string`: Input text (passed through unchanged), serves to trigger the action
- `model_search_string`: Part of the model name to search for (default: "llama"). Spaces are treated as wild cards. First matching LLM is loaded. So "nemo q5" will match "Mistral-Nemo-Instruct-2407-abliterated.Q5_K_M.gguf" for example. You can have multiple searches separated by "|" and one will be chosen at random.
- `offload`: Part of the model offloaded to GPU (from 0.0 to 1.0)
- `context_length`: Model context length (default: 4096)
- `seed`: Random seed to ensure that this runs on each request
- `unload_image_models_first`: Whether to unload ComfyUI image models before loading LLM

Returns:
- `export_string`: The input string (unchanged)
- `model_selected`: Full path of the loaded model or error message
- `models_found`: Complete list of available models

### EBU LM Studio Unload Models

Unloads all currently loaded LLM models from LM Studio.

Parameters:
- `input_string`: Input text (passed through unchanged), serves to trigger the action
- `seed`: Random seed to ensure that this runs on each request

Returns:
- `export_string`: The input string (unchanged)

### EBU LM Studio Unload Guider Models

Unloads all currently loaded LLM models from LM Studio, on a guider line. Great for Flux.

Parameters:
- `input_guider`: Input text (passed through unchanged), serves to trigger the action
- `seed`: Random seed to ensure that this runs on each request

Returns:
- `export_guider`: The input string (unchanged)


### EBU LM Studio Submit Prompt

Submits a prompt to the currently loaded LLM and returns the response.

Parameters:
- `prompt`: The text prompt to send to the LLM
- `system_message`: System context message (default provides image prompt generation context)
- `url`: LM Studio API endpoint (default: http://127.0.0.1:1234/v1/chat/completions)
- `context_length`: Context window size (default: 4096)
- `seed`: Random seed for reproducibility
- `max_tokens`: Maximum response length (default: 300)
- `temp`: Temperature parameter (default: 0.5)
- `top_p`: Top-p sampling parameter (default: 0.9)
- `utf8_safe_replace`: Convert response to safe ASCII characters (default: false)

Returns:
- `generated_text`: The LLM's response

## Requirements

- Python 3.x
- ComfyUI
- LM Studio with CLI access and local server configured

## Common Issues and Solutions

### "Can't connect to LM Studio"
- Ensure LM Studio is running
- Check that LMS CLI Access is healthy
- Verify the port number matches your LM Studio settings
- Confirm LM Studio is in your system PATH

### "no model by that name found"
- Check that the model name search string matches part of your model's name
- Use `lms ls --detailed` in terminal to see available models
- Models must be added to LM Studio before they can be loaded

### Model Loading Issues
- Ensure you have enough VRAM available
- Try unloading image models first
- Check LM Studio logs for detailed error messages

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

- Initially inspired by the LLM node from CrasH Utils Custom Nodes
- Thanks to the ComfyUI and LM Studio teams

## For more help

See the included example workflow `flux_example_ebu_lmstudio.json` to see how these three nodes can work together for VRAM efficient prompt generation to image generation with Flux.
