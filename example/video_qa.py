# Modified from https://github.com/vllm-project/vllm/blob/main/docs/features/multimodal_inputs.md#video-inputs

from transformers import AutoProcessor
from nanovllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

# Please change to your own video path and model path
video_path = "/home/czh/WorkSpace/svqa/data/videos/sample_213_real.mp4"
model_path = '/home/czh/model_zoo/Qwen2.5-VL-7B-Instruct'

def main():
    
    messages = [
        {  
            "role": "user",  
            "content": [  
                {"type": "text", "text": "describe this video."},  
                {  
                    "type": "video",  
                    "video": video_path, 
                    "total_pixels": 20480 * 28 * 28,
                    "min_pixels": 16 * 28 * 28,
                },  
            ]  
        }, 
    ]
    
    # ================ Prompt Content ================
    # <|im_start|>system
    # You are a helpful assistant.<|im_end|>
    # <|im_start|>user
    # describe this video.<|vision_start|><|video_pad|><|vision_end|><|im_end|>
    # <|im_start|>assistant
    # ================================================
    processor = AutoProcessor.from_pretrained(model_path)
    prompt= processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    _, video_inputs = process_vision_info(messages)
    # video_outputs is the dict with `pixel_values_videos` & `video_grid_thw` keys
    video_outputs = processor.video_processor.preprocess(
        video_inputs, 
        total_pixels=20480 * 28 * 28,
        min_pixels=16 * 28 * 28
    )

    llm_inputs = {
        "prompt": prompt,
        "video_data": video_outputs,
    }

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
    outputs = llm.generate(llm_inputs, sampling_params)

    for output in outputs:
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")
    
    
if __name__ == "__main__":
    main()