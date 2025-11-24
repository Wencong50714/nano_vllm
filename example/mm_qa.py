from transformers import AutoProcessor
from nanovllm import LLM, SamplingParams
import asyncio

# Please change to your own video path and model path
model_path = '/data/zbw/huggingface/model_zoo/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/'

async def image_example():
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "/home/czh/workspace/nano_vllm/assets/image.jpg",
                },
                {"type": "text", "text": "describe this image"},
            ],
        }
    ]
    
    processor = AutoProcessor.from_pretrained(model_path)
    # `<|vision_start|><|image_pad|><|vision_end|>`
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompts = [prompt]
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.6, max_tokens=2048)
    outputs = await llm.generate(prompts, sampling_params, image_data=messages[0]["content"][0]["image"])
    
    print(outputs)

async def video_example():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "/home/czh/workspace/nano_vllm/assets/video.mp4",
                },
                {"type": "text", "text": "describe this video in short"},
            ],
        }
    ]
    
    processor = AutoProcessor.from_pretrained(model_path)
    # `<|vision_start|><|video_pad|><|vision_end|>`
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompts = [prompt]
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.6, max_tokens=65536)
    outputs = await llm.generate(prompts, sampling_params, video_data=messages[0]["content"][0]["video"])
    
    print(outputs)
    

if __name__ == "__main__":
    # asyncio.run(image_example())
    asyncio.run(video_example())