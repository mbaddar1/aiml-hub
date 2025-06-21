from vllm import LLM

# Sample prompts.
text_1 = "What is the capital of France?"
texts_2 = [
    "The capital of Brazil is Brasilia.", "The capital of France is Paris."
]

# Create an LLM.
# You should pass task="score" for cross-encoder models
model = LLM(
    model="BAAI/bge-reranker-v2-m3",
    task="score",
    enforce_eager=True,
)

# Generate scores. The output is a list of ScoringRequestOutputs.
outputs = model.score(text_1, texts_2)

# Print the outputs.
for text_2, output in zip(texts_2, outputs):
    score = output.outputs.score
    print(f"Pair: {[text_1, text_2]!r} | Score: {score}")
# import re
#
# from loguru import logger
# from vllm import LLM, SamplingParams
# from datasets import load_dataset
# if __name__=='__main__':
#     ds_name = "Rowan/hellaswag"
#     ds = load_dataset(ds_name)
#     ds_train = ds["train"]
#     samples = ds_train.select([0,1])
#     sample = samples[0]
#     ctx = sample["ctx"]
#     ctx_a = sample["ctx_a"]
#     endings = sample["endings"]
#     sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
#     llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",dtype="half",task="score",
#     enforce_eager=True)
#     scores = llm.score(ctx, endings)
#     print(scores)
#
#     # output = llm.generate(query_text, sampling_params)
#     # print(output[0].outputs[0])
#
#     # for output in outputs:
#     #     prompt = output.prompt
#     #     generated_text = output.outputs[0].text
#     #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")