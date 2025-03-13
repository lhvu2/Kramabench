from systems.baseline_example import ExampleBaselineSystem
from systems.reflection_example import ExampleReflectionSystem
from systems.mixtrue_agent_example import ExampleMixtrueAgentSystem

def system_selector(sut: str = None):
    if sut == "baseline-gpt-4o-mini":
        return ExampleBaselineSystem("gpt-4o-mini")
    elif sut == "baseline-gpt-4o":
        return ExampleBaselineSystem("gpt-4o")
    elif sut == "baseline-gemma-2b-it":
        return ExampleBaselineSystem("google/gemma-2b-it")
    elif sut == "baseline-llama-2-13b-chat-hf":
        return ExampleBaselineSystem("meta-llama/Llama-2-13b-chat-hf")
    elif sut == "baseline-deepseek-r1-distill-qwen-14b":
        return ExampleBaselineSystem("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    elif sut == "reflection-gpt-4o-mini":
        return ExampleReflectionSystem({"executor": "gpt-4o-mini", "reflector": "gpt-4o-mini"})
    elif sut == "reflection-gpt-4o":
        return ExampleReflectionSystem({"executor": "gpt-4o", "reflector": "gpt-4o"})
    elif sut == "mixtrue-agent-gpt-4o-mini":
        return ExampleMixtrueAgentSystem({"suggesters": ["gpt-4o-mini", "gpt-4o-mini"], "merger": "gpt-4o-mini"})
    elif sut == "mixtrue-agent-gpt-4o":
        return ExampleMixtrueAgentSystem({"suggesters": ["gpt-4o-mini", "gpt-4o-mini"], "merger": "gpt-4o"})
    else:
        raise ValueError(f"System {sut} not found")
