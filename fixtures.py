from systems.baseline_example import ExampleBaselineSystem
from systems.mixture_agent_example import ExamplemixtureAgentSystem
from systems.reflection_example import ExampleReflectionSystem


def system_selector(sut: str = None, verbose=False):
    # keep this for system level arguments
    kwargs = {"verbose": verbose}

    systems = {
        "baseline-gpt-4o-mini": ExampleBaselineSystem("gpt-4o-mini", **kwargs),
        "baseline-gpt-4o": ExampleBaselineSystem("gpt-4o", **kwargs),
        "baseline-gemma-2b-it": ExampleBaselineSystem("google/gemma-2b-it", **kwargs),
        "baseline-llama-2-13b-chat-hf": ExampleBaselineSystem(
            "meta-llama/Llama-2-13b-chat-hf", **kwargs
        ),
        "baseline-deepseek-r1-distill-qwen-14b": ExampleBaselineSystem(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", **kwargs
        ),
        "reflection-gpt-4o-mini": ExampleReflectionSystem(
            {"executor": "gpt-4o-mini", "reflector": "gpt-4o-mini"}, **kwargs
        ),
        "reflection-gpt-4o": ExampleReflectionSystem(
            {"executor": "gpt-4o", "reflector": "gpt-4o"}, **kwargs
        ),
        "mixture-agent-gpt-4o-mini": ExamplemixtureAgentSystem(
            {"suggesters": ["gpt-4o-mini", "gpt-4o-mini"], "merger": "gpt-4o-mini"},
            **kwargs,
        ),
        "mixture-agent-gpt-4o": ExamplemixtureAgentSystem(
            {"suggesters": ["gpt-4o-mini", "gpt-4o-mini"], "merger": "gpt-4o"}, **kwargs
        ),
    }

    return systems[sut]