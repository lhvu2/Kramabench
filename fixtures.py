from benchmark.metrics import *
from systems.baseline_example import ExampleBaselineSystem


class SUTFactory:
    @classmethod
    def __init__(cls, sut: str = None):
        if sut == "baseline-gpt-4o-mini":
            return ExampleBaselineSystem("gpt-4o-mini")
        elif sut == "baseline-gpt-4o":
            return ExampleBaselineSystem("gpt-4o")
        elif sut == "baseline-mixtral":
            return ExampleBaselineSystem("mixtral")
        elif sut == "baseline-llama3":
            return ExampleBaselineSystem("llama3")
        else:
            raise ValueError(f"System {sut} not found")


class MetricFactory:
    @classmethod
    def __init__(cls, metric: str = None):
        if metric == "Precision":
            return Precision()
        elif metric == "Recall":
            return Recall()
        elif metric == "F1":
            return F1()
        elif metric == "BLEU":
            return BleuScore()
        elif metric == "ROUGE":
            return RougeScore()
        elif metric == "Success":
            return Success()
        else:
            raise ValueError(f"Metric {metric} not found")
