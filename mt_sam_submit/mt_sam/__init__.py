from .prompting import NoPromptCandidatesError

__all__ = ["MTSAMConfig", "MTSAMPredictor", "NoPromptCandidatesError"]


def __getattr__(name):
    if name in {"MTSAMConfig", "MTSAMPredictor"}:
        from .predictor import MTSAMConfig, MTSAMPredictor

        return {"MTSAMConfig": MTSAMConfig, "MTSAMPredictor": MTSAMPredictor}[name]
    raise AttributeError(name)
