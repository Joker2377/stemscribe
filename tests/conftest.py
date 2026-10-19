"""Skip tests whose stage venv is not the one running pytest (each stage has its own venv)."""
import importlib.util

# ponytail: torch only lives in the model venvs; the tools venv / CI skip the training test
collect_ignore = [] if importlib.util.find_spec("torch") else ["test_train_guitar_split.py"]
