import pytest

from vllm import LLM


@pytest.fixture(autouse=True)
def v1(run_with_both_engines):
    # Simple autouse wrapper to run both engines for each test
    # This can be promoted up to conftest.py to run for every
    # test in a package
    pass


@pytest.mark.parametrize("enforce_eager", [False, True])
def test_empty_prompt(enforce_eager):
    llm = LLM(model="gpt2", enforce_eager=enforce_eager)
    with pytest.raises(ValueError, match='Prompt cannot be empty'):
        llm.generate([""])


@pytest.mark.skip_v1
@pytest.mark.parametrize("enforce_eager", [False, True])
def test_out_of_vocab_token(enforce_eager):
    llm = LLM(model="gpt2", enforce_eager=enforce_eager)
    with pytest.raises(ValueError, match='out of vocabulary'):
        llm.generate({"prompt_token_ids": [999999]})
