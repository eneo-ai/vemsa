from types import SimpleNamespace

import pytest

from tolka.pipeline.transcribe import EasyTranscriberEngine, words_from_alignments


def test_words_from_nested_segments():
    segments = [
        SimpleNamespace(
            words=[
                SimpleNamespace(word=" hej ", start=0.0, end=0.4),
                SimpleNamespace(word="då", start=0.5, end=0.8),
            ]
        ),
        SimpleNamespace(words=[SimpleNamespace(word="sen", start=1.0, end=1.2)]),
    ]
    words = words_from_alignments(segments)
    assert [w.word for w in words] == ["hej", "då", "sen"]
    assert words[0].start == 0.0


def test_words_from_flat_word_segments_with_text_field():
    segments = [
        SimpleNamespace(text="andra", start=1.0, end=1.5),
        SimpleNamespace(text="första", start=0.0, end=0.5),
    ]
    words = words_from_alignments(segments)
    # sorted by start regardless of input order
    assert [w.word for w in words] == ["första", "andra"]


def test_unrecognized_shape_raises():
    with pytest.raises(ValueError, match="unrecognized alignment shape"):
        words_from_alignments([SimpleNamespace(banana=1)])


def test_engine_module_imports_without_ml_stack(settings):
    # constructing the engine must not import torch/easytranscriber
    engine = EasyTranscriberEngine(settings)
    assert engine is not None
