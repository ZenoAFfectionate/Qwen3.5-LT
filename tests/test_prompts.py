from __future__ import annotations

from prompts.templates import SYSTEM_PROMPT, build_user_prompt, load_class_names


def test_system_prompt_contains_json_instruction() -> None:
    assert "JSON" in SYSTEM_PROMPT
    assert "class" in SYSTEM_PROMPT
    assert "confidence" in SYSTEM_PROMPT


def test_build_user_prompt_includes_all_classes() -> None:
    classes = ["cat", "dog", "horse"]
    prompt = build_user_prompt(classes, inject_all=True)
    for name in classes:
        assert name in prompt
    assert "3" in prompt


def test_build_user_prompt_without_class_list() -> None:
    prompt = build_user_prompt(["zebra_xyz", "koala_abc"], inject_all=False)
    assert "zebra_xyz" not in prompt and "koala_abc" not in prompt
    assert "ImageNet" in prompt or "category" in prompt


def test_load_class_names_from_text_file(tmp_path) -> None:
    f = tmp_path / "cls.txt"
    f.write_text("cat\ndog\nhorse\n")
    names = load_class_names(f)
    assert names == ["cat", "dog", "horse"]
