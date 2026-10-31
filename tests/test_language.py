"""The whole pack is English only: node names, tooltips, messages, templates, notes and docs."""
import glob
import json
import os
import re

import pytest

PACK = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
NON_ENGLISH = re.compile("[áéíóúñ¿¡ÁÉÍÓÚÑ]")
SPANISH_WORDS = re.compile(r"\b(los|las|del|para|con|sin|una|antes|después|niebla|prueba|empieza|crea|pero|cómo)\b", re.I)


def _strings(node):
    for key in ("title", "label", "name"):
        if isinstance(node.get(key), str):
            yield node[key]
    props = node.get("properties") or {}
    if isinstance(props.get("text"), str):
        yield props["text"]
    if node.get("type") in ("Note", "MarkdownNote"):
        for v in node.get("widgets_values") or []:
            if isinstance(v, str):
                yield v


def _templates():
    return sorted(glob.glob(os.path.join(PACK, "example_workflows", "**", "*.json"), recursive=True))


def test_template_notes_titles_and_groups_have_no_non_english_characters():
    for path in _templates():
        data = json.load(open(path, encoding="utf-8"))
        texts = [t for n in data["nodes"] for t in _strings(n)] + [g["title"] for g in data.get("groups", [])]
        for t in texts:
            assert not NON_ENGLISH.search(t), (os.path.basename(path), t[:60])
            assert not SPANISH_WORDS.search(t), (os.path.basename(path), t[:60])


def test_display_names_and_tooltips_are_english(helm):
    for cls, name in helm.nodes.NODE_DISPLAY_NAME_MAPPINGS.items():
        assert not NON_ENGLISH.search(name), cls
    for cls in helm.nodes.NODE_CLASS_MAPPINGS.values():
        spec = cls.INPUT_TYPES()
        for section in spec.values():
            for name, cfg in section.items():
                tip = cfg[1].get("tooltip", "") if len(cfg) > 1 else ""
                assert not NON_ENGLISH.search(tip), (cls.__name__, name)
                assert not SPANISH_WORDS.search(tip), (cls.__name__, name)


@pytest.mark.parametrize("rel", ["nodes.py", "__init__.py", "web/helm_axes.js", "README.md", "CHANGELOG.md",
                                 "docs/AXES.md", "docs/UX-NOTES.md", "docs/SPEC-0.5.md", "helm_core.py", "web/docs/HelmDial.md",
                                 "web/docs/HelmMakeAxis.md", "docs/EXPERIMENTS.md", "study/DATASET.md",
                                 "study/CONTENT-CALENDAR.md", "axes_catalog.json", "calibration.json"])
def test_messages_docs_and_catalog_are_english(rel):
    text = open(os.path.join(PACK, rel), encoding="utf-8").read()
    assert not NON_ENGLISH.search(text), rel
    assert not SPANISH_WORDS.search(text), rel
