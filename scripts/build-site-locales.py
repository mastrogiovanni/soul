#!/usr/bin/env python3
"""Render the default Italian page from the English layout using only stdlib.

The committed output works on GitHub Pages without a build step or JavaScript.
Fail on missing translations so new English copy cannot silently leak into Italian.
"""

import argparse
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://mastrogiovanni.github.io/soul-community/"
HUMAN_ATTRIBUTES = {"alt", "aria-label", "placeholder", "data-copy-message"}
ATTRIBUTE = re.compile(r'([\w:-]+)="([^"]*)"')
CODE_TRANSLATIONS = {
    "# Set the inference key supplied by the operator.": "# Imposta la chiave di inferenza fornita dal gestore.",
    "# Discover prepared, available models.": "# Scopri i modelli preparati e disponibili.",
    "# Replace MODEL_ID with an ID from that list.": "# Sostituisci MODEL_ID con un ID presente nella lista.",
    "Summarize these volunteer notes: Food collection moved to Friday. Maya will confirm drivers.":
        "Riassumi queste note dei volontari: La raccolta alimentare è spostata a venerdì. Maya confermerà gli autisti.",
}


class ItalianPage(HTMLParser):
    def __init__(self, translations):
        super().__init__(convert_charrefs=True)
        self.translations = translations
        self.parts = []
        self.code_depth = 0

    def translate(self, value):
        key = " ".join(value.split())
        if not any(character.isalpha() for character in key):
            return value
        if key not in self.translations:
            raise ValueError(f"Missing Italian translation: {key}")
        prefix = value[:len(value) - len(value.lstrip())]
        suffix = value[len(value.rstrip()):]
        return prefix + self.translations[key] + suffix

    def handle_starttag(self, tag, attributes):
        values = dict(attributes)
        raw = self.get_starttag_text()

        def translate_attribute(match):
            name, value = match.groups()
            if name in HUMAN_ATTRIBUTES or (
                tag == "meta" and name == "content" and (
                    values.get("name") == "description" or
                    values.get("property") in {"og:title", "og:description"}
                )
            ):
                value = html.escape(self.translate(html.unescape(value)), quote=True)
            if name in {"src", "href"} and value.startswith("../") and len(value) > 3:
                value = value[3:]
            if (values.get("rel") == "canonical" or values.get("property") == "og:url") and value == BASE_URL + "en/":
                value = BASE_URL
            return f'{name}="{value}"'

        raw = ATTRIBUTE.sub(translate_attribute, raw)
        if tag == "html":
            raw = raw.replace('lang="en"', 'lang="it"')
        if values.get("property") == "og:locale":
            raw = raw.replace('content="en_US"', 'content="it_IT"')
        if values.get("property") == "og:locale:alternate":
            raw = raw.replace('content="it_IT"', 'content="en_US"')
        if "data-language-link" in values:
            if values["lang"] == "it":
                raw = raw.replace('href="../"', 'href="./"').replace('>', ' aria-current="page">')
            else:
                raw = raw.replace('href="./"', 'href="en/"').replace(' aria-current="page"', '')
        self.parts.append(raw)
        if tag in {"code", "script", "style"}:
            self.code_depth += 1

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        self.parts.append(f"</{tag}>")
        if tag in {"code", "script", "style"}:
            self.code_depth -= 1

    def handle_data(self, data):
        if self.code_depth:
            for source, translation in CODE_TRANSLATIONS.items():
                data = data.replace(source, translation)
        else:
            data = self.translate(data)
        self.parts.append(html.escape(data, quote=False))

    def handle_decl(self, decl):
        self.parts.append(f"<!{decl}>")

    def handle_comment(self, data):
        self.parts.append(f"<!--{data}-->")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if committed Italian HTML is stale")
    options = parser.parse_args()
    translations = json.loads((ROOT / "website/locales/it.json").read_text())
    page = ItalianPage(translations)
    page.feed((ROOT / "site/en/index.html").read_text())
    page.close()
    output = "".join(page.parts)
    destination = ROOT / "site/index.html"
    if options.check:
        if destination.read_text() != output:
            raise SystemExit("Italian HTML is stale. Run: python3 scripts/build-site-locales.py")
        print("Italian HTML matches the English layout and translation catalog.")
    else:
        destination.write_text(output)
        print("Generated site/index.html (Italian).")


if __name__ == "__main__":
    main()
