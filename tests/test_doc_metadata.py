"""Unit tests for deterministic document metadata. All fixtures are synthetic."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from memory_store import doc_metadata as dm

GERMAN_CLEAN = (
    "Sehr geehrte Frau Beispiel,\n\n"
    "hiermit bestätigen wir den Eingang Ihrer Jahresabrechnung für die Wohnung in der "
    "Musterstraße. Die Nebenkosten wurden nach dem vereinbarten Schlüssel verteilt und der "
    "Betrag wird mit der nächsten Zahlung verrechnet. Bitte prüfen Sie die Unterlagen und "
    "melden Sie sich bei Fragen.\n\nMit freundlichen Grüßen\nHausverwaltung Muster GmbH"
)
ENGLISH_CLEAN = (
    "Dear Mr Example, thank you for your letter regarding the annual statement. We have "
    "reviewed the documents and the balance will be credited to your account with the next "
    "payment. Please contact us if you have any questions about this notice. Yours "
    "sincerely, Example Services Ltd."
)
RUSSIAN_CLEAN = (
    "Уважаемый господин Образцов, настоящим подтверждаем получение вашего письма по "
    "вопросу оплаты счёта за электроэнергию. Просим вас проверить документы."
)
FRENCH_CLEAN = (
    "Bonjour Madame, nous avons bien reçu votre lettre concernant la facture annuelle "
    "et nous vous répondrons dans les meilleurs délais."
)
# Synthetic garbled OCR: digit/letter confusion, symbol soup, stray fragments.
GARBLED_DE = (
    "Rcchnung Nr. 4S6 vom l2.O3.2O24 Zah1ung fä11ig bi5 l5.O4. ~~ ¦¦ ■■ Slr0m 1ll1 Iiil "
    ",,,, ;;;; Ja4resabr3chnung für d1e W0hnung ### x ~ ¬ ¿ q z"
)
SYMBOL_SOUP = "¦■ ~~ ¿¡ ^^ ``` ■□▪ �� ◊◊ ¶¶ †‡ ¦¦ |||| ,,,, ;;;;"


class FrontmatterTests(unittest.TestCase):
    def test_split(self):
        fields, body = dm.split_frontmatter('---\ntitle: "A B"\nDate: 2024-01-02\n---\nBody')
        self.assertEqual(fields, {"title": "A B", "date": "2024-01-02"})
        self.assertEqual(body, "Body")

    def test_ignores_nested_and_comment_lines_and_missing_block(self):
        fields, _ = dm.split_frontmatter("---\n# note\n  nested: x\n- item\nkey: v\n---\nx")
        self.assertEqual(fields, {"key": "v"})
        self.assertEqual(dm.split_frontmatter("no frontmatter"), ({}, "no frontmatter"))
        self.assertEqual(
            dm.split_frontmatter("---\nunterminated: yes\n"), ({}, "---\nunterminated: yes\n")
        )


class DateTests(unittest.TestCase):
    def date(self, content, rel="folder/doc.md", **kwargs):
        return dm.extract_doc_date(content, rel, **kwargs)

    def test_frontmatter_wins_and_accepts_timestamps(self):
        content = "---\ndate: 2021-05-06T10:00:00Z\n---\nDatum: 01.01.2020"
        self.assertEqual(self.date(content, "2019-03-04_x.md"), "2021-05-06")

    def test_frontmatter_key_priority_and_custom_keys(self):
        content = "---\ndate: 2020-01-01\ndocument_date: 12.03.2019\n---\nx"
        self.assertEqual(self.date(content), "2019-03-12")
        self.assertIsNone(self.date("---\ncreated: 2022-02-02\n---\nx"))
        self.assertEqual(
            self.date("---\ncreated: 2022-02-02\n---\nx", frontmatter_keys=("created",)),
            "2022-02-02",
        )

    def test_unparsable_frontmatter_date_falls_through(self):
        self.assertEqual(self.date("---\ndate: sometime\n---\nDatum: 3.2.2020"), "2020-02-03")

    def test_german_labelled_formats(self):
        self.assertEqual(self.date("# Titel\nDatum: 12. März 2024\ntext"), "2024-03-12")
        self.assertEqual(self.date("Datum 05.11.2023"), "2023-11-05")
        self.assertEqual(self.date("| Rechnungsdatum | 05.11.2023 |"), "2023-11-05")
        self.assertEqual(self.date("**Ausstellungsdatum:** 1. Dezember 2022"), "2022-12-01")
        self.assertEqual(self.date("Datum: 3. Maerz 2021"), "2021-03-03")

    def test_english_labelled_formats(self):
        self.assertEqual(self.date("Invoice date: March 3rd, 2022"), "2022-03-03")
        self.assertEqual(self.date("Date: 14 Sep 2020"), "2020-09-14")
        self.assertEqual(self.date("Issue date: 2020-02-29"), "2020-02-29")
        self.assertEqual(self.date("Dated: 2nd of May 2019"), "2019-05-02")

    def test_letter_date_line_needs_den_or_month_name(self):
        self.assertEqual(self.date("Musterstadt, den 3.2.2020\nHallo"), "2020-02-03")
        self.assertEqual(self.date("Musterstadt, 5. Dezember 2022\nHallo"), "2022-12-05")
        self.assertIsNone(self.date("Mustermann, 15.03.1985\nHallo"))

    def test_filename_patterns(self):
        self.assertEqual(self.date("x", "a/2019-03-04_rechnung.md"), "2019-03-04")
        self.assertEqual(self.date("x", "a/scan_20180711.md"), "2018-07-11")
        self.assertEqual(self.date("x", "a/rechnung_12.03.2017.md"), "2017-03-12")
        self.assertIsNone(self.date("x", "a/rechnung_99.99.2017.md"))
        self.assertIsNone(self.date("x", "a/inv_20241301.md"))

    def test_content_outranks_filename(self):
        self.assertEqual(self.date("Datum: 2020-06-01", "a/2019-03-04_x.md"), "2020-06-01")

    def test_never_guesses(self):
        for content in (
            "Geburtsdatum: 12.03.1980",
            "Due date: 01.01.2030",
            "Datum der Geburt: 12.03.1980",
            "Der Termin ist am 12.03.2024 um 10 Uhr.",
            "Datum: 31.02.2024",
            "Datum: 1.1.1800",
            "Datum: 12/03/2024",
            "",
        ):
            self.assertIsNone(self.date(content), content)

    def test_only_top_of_document_is_scanned(self):
        content = "\n".join(["zeile"] * 80 + ["Datum: 12.03.2024"])
        self.assertIsNone(self.date(content))

    def test_returns_none_for_empty_label(self):
        self.assertIsNone(self.date("Datum:\nnichts"))


class TitleTests(unittest.TestCase):
    def test_sources_in_order(self):
        self.assertEqual(dm.extract_title("---\ntitle: A | B\n---\n# Ignored", "x.md"), "A B")
        self.assertEqual(dm.extract_title("text\n## Heading Foo ##\n", "a/f.md"), "Heading Foo")
        self.assertEqual(dm.extract_title("nothing", "a/my_file-name.md"), "my file name")

    def test_length_is_capped(self):
        title = dm.extract_title("# " + "word " * 60, "x.md")
        self.assertLessEqual(len(title), 80)
        self.assertFalse(title.endswith(" "))


class CategoryTests(unittest.TestCase):
    def test_path_segments(self):
        cases = {
            "personal/versicherung/haftpflicht-2021.md": "insurance",
            "shared/kfz/tuev-bericht.md": "vehicle",
            "personal/gesundheit/befund.md": "health",
            "personal/steuer/steuerbescheid.md": "finance",
            "work/arbeitsvertrag.md": "work",
            "personal/wohnung/mietvertrag.md": "home",
            "personal/kita/anmeldung.md": "family",
            "personal/familienversicherung.md": "insurance",
            "misc/notes.md": "other",
        }
        for path, expected in cases.items():
            self.assertEqual(dm.classify_category(path, "x"), expected, path)

    def test_path_beats_body(self):
        self.assertEqual(
            dm.classify_category("shared/kfz/x.md", "Kinder Kita Schule Eltern"), "vehicle"
        )

    def test_body_needs_two_distinct_keywords(self):
        self.assertEqual(dm.classify_category("scan/001.md", "Die Wohnung ist schön."), "other")
        self.assertEqual(dm.classify_category("scan/001.md", GERMAN_CLEAN), "home")

    def test_frontmatter_is_not_scanned_and_ties_follow_precedence(self):
        content = "---\ntags: versicherung police\n---\nnichts"
        self.assertEqual(dm.classify_category("scan/001.md", content), "other")
        self.assertEqual(dm.classify_category("a/versicherung-kfz.md", "x"), "insurance")

    def test_compound_words_match_long_keywords_only(self):
        self.assertEqual(dm.classify_category("a/steuererklaerung-2020.md", "x"), "finance")
        self.assertEqual(dm.classify_category("a/network.md", "x"), "other")


class LanguageTests(unittest.TestCase):
    def test_languages(self):
        self.assertEqual(dm.detect_language(GERMAN_CLEAN), "de")
        self.assertEqual(dm.detect_language(ENGLISH_CLEAN), "en")
        self.assertEqual(dm.detect_language(RUSSIAN_CLEAN), "ru")
        self.assertEqual(dm.detect_language(FRENCH_CLEAN), "other")

    def test_degenerate_inputs_are_other(self):
        for text in ("", "12345 6789", "| --- | --- |", "alpha beta gamma delta"):
            self.assertEqual(dm.detect_language(text), "other", text)

    def test_stopword_sets_do_not_overlap(self):
        self.assertFalse(dm._STOPWORDS_DE & dm._STOPWORDS_EN)

    def test_tie_is_other(self):
        self.assertEqual(dm.detect_language("der die das the and of"), "other")


class OcrQualityTests(unittest.TestCase):
    def test_clean_text_scores_high_in_german_and_english(self):
        self.assertGreaterEqual(dm.ocr_quality(GERMAN_CLEAN), 0.9)
        self.assertGreaterEqual(dm.ocr_quality(ENGLISH_CLEAN), 0.9)
        self.assertGreaterEqual(dm.ocr_quality(RUSSIAN_CLEAN), 0.9)
        self.assertGreaterEqual(dm.ocr_quality(FRENCH_CLEAN), 0.9)

    def test_garbled_text_scores_low(self):
        self.assertLess(dm.ocr_quality(GARBLED_DE), 0.4)
        self.assertEqual(dm.ocr_quality(SYMBOL_SOUP), 0.0)

    def test_garbled_scores_below_clean_and_stays_in_range(self):
        for text in (GERMAN_CLEAN, GARBLED_DE, SYMBOL_SOUP, "x", "a b c"):
            self.assertTrue(0.0 <= dm.ocr_quality(text) <= 1.0)
        self.assertLess(dm.ocr_quality(GARBLED_DE), dm.ocr_quality(GERMAN_CLEAN))

    def test_garbage_characters(self):
        for char in ("\u25c6", "\x00", "\ufffd", "\u00a6"):
            self.assertTrue(dm._is_garbage_char(char), repr(char))
        for char in ("a", "\u00e4", ".", "\u20ac", "\u00a9", "\u00b0", "\u0416"):
            self.assertFalse(dm._is_garbage_char(char), repr(char))

    def test_no_content_is_unknown(self):
        self.assertIsNone(dm.ocr_quality(""))
        self.assertIsNone(dm.ocr_quality("  \n\t "))
        self.assertIsNone(dm.ocr_quality("| --- | --- |\n|---|---|"))

    def test_numbers_and_codes_are_plausible(self):
        self.assertEqual(dm.ocr_quality("IBAN DE12 3456 7890 A4 12kg 100m"), 1.0)

    def test_token_shapes(self):
        plausible = ("Nr", "Str", "Musterstraße", "Schlechtsprechen", "GmbH", "iPhone", "I", "a")
        implausible = ("lllll", "Iil", "aaaa", "bcdfghjk", "aBcDe", "xx", "b", "q" * 41, "ll")
        for word in plausible:
            self.assertTrue(dm._plausible_alpha(word), word)
        for word in implausible:
            self.assertFalse(dm._plausible_alpha(word), word)
        for token in ("2024", "A4", "12kg", "x"):
            self.assertIn(dm._plausible_token(token), (True, False))
        self.assertTrue(dm._plausible_token("A4"))
        self.assertFalse(dm._plausible_token("Zah1ung"))
        self.assertFalse(dm._plausible_token("2O24"))

    def test_stopword_density_lowers_long_wordlists(self):
        wordlist = " ".join(["Muster", "Beispiel", "Kabel", "Lampe", "Tisch", "Stuhl"] * 8)
        self.assertLess(dm.ocr_quality(wordlist), dm.ocr_quality(GERMAN_CLEAN))


class OrchestrationTests(unittest.TestCase):
    def test_extract_document_metadata_and_header(self):
        content = "---\ndate: 2023-02-01\n---\n# Rechnung Strom\nStrom Wasser Gas Nebenkosten. Zahlung der Rechnung."
        meta = dm.extract_document_metadata(content, "personal/x.md")
        self.assertEqual(meta.title, "Rechnung Strom")
        self.assertEqual(meta.doc_date, "2023-02-01")
        self.assertEqual(meta.category, "home")
        self.assertIn(meta.language, dm.LANGUAGES)
        self.assertEqual(dm.context_header(meta), "Rechnung Strom | 2023-02-01 | home")

    def test_header_omits_unknown_parts(self):
        meta = dm.DocumentMetadata("Notizen", None, "other", "de")
        self.assertEqual(dm.context_header(meta), "Notizen")
        meta = dm.DocumentMetadata("", None, "other", "de")
        self.assertEqual(dm.context_header(meta), "")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
