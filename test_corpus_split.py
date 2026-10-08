"""Regression checks for original Lens ===SPLIT=== corpus boundaries."""
import tempfile
import unittest
from pathlib import Path

import main


class CorpusSplitTests(unittest.TestCase):
    def read_corpus(self, raw: bytes, min_length=2, max_examples=10000, max_length=1500):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'github-code.txt'
            file.write_bytes(raw)
            return main.load_corpus(str(file), min_length, max_examples, max_length)

    def test_multiline_chunk_and_boundary_newlines(self):
        corpus = (b'foo() {\n  return 1;\n}\n===SPLIT===\r\n'
                  b'class Foo:\n    pass\n\r\n===SPLIT===\n'
                  b'last\n\nline\n')
        self.assertEqual(self.read_corpus(corpus), [
            b'foo() {\n  return 1;\n}', b'class Foo:\n    pass', b'last\n\nline'])

    def test_inline_marker_and_empty_chunks(self):
        self.assertEqual(self.read_corpus(b'===SPLIT===a\nb===SPLIT=====' +
                                          b'=SPLIT===x===SPLIT===yy'),
                         [b'a\nb', b'yy'])

    def test_no_marker_means_one_document_not_lines(self):
        self.assertEqual(self.read_corpus(b'one\ntwo\nthree\n'), [b'one\ntwo\nthree'])

    def test_min_length_and_max_examples(self):
        self.assertEqual(self.read_corpus(b'a===SPLIT===ab===SPLIT===cde===SPLIT===efg',
                                          min_length=2, max_examples=2), [b'ab', b'cde'])

    def test_marker_spanning_io_boundary(self):
        # The marker starts at the final 4 bytes of the first 1 MiB block.
        text = b'A' * ((1 << 20) - 4) + b'===SPLIT===' + b'B\nB'
        chunks = self.read_corpus(text)
        self.assertEqual(chunks, [b'B\nB'])

    def test_byte_preservation_within_limit(self):
        original = bytes(range(256)) * 5
        self.assertEqual(self.read_corpus(original + b'===SPLIT===end'),
                         [original, b'end'])

    def test_long_chunks_are_skipped_not_truncated(self):
        raw = (b'A' * 1499 + b'===SPLIT===' + b'B' * 1500
               + b'===SPLIT===' + b'C' * 1501 + b'===SPLIT===fine')
        self.assertEqual(self.read_corpus(raw), [b'A' * 1499, b'fine'])

    def test_skip_long_chunk_before_counting_towards_corpus_chunks(self):
        raw = b'A' * 1800 + b'===SPLIT===hello===SPLIT===world'
        self.assertEqual(self.read_corpus(raw, max_examples=1), [b'hello'])

    def test_length_is_measured_after_boundary_newline_strip(self):
        self.assertEqual(self.read_corpus(b'\r\n' + b'A' * 1499 +
                                          b'\r\n===SPLIT===' + b'end'),
                         [b'A' * 1499, b'end'])

    def test_long_chunk_no_delimiter_not_split_by_newline(self):
        import contextlib
        with self.assertRaisesRegex(ValueError, 'no chunks'):
            self.read_corpus(b'code\n' * 400)

    def test_override_max_chunk_for_backwards_compatibility(self):
        original = bytes(range(256)) * 1000
        self.assertEqual(self.read_corpus(original + b'===SPLIT===end',
                                          max_length=len(original) + 1),
                         [original, b'end'])


if __name__ == '__main__':
    unittest.main()
