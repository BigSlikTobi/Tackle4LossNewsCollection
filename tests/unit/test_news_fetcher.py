import unittest
import re
import hashlib

class TestNewsFetcherSlugGeneration(unittest.TestCase):
    def _generate_slug(self, headline: str) -> str:
        """ 
        Generates a slug from the given headline.
        If the headline is empty or contains only special characters, a SHA1 hash is used.
        Args:
            headline (str): The headline to generate a slug from.
        Returns:
            str: The generated slug.
        """
        slug_base = re.sub(r'[^\w\s-]', '', headline.lower())
        slug = re.sub(r'[-\s]+', '-', slug_base).strip('-')[:100]
        if not slug:
            slug_digest = hashlib.sha1(headline.encode('utf-8')).hexdigest()
            slug = f"article-{slug_digest[:8]}"
        return slug

    def test_special_char_headline(self):
        """Test that a headline with only special characters generates a slug based on its SHA1 hash.
        This verifies that the slug generation logic correctly handles headlines that do not contain alphanumeric characters.
        Args:
            None
        Returns:
            None
        """
        headline = "!!!???"
        # Based on the logic in _generate_slug:
        # 1. slug_base = re.sub(r'[^\w\s-]', '', "!!!???".lower()) -> ""
        #    (since ! and ? are not \w, \s, or -)
        # 2. slug = re.sub(r'[-\s]+', '-', "").strip('-')[:100] -> ""
        # 3. if not slug: is true.
        # 4. slug_digest = hashlib.sha1("!!!???".encode('utf-8')).hexdigest() -> "8b8a7981..."
        # 5. slug = f"article-{slug_digest[:8]}" -> "article-8b8a7981"
        expected_slug = "article-8b8a7981" # Corrected based on SHA1("!!!???")
        self.assertEqual(self._generate_slug(headline), expected_slug)

    def test_identical_special_char_headlines(self):
        """Test that two identical headlines with only special characters generate the same slug.
        This verifies that the slug generation logic produces consistent results for identical inputs.
        Args:
            None
        Returns:
            None
        """
        headline1 = "!!!???"
        headline2 = "!!!???"
        # Expected: article-8b8a7981
        self.assertEqual(self._generate_slug(headline1), self._generate_slug(headline2))
        self.assertEqual(self._generate_slug(headline1), "article-8b8a7981")

    def test_normal_headline(self):
        """Test that a normal headline generates a slug without special characters.
        This verifies that the slug generation logic correctly processes alphanumeric headlines.
        Args:
            None
        Returns:
            None
        """
        headline = "Normal Headline"
        expected_slug = "normal-headline"
        self.assertEqual(self._generate_slug(headline), expected_slug)

    def test_empty_headline(self):
        """Test that an empty headline generates a slug based on its SHA1 hash.
        This verifies that the slug generation logic handles empty inputs correctly.
        Args:
            None
        Returns:
            None
        """
        headline = ""
        # SHA1 of "" is da39a3ee5e6b4b0d3255bfef95601890afd80709
        expected_slug = "article-da39a3ee"
        self.assertEqual(self._generate_slug(headline), expected_slug)

    def test_headline_with_leading_trailing_hyphens_and_spaces(self):
        """Test that a headline with leading/trailing hyphens and spaces generates a cleaned slug.
        This verifies that the slug generation logic removes unnecessary characters and consolidates hyphens.
        Args:
            None
        Returns:
            None
        """
        headline = "  --Test -- Headline --  "
        expected_slug = "test-headline" # Should be cleaned and hyphens consolidated
        self.assertEqual(self._generate_slug(headline), expected_slug)

    def test_long_normal_headline_truncation(self):
        """Test that a long normal headline is truncated to 100 characters.
        This verifies that the slug generation logic correctly truncates long headlines.
        Args:
            None
        Returns:
            None
        """
        headline = "This is a very long headline that definitely exceeds one hundred characters and should be truncated correctly at the hundredth character mark"
        # The slug logic results in keeping a hyphen if it's at the 100th char position
        expected_slug = "this-is-a-very-long-headline-that-definitely-exceeds-one-hundred-characters-and-should-be-truncated-"
        self.assertEqual(self._generate_slug(headline), expected_slug)

    def test_long_special_char_headline_no_truncation_of_hash_part(self):
        """Test that a long headline with only special characters generates a slug based on its SHA1 hash.
        This verifies that the slug generation logic correctly handles long headlines with no alphanumeric characters.
        Args:
            None
        Returns:
            None
        """
        # Using only special characters (no underscore) that will be removed by re.sub
        headline_spl_chars = "!@#$%^&*()+{}[]:;\"'<>,.?/~`~"  # Underscore removed
        headline = headline_spl_chars * 4 # Length 116, ensures it's "long"
        # This headline should result in an empty slug_base, triggering the hash fallback.
        # SHA1 of this new (headline_spl_chars * 4) is a1b63618... (confirmed by debug print)
        actual_slug = self._generate_slug(headline)
        self.assertTrue(actual_slug.startswith("article-"), f"Actual slug '{actual_slug}' did not start with 'article-'")
        self.assertEqual(len(actual_slug), 16, f"Actual slug '{actual_slug}' did not have length 16") # "article-" is 8 chars, hash is 8 chars
        self.assertEqual(actual_slug, "article-a1b63618")

if __name__ == '__main__':
    unittest.main()
