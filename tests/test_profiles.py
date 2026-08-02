import tempfile
import unittest
from pathlib import Path

from teams_voice_translator.profiles import CourseProfileStore


class CourseProfileTests(unittest.TestCase):
    def test_profile_round_trip_and_delete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = CourseProfileStore(root)
            store.save(
                "有限元课程",
                {
                    "translation_domain": "Finite element analysis",
                    "translation_terms": '[{"source":"网格","target":"mesh"}]',
                    "translation_style": "academic",
                    "voice": "myvoice",
                },
            )
            reloaded = CourseProfileStore(root)
            profile = reloaded.get("有限元课程")
            self.assertIsNotNone(profile)
            self.assertEqual(profile["translation_style"], "academic")
            self.assertEqual(profile["voice"], "myvoice")
            self.assertTrue(reloaded.delete("有限元课程"))
            self.assertEqual(reloaded.names(), [])


if __name__ == "__main__":
    unittest.main()
