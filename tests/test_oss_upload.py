import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from teams_voice_translator.oss_upload import OssTemporaryUploader, OssUploadError


class FakeOssClient:
    def __init__(self, url="https://voice-private.oss-cn-beijing.aliyuncs.com/file.wav?signature=x"):
        self.url = url
        self.uploads = []
        self.presigns = []
        self.deletes = []

    def put_object_from_file(self, request, filepath):
        self.uploads.append((request, filepath))

    def presign(self, request, **kwargs):
        self.presigns.append((request, kwargs))
        return SimpleNamespace(url=self.url)

    def delete_object(self, request):
        self.deletes.append(request)


class OssUploadTests(unittest.TestCase):
    def make_uploader(self, client):
        return OssTemporaryUploader(
            region="cn-beijing",
            bucket="voice-private",
            access_key_id="id",
            access_key_secret="secret",
            client=client,
        )

    def test_upload_sign_and_delete_temporary_sample(self):
        client = FakeOssClient()
        uploader = self.make_uploader(client)
        with tempfile.TemporaryDirectory() as temp:
            sample = Path(temp) / "sample.wav"
            sample.write_bytes(b"RIFF-test")
            uploaded = uploader.upload(sample)

        self.assertTrue(uploaded.key.startswith("teams-voice-translator/temporary/"))
        self.assertTrue(uploaded.key.endswith(".wav"))
        self.assertTrue(uploaded.signed_url.startswith("https://"))
        self.assertEqual(client.uploads[0][0].content_type, "audio/wav")
        self.assertEqual(client.presigns[0][0].key, uploaded.key)
        uploader.delete(uploaded.key)
        self.assertEqual(client.deletes[0].key, uploaded.key)

    def test_non_https_presign_is_rejected_and_object_is_deleted(self):
        client = FakeOssClient(url="http://insecure.example/sample.wav")
        uploader = self.make_uploader(client)
        with tempfile.TemporaryDirectory() as temp:
            sample = Path(temp) / "sample.wav"
            sample.write_bytes(b"RIFF-test")
            with self.assertRaises(OssUploadError):
                uploader.upload(sample)
        self.assertEqual(len(client.deletes), 1)


if __name__ == "__main__":
    unittest.main()
