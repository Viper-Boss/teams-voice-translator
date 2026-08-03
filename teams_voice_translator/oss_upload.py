from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import alibabacloud_oss_v2 as oss


class OssUploadError(RuntimeError):
    pass


@dataclass(frozen=True)
class UploadedVoiceSample:
    key: str
    signed_url: str


class OssTemporaryUploader:
    def __init__(
        self,
        *,
        region: str,
        bucket: str,
        access_key_id: str,
        access_key_secret: str,
        expiry_seconds: int = 900,
        client=None,
    ) -> None:
        self.region = region.strip()
        self.bucket = bucket.strip()
        self.expiry_seconds = max(60, min(3600, int(expiry_seconds)))
        if not all((self.region, self.bucket, access_key_id.strip(), access_key_secret.strip())):
            raise OssUploadError("OSS 配置不完整，请填写地域、Bucket、AccessKey ID 和 Secret")

        if client is None:
            config = oss.config.load_default()
            config.credentials_provider = oss.credentials.StaticCredentialsProvider(
                access_key_id.strip(),
                access_key_secret.strip(),
            )
            config.region = self.region
            config.endpoint = f"https://oss-{self.region}.aliyuncs.com"
            client = oss.Client(config)
        self.client = client

    def upload(self, local_path: str | Path) -> UploadedVoiceSample:
        path = Path(local_path)
        key = f"teams-voice-translator/temporary/{uuid.uuid4().hex}.wav"
        try:
            self.client.put_object_from_file(
                oss.PutObjectRequest(
                    bucket=self.bucket,
                    key=key,
                    content_type="audio/wav",
                    forbid_overwrite=True,
                ),
                str(path),
            )
            result = self.client.presign(
                oss.GetObjectRequest(bucket=self.bucket, key=key),
                expires=timedelta(seconds=self.expiry_seconds),
            )
        except Exception as exc:
            try:
                self.delete(key)
            except Exception:
                pass
            raise OssUploadError(f"OSS 临时上传失败：{exc}") from exc

        signed_url = str(getattr(result, "url", "") or "").strip()
        if not signed_url.startswith("https://"):
            try:
                self.delete(key)
            except Exception:
                pass
            raise OssUploadError("OSS 没有生成可用的 HTTPS 临时地址，请检查地域和 Bucket")
        return UploadedVoiceSample(key=key, signed_url=signed_url)

    def delete(self, key: str) -> None:
        self.client.delete_object(oss.DeleteObjectRequest(bucket=self.bucket, key=key))
