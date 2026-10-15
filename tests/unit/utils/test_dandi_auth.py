import os

import pytest

from guppy.utils import dandi_auth
from guppy.utils.dandi_auth import (
    dandi_api_key_available,
    require_dandi_api_key,
    use_dandi_api_key,
)


@pytest.fixture
def keyring_key(monkeypatch):
    """Stand in for the user's keyring, which tests must not read; holds no key until set."""
    stored = {"key": None}
    monkeypatch.setattr(dandi_auth, "keyring_lookup", lambda service, username: (None, stored[username]))
    monkeypatch.delenv("DANDI_API_KEY", raising=False)
    return stored


class TestDandiApiKeyAvailable:
    def test_none_without_a_variable_or_keyring_entry(self, keyring_key):
        assert dandi_api_key_available() is False

    def test_the_environment_variable_counts(self, keyring_key, monkeypatch):
        monkeypatch.setenv("DANDI_API_KEY", "abc123")

        assert dandi_api_key_available() is True

    def test_a_key_stored_by_dandi_login_counts(self, keyring_key):
        keyring_key["key"] = "abc123"

        assert dandi_api_key_available() is True


class TestUseDandiApiKey:
    def test_sets_the_variable_dandi_reads(self, keyring_key):
        use_dandi_api_key(api_key="abc123")

        assert os.environ["DANDI_API_KEY"] == "abc123"


class TestRequireDandiApiKey:
    def test_refuses_without_a_key(self, keyring_key):
        with pytest.raises(ValueError) as excinfo:
            require_dandi_api_key()

        assert "paste it into the DANDI API key field at the top of the DANDI panel" in str(excinfo.value)

    def test_allows_streaming_with_a_key(self, keyring_key, monkeypatch):
        monkeypatch.setenv("DANDI_API_KEY", "abc123")

        require_dandi_api_key()
