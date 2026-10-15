"""Where GuPPy finds the DANDI API key that streaming an asset needs."""

import os

from dandi.keyring_utils import keyring_lookup

DANDI_API_KEY_VARIABLE = "DANDI_API_KEY"

# The keyring entry ``dandi login`` stores the key under, for the production archive.
_KEYRING_SERVICE = "dandi-api-dandi"


def dandi_api_key_available() -> bool:
    """Whether a DANDI API key is available without asking for one.

    Looks where the ``dandi`` library does, the ``DANDI_API_KEY`` environment variable and then
    the keyring ``dandi login`` writes to, without falling back to its terminal prompt.

    Returns
    -------
    bool
        True when a key was found.
    """
    if os.environ.get(DANDI_API_KEY_VARIABLE):
        return True
    _keyring_backend, api_key = keyring_lookup(_KEYRING_SERVICE, "key")
    return api_key is not None


def use_dandi_api_key(*, api_key: str) -> None:
    """Use ``api_key`` for every DANDI request this GuPPy process makes; it is not saved.

    Parameters
    ----------
    api_key : str
        The key from the user's account page on dandiarchive.org.
    """
    os.environ[DANDI_API_KEY_VARIABLE] = api_key


def require_dandi_api_key() -> None:
    """Refuse to stream from DANDI without an API key.

    Raises
    ------
    ValueError
        If no key is available, naming where to enter one.
    """
    if not dandi_api_key_available():
        raise ValueError(
            "Streaming from the DANDI Archive needs a DANDI API key, even for a public dandiset. "
            "Copy yours from your account page on dandiarchive.org and paste it into the DANDI API key "
            "field at the top of the DANDI panel, then try again."
        )
