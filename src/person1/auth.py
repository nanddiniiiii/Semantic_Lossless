"""Refresh existing local OAuth access before expiry, without exposing secrets."""
from datetime import datetime,timedelta,timezone
import os
from pathlib import Path


def ensure_fresh_authorization(username):
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kaggle_creds import KaggleCredentials
    from src.persistence import file_lock
    path=Path(os.path.expanduser(KaggleCredentials.DEFAULT_CREDENTIALS_FILE))
    if not path.is_file():return False
    with file_lock(path.with_suffix('.refresh.lock')) as owned:
        if not owned:return False
        with KaggleApi().build_kaggle_client() as client:
            credentials=KaggleCredentials.load(client)
            if not credentials:return False
            if credentials.get_username().lower()!=username.lower():
                raise RuntimeError('Cached OAuth account differs from the requested Kaggle owner')
            expiration=credentials._access_token_expiration
            # SDK2.2.5's expiry check waits until30 minutes after expiration.
            # Use its official refresh method proactively under a shared lock.
            if expiration is None or expiration<=datetime.now(timezone.utc)+timedelta(minutes=5):
                try:credentials.refresh_access_token()
                except Exception as error:
                    raise RuntimeError('Existing Kaggle OAuth authorization could not be refreshed') from error
                return True
    return False
