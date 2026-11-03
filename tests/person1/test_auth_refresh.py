from contextlib import nullcontext
from datetime import datetime,timedelta,timezone
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('minutes,expected',[(-1,True),(3,True),(60,False)])
def test_expiry_refreshes_existing_authorization_before_cli_reuses_stale_token(tmp_path,monkeypatch,minutes,expected):
    from src.person1.auth import ensure_fresh_authorization
    KaggleApi=pytest.importorskip('kaggle.api.kaggle_api_extended').KaggleApi
    from kagglesdk.kaggle_creds import KaggleCredentials
    path=tmp_path/'credentials.json';path.write_text('{}')
    monkeypatch.setattr(KaggleCredentials,'DEFAULT_CREDENTIALS_FILE',str(path))
    calls=[]
    credentials=SimpleNamespace(_access_token_expiration=datetime.now(timezone.utc)+timedelta(minutes=minutes),
        get_username=lambda:'hmmmkk',refresh_access_token=lambda:calls.append(1))
    monkeypatch.setattr(KaggleCredentials,'load',lambda client:credentials)
    monkeypatch.setattr(KaggleApi,'build_kaggle_client',lambda self:nullcontext(object()))
    assert ensure_fresh_authorization('hmmmkk')==expected
    assert bool(calls)==expected
    with pytest.raises(RuntimeError,match='account differs'):ensure_fresh_authorization('other-account')
