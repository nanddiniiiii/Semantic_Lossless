import json
import pytest
from src.person1.campaign import require_pilot_qualification


def test_completed_updates_cannot_bypass_failed_resume(tmp_path):
    (tmp_path/'pilot_passed.json').write_text('{}')
    with pytest.raises(RuntimeError,match='resume_probe'):
        require_pilot_qualification(tmp_path)
    (tmp_path/'resume_probe.json').write_text(json.dumps({'passed':False}))
    (tmp_path/'pilot_only.json').write_text(json.dumps({'status':'pilot_completed'}))
    with pytest.raises(RuntimeError,match='continuation'):
        require_pilot_qualification(tmp_path)
    (tmp_path/'resume_probe.json').write_text(json.dumps({'passed':True}))
    require_pilot_qualification(tmp_path)
