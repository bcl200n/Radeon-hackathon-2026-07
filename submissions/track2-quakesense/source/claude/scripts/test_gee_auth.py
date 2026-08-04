#!/usr/bin/env python3
"""One-off smoke test: confirm the GEE service-account credential the user
placed at /workspace/keys/gee_service_account.json (permissions 600,
root-owned) actually authenticates, without this script -- or whoever runs
it -- ever printing the key file's contents. Only a success/failure summary
and a trivial, non-sensitive query result are printed.
"""

from __future__ import annotations

import json
from pathlib import Path

import ee
from google.oauth2 import service_account

KEY_PATH = Path("/workspace/keys/gee_service_account.json")
SCOPES = ["https://www.googleapis.com/auth/earthengine"]


def main() -> None:
    if not KEY_PATH.exists():
        raise SystemExit(f"key file not found at {KEY_PATH}")

    # google-auth reads the service-account email/project straight out of
    # the JSON file itself -- this script never needs to (and does not)
    # print any field from it.
    credentials = service_account.Credentials.from_service_account_file(
        str(KEY_PATH), scopes=SCOPES
    )
    ee.Initialize(credentials, project=json.loads(KEY_PATH.read_text())["project_id"])

    print("[1/2] ee.Initialize() succeeded")

    # Trivial, non-sensitive query: count features in a tiny public
    # FeatureCollection, just to prove the authenticated session can
    # actually talk to the Earth Engine backend, not just construct a
    # credentials object locally.
    fc = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq("country_na", "Italy"))
    count = fc.size().getInfo()
    print(f"[2/2] live query ok: {count} feature(s) returned for a trivial test filter")


if __name__ == "__main__":
    main()
