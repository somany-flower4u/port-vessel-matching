"""DACON code submission via the official dacon_submit_api client.
usage: submit.py check | submit.py submit ZIP MEMO   (token / team from DACON_TOKEN / DACON_TEAM)"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import requests
from dacon_submit_api import dacon_submit_api as api
CPT = "236771"
token, team = os.environ["DACON_TOKEN"], os.environ["DACON_TEAM"]
if sys.argv[1] == "check":
    r = requests.post("https://app.dacon.io/api/v1/code-submission/validate",
                      data={"cptId": CPT, "teamName": team, "apiToken": token}, timeout=15)
    body = r.json()
    print(r.status_code, {k: v for k, v in body.items() if k in ("quota", "message", "upload_filesize_limit")})
else:
    print(api.post_code_submission_file(sys.argv[2], token, CPT, team, sys.argv[3] if len(sys.argv) > 3 else ""))
