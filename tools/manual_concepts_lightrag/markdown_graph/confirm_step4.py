#!/usr/bin/env python3
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
from common import load_json, write_json
def main():
 p=argparse.ArgumentParser(); p.add_argument('--confirmation',required=True,type=Path); a=p.parse_args(); path=a.confirmation.resolve(); data=load_json(path) if path.is_file() else {}; data.update({'confirmed':True,'confirmed_at':datetime.now(timezone.utc).isoformat()}); write_json(path,data); print(path)
if __name__=='__main__': main()
