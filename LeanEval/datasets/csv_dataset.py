# csv_dataset.py
from __future__ import annotations

import csv
import json
import re
from typing import Iterable, List

from .base import BaseDataset
from .schema import LeanItem


class CsvDataset(BaseDataset):
    """一次性加载整个 CSV 文件"""

    _SEP_PATTERN = re.compile(r"[;,]\s*")  # imports 字段分隔符：; 或 ,

    # ---------- 读取原始行 ----------
    def _read_raw(self) -> Iterable[dict]:
        with self.path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # ---- 字段清洗 ----
                # 1) imports: str → List[str]
                imports_str = (row.get("imports") or "").strip()
                if imports_str:
                    try:
                        # 允许直接写 JSON list
                        row["imports"] = json.loads(imports_str)
                        if not isinstance(row["imports"], list):
                            raise ValueError
                    except Exception:
                        # 或者按 ; , 拆分
                        row["imports"] = [
                            s.strip() for s in self._SEP_PATTERN.split(imports_str) if s.strip()
                        ]
                else:
                    row["imports"] = []

                # 2) difficulty: str → int
                if row.get("difficulty"):
                    row["difficulty"] = int(row["difficulty"])

                # 3) None → ""
                for k, v in row.items():
                    if v is None:
                        row[k] = ""

                yield row

    # ---------- 构造 LeanItem ----------
    def _load(self) -> List[LeanItem]:
        return [LeanItem(**item) for item in self._read_raw()]
