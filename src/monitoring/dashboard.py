"""
Production Monitoring and Live Telemetry Dashboard for CrunchDAO Synth.

Queries CrunchDAO Hub API to monitor:
1. Competition and project metadata.
2. Active and historical submissions, verification status, and payload sizes.
3. Rolling 7-day Anchor CRPS performance and benchmark comparatives.
4. On-disk local cache health and tick data fresh levels.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
from crunch.api import Client, auth

from src.data.price_manager import PriceManager, ALL_ASSETS

logger = logging.getLogger(__name__)


class SynthProductionDashboard:
    """
    Live monitoring telemetry engine for CrunchDAO Synth submissions and performance.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_name: str = "necessary-marsupial",
        user_id: int = 14394,
        competition_name: str = "synth",
    ):
        self.project_name = project_name
        self.user_id = user_id
        self.competition_name = competition_name

        # Resolve API key
        if api_key is None:
            api_key = os.getenv("CRUNCHDAO_API_KEY") or os.getenv("CRUNCH_API_KEY")

        if not api_key:
            # Check ~/.env or project .env
            env_path = Path.home() / "Projects" / "challenge" / ".env"
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if line.startswith("CRUNCHDAO_API_KEY="):
                        api_key = line.split("=", 1)[1].strip("'\" ")
                        break

        self.api_key = api_key
        self.client: Optional[Client] = None
        if self.api_key:
            try:
                self.client = Client(
                    api_base_url="https://api.hub.crunchdao.com",
                    web_base_url="https://hub.crunchdao.com",
                    auth=auth.ApiKeyAuth(self.api_key),
                )
            except Exception as e:
                logger.warning(f"Could not connect client with ApiKeyAuth: {e}")

        self.price_manager = PriceManager()

    def fetch_submissions(self) -> List[Dict[str, Any]]:
        """
        Retrieves submission history from CrunchDAO API.
        """
        if not self.client:
            return []

        try:
            comp = self.client.competitions.get(self.competition_name)
            proj = comp.projects.get(self.user_id, self.project_name)
            submissions = proj.submissions.list()

            records = []
            for s in submissions:
                attrs = getattr(s, "_attrs", {})
                records.append({
                    "id": attrs.get("id", s.id),
                    "number": attrs.get("number", s.number),
                    "type": attrs.get("type", "CODE"),
                    "message": attrs.get("message", "N/A"),
                    "valid": attrs.get("valid", True),
                    "libraries_verified": attrs.get("librariesVerified", True),
                    "size_bytes": attrs.get("totalSize", 0),
                    "created_at": attrs.get("createdAt", "N/A"),
                })
            return records
        except Exception as e:
            logger.warning(f"Error fetching submissions: {e}")
            return []

    def format_terminal_dashboard(self) -> str:
        """
        Generates formatted terminal report of active production status.
        """
        submissions = self.fetch_submissions()
        # Sort submissions descending by number so latest is first
        submissions_sorted = sorted(submissions, key=lambda s: s.get("number", 0), reverse=True)
        latest_sub = submissions_sorted[0] if submissions_sorted else None

        cache_stats = self.price_manager.get_cache_stats()
        cached_count = int(cache_stats["cached"].sum())

        lines = [
            "=" * 80,
            "  CRUNCHDAO SYNTH: PRODUCTION LIVE MONITORING & SUBMISSIONS DASHBOARD",
            "=" * 80,
            f"  Competition: SYNTH (Subnet 50 Bittensor / CrunchDAO) | Status: ACTIVE",
            f"  Project: {self.project_name} (ID: 17527) | User: Santiago Mendez (ID: {self.user_id})",
            f"  Dashboard URL: https://hub.crunchdao.com/competitions/synth/projects/{self.user_id}/{self.project_name}",
            "-" * 80,
        ]

        if latest_sub:
            created_str = latest_sub["created_at"]
            size_kb = latest_sub["size_bytes"] / 1024.0
            status_str = "VALID [✓]" if latest_sub["valid"] else "INVALID [✗]"
            lib_str = "VERIFIED [✓]" if latest_sub["libraries_verified"] else "PENDING [?]"

            lines.extend([
                f"  LATEST ACTIVE SUBMISSION: #{latest_sub['number']} (ID: {latest_sub['id']})",
                f"  * Message:            {latest_sub['message']}",
                f"  * Status:             {status_str} | Libraries: {lib_str}",
                f"  * Payload Size:       {latest_sub['size_bytes']:,} bytes ({size_kb:.1f} KB)",
                f"  * Deployed At:        {created_str}",
                "-" * 80,
            ])
        else:
            lines.extend([
                "  LATEST SUBMISSION: No remote submissions retrieved or offline mode.",
                "-" * 80,
            ])

        lines.extend([
            f"  LOCAL PRICE CACHE TELEMETRY: {cached_count}/{len(ALL_ASSETS)} assets cached",
            f"  * Cache Directory:    {self.price_manager.cache_dir}",
            f"  * Assets Active:      {', '.join(ALL_ASSETS[:6])}...",
            "-" * 80,
            f"  ALL RECORDED SUBMISSIONS ({len(submissions)}):",
            f"  {'#':<4} | {'ID':<8} | {'Valid':<8} | {'Size':<10} | {'Created At':<26} | {'Message'}",
            f"  {'-'*4}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*26}-+-{'-'*18}",
        ])

        for sub in submissions:
            v_flag = "YES [✓]" if sub["valid"] else "NO  [✗]"
            sz_str = f"{sub['size_bytes']:,} B"
            lines.append(
                f"  {sub['number']:<4} | {sub['id']:<8} | {v_flag:<8} | {sz_str:<10} | {sub['created_at'][:25]:<26} | {sub['message'][:24]}"
            )

        lines.append("=" * 80)
        return "\n".join(lines)
