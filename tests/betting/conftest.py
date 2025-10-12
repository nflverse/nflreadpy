import copy
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import pytest

from nflreadpy.betting.alerts import AlertSink
from nflreadpy.betting.ingestion import OddsIngestionService
from nflreadpy.betting.scrapers.base import (
    OddsQuote,
    SportsbookScraper,
    StaticScraper,
)
from nflreadpy.betting.scrapers.draftkings import DraftKingsScraper
from nflreadpy.betting.scrapers.fanduel import FanDuelScraper
from nflreadpy.betting.scrapers.pinnacle import PinnacleScraper


FANDUEL_URL = "https://stub/fanduel"
DRAFTKINGS_URL = "https://stub/draftkings"
PINNACLE_URL = "https://stub/pinnacle"


class RecordingAlertSink(AlertSink):
    def __init__(self) -> None:
        self.messages: List[Tuple[str, str, Mapping[str, Any] | None]] = []

    def send(
        self,
        subject: str,
        body: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.messages.append((subject, body, metadata))


class StubHTTPClient:
    def __init__(
        self,
        responses: Dict[str, Dict[str, Any]],
        *,
        freshen: bool,
    ) -> None:
        self._responses = responses
        self._freshen = freshen
        self._timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
        self.calls: list[str] = []

    async def get_json(
        self,
        url: str,
        *,
        params: Dict[str, Any] | None = None,
        headers: Dict[str, str] | None = None,
    ) -> Dict[str, Any]:
        del params, headers
        self.calls.append(url)
        payload = copy.deepcopy(self._responses[url])
        if self._freshen:
            _refresh_event_timestamps(payload, self._timestamp)
        return payload


def _refresh_event_timestamps(payload: Dict[str, Any], timestamp: str) -> None:
    events = payload.get("events", [])
    for event in events:
        if "lastUpdated" in event:
            event["lastUpdated"] = timestamp
        if "lastUpdate" in event:
            event["lastUpdate"] = timestamp


@pytest.fixture()
def now() -> dt.datetime:
    return dt.datetime(2024, 9, 1, 12, tzinfo=dt.timezone.utc)


@pytest.fixture()
def sportsbook_payloads() -> Dict[str, Dict[str, Any]]:
    payload_dir = Path(__file__).parent / "payloads"
    return {
        FANDUEL_URL: json.loads((payload_dir / "fanduel.json").read_text()),
        DRAFTKINGS_URL: json.loads((payload_dir / "draftkings.json").read_text()),
        PINNACLE_URL: json.loads((payload_dir / "pinnacle.json").read_text()),
    }


@pytest.fixture()
def fresh_stub_client(sportsbook_payloads: Dict[str, Dict[str, Any]]) -> StubHTTPClient:
    return StubHTTPClient(sportsbook_payloads, freshen=True)


@pytest.fixture()
def stale_stub_client(sportsbook_payloads: Dict[str, Dict[str, Any]]) -> StubHTTPClient:
    return StubHTTPClient(sportsbook_payloads, freshen=False)


@pytest.fixture()
def http_scrapers(fresh_stub_client: StubHTTPClient) -> List[SportsbookScraper]:
    return [
        FanDuelScraper(FANDUEL_URL, client=fresh_stub_client, rate_limit_per_second=None),
        DraftKingsScraper(
            DRAFTKINGS_URL, client=fresh_stub_client, rate_limit_per_second=None
        ),
        PinnacleScraper(PINNACLE_URL, client=fresh_stub_client, rate_limit_per_second=None),
    ]


@pytest.fixture()
def scraper_configs(stale_stub_client: StubHTTPClient) -> List[Dict[str, Any]]:
    return [
        {
            "type": "fanduel",
            "endpoint": FANDUEL_URL,
            "client": stale_stub_client,
            "rate_limit_per_second": None,
        },
        {
            "type": "draftkings",
            "endpoint": DRAFTKINGS_URL,
            "client": stale_stub_client,
            "rate_limit_per_second": None,
        },
        {
            "type": "pinnacle",
            "endpoint": PINNACLE_URL,
            "client": stale_stub_client,
            "rate_limit_per_second": None,
        },
    ]


@pytest.fixture()
def alert_sink() -> RecordingAlertSink:
    return RecordingAlertSink()


@pytest.fixture()
def static_quotes(now: dt.datetime) -> List[OddsQuote]:
    return [
        OddsQuote(
            event_id="2024-NE-NYJ",
            sportsbook="testbook",
            book_market_group="Game Lines",
            market="moneyline",
            scope="game",
            entity_type="team",
            team_or_player="NE",
            side=None,
            line=None,
            american_odds=-135,
            observed_at=now,
            extra={"opponent": "NYJ"},
        ),
        OddsQuote(
            event_id="2024-NE-NYJ",
            sportsbook="testbook",
            book_market_group="Game Lines",
            market="moneyline",
            scope="game",
            entity_type="team",
            team_or_player="NYJ",
            side=None,
            line=None,
            american_odds=+125,
            observed_at=now,
            extra={"opponent": "NE"},
        ),
        OddsQuote(
            event_id="2024-NE-NYJ",
            sportsbook="testbook",
            book_market_group="Game Lines",
            market="spread",
            scope="game",
            entity_type="team",
            team_or_player="NE",
            side="fav",
            line=-3.5,
            american_odds=-105,
            observed_at=now,
            extra={},
        ),
        OddsQuote(
            event_id="2024-NE-NYJ",
            sportsbook="testbook",
            book_market_group="Game Lines",
            market="total",
            scope="game",
            entity_type="total",
            team_or_player="Total",
            side="over",
            line=41.5,
            american_odds=-108,
            observed_at=now,
            extra={},
        ),
        OddsQuote(
            event_id="2024-NE-NYJ",
            sportsbook="testbook",
            book_market_group="Player Props",
            market="receiving_yards",
            scope="game",
            entity_type="player",
            team_or_player="Garrett Wilson",
            side="over",
            line=68.5,
            american_odds=+120,
            observed_at=now,
            extra={"projection_mean": 82.0, "projection_stdev": 16.0},
        ),
        OddsQuote(
            event_id="2024-NE-NYJ",
            sportsbook="testbook",
            book_market_group="Either Player",
            market="longest_reception",
            scope="game",
            entity_type="either",
            team_or_player="Sutton/Dobbins",
            side="yes",
            line=29.5,
            american_odds=+140,
            observed_at=now,
            extra={
                "participants": ["Courtland Sutton", "J.K. Dobbins"],
                "projection_mean": 0.55,
            },
        ),
    ]


@pytest.fixture()
def static_scraper(static_quotes: List[OddsQuote]) -> StaticScraper:
    return StaticScraper("testbook", static_quotes)


@pytest.fixture()
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "odds.sqlite3"


@pytest.fixture()
def ingestion(static_scraper: StaticScraper, tmp_db_path: Path) -> OddsIngestionService:
    return OddsIngestionService(
        [static_scraper], storage_path=tmp_db_path, stale_after=dt.timedelta(days=5000)
    )
