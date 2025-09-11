"""Data downloading functionality for nflreadpy."""

from typing import Any
from urllib.parse import urljoin

import polars as pl
import requests
from tqdm import tqdm

from .cache import get_cache_manager
from .config import DataFormat, get_config


class NflverseDownloader:
    """Downloads data from nflverse repositories."""

    BASE_URLS = {
        "nflverse-data": "https://github.com/nflverse/nflverse-data/releases/download/",
        "nfldata": "https://github.com/nflverse/nfldata/raw/master/data/",
        "espnscraper": "https://github.com/nflverse/espnscrapeR-data/raw/master/data/",
        "dynastyprocess": "https://github.com/dynastyprocess/data/raw/master/files/",
        "ffopportunity": "https://github.com/ffverse/ffopportunity/releases/download/",
    }

    def __init__(self) -> None:
        self.session = requests.Session()
        self.cache = get_cache_manager()

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for requests."""
        config = get_config()
        return {
            "User-Agent": config.user_agent,
            "Accept": "application/octet-stream, text/csv, */*",
        }

    def _build_url(self, repository: str, path: str, format_type: DataFormat) -> str:
        """Build the full URL for a data file."""
        if repository not in self.BASE_URLS:
            raise ValueError(f"Unknown repository: {repository}")

        base_url = self.BASE_URLS[repository]

        # Add format extension if not present
        if not path.endswith((".parquet", ".csv")):
            ext = ".parquet" if format_type == DataFormat.PARQUET else ".csv"
            path = f"{path}{ext}"

        return urljoin(base_url, path)

    def _download_file(self, url: str, **kwargs: Any) -> pl.DataFrame:
        """Download and parse a data file."""
        config = get_config()

        # Check cache first
        cached_data = self.cache.get(url, **kwargs)
        if cached_data is not None:
            return cached_data

        # Configure session
        self.session.headers.update(self._get_headers())

        try:
            if config.verbose:
                print(f"Downloading {url}")

            response = self.session.get(url, timeout=config.timeout, stream=True)
            response.raise_for_status()

            # Get content length for progress bar
            total_size = int(response.headers.get("content-length", 0))

            # Download with progress bar if verbose
            content = b""
            if config.verbose and total_size > 0:
                with tqdm(
                    total=total_size, unit="B", unit_scale=True, desc="Downloading"
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            content += chunk
                            pbar.update(len(chunk))
            else:
                content = response.content

            # Parse data based on URL extension
            if url.endswith(".parquet"):
                data = pl.read_parquet(content)
            elif url.endswith(".csv"):
                data = pl.read_csv(content, null_values=["NA", "NULL", ""])
            else:
                # Try to detect format from content
                try:
                    data = pl.read_parquet(content)
                except Exception:
                    data = pl.read_csv(content, null_values=["NA", "NULL", ""])

            # Cache the result
            self.cache.set(url, data, **kwargs)

            return data

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to download {url}: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to parse data from {url}: {e}") from e

    def download(
        self,
        repository: str,
        path: str,
        format_preference: DataFormat | None = None,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """
        Download data from an nflverse repository.

        Args:
            repository: The repository name (e.g., 'nflverse-data')
            path: The path to the data file within the repository
            format_preference: Preferred format (parquet or csv)
            **kwargs: Additional parameters for caching

        Returns:
            Polars DataFrame with the requested data
        """
        config = get_config()
        format_type = format_preference or config.prefer_format

        # Try preferred format first
        try:
            url = self._build_url(repository, path, format_type)
            return self._download_file(url, **kwargs)
        except (ConnectionError, requests.exceptions.HTTPError) as e:
            # If preferred format fails, try the alternative
            if format_type == DataFormat.PARQUET:
                alt_format = DataFormat.CSV
            else:
                alt_format = DataFormat.PARQUET

            if config.verbose:
                print(
                    f"Failed to download {format_type.value} format, trying {alt_format.value}"
                )

            try:
                url = self._build_url(repository, path, alt_format)
                return self._download_file(url, **kwargs)
            except Exception:
                # Re-raise the original error
                raise e from None


# Global downloader instance
_downloader = NflverseDownloader()


def get_downloader() -> NflverseDownloader:
    """Get the global downloader instance."""
    return _downloader
