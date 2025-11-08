"""InfluxDB integration utilities."""

from datetime import datetime
from typing import Any, Dict, Optional

import structlog


logger = structlog.get_logger()


class InfluxDBReporter:
    """Reporter for sending metrics to InfluxDB.
    
    This is a simple integration that can be extended as needed.
    """

    def __init__(
        self,
        url: str,
        token: str,
        org: str,
        bucket: str,
        measurement: str = "numerai_metrics",
    ):
        """Initialize InfluxDB reporter.
        
        Args:
            url: InfluxDB URL
            token: Authentication token
            org: Organization name
            bucket: Bucket name
            measurement: Measurement name for metrics
        """
        try:
            from influxdb_client import InfluxDBClient, Point
            from influxdb_client.client.write_api import SYNCHRONOUS
            
            self.client = InfluxDBClient(url=url, token=token, org=org)
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.bucket = bucket
            self.org = org
            self.measurement = measurement
            self.Point = Point
            
            logger.info("influxdb_initialized", url=url, org=org, bucket=bucket)
        except ImportError:
            logger.error("influxdb_client_not_installed")
            raise

    def report_metrics(
        self,
        metrics: Dict[str, Any],
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Report metrics to InfluxDB.
        
        Args:
            metrics: Dictionary of metric names and values
            tags: Optional tags for the metrics
            timestamp: Optional timestamp (default: now)
        """
        try:
            point = self.Point(self.measurement)
            
            # Add tags
            if tags:
                for key, value in tags.items():
                    point = point.tag(key, value)
            
            # Add metrics as fields
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    point = point.field(key, value)
            
            # Add timestamp
            if timestamp:
                point = point.time(timestamp)
            
            # Write to InfluxDB
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            
            logger.debug("metrics_reported", metrics=list(metrics.keys()))
            
        except Exception as e:
            logger.error("influxdb_write_error", error=str(e))

    def close(self) -> None:
        """Close InfluxDB client."""
        if hasattr(self, "client"):
            self.client.close()
            logger.info("influxdb_closed")


def create_influxdb_reporter(
    url: str,
    token: str,
    org: str,
    bucket: str,
) -> Optional[InfluxDBReporter]:
    """Create an InfluxDB reporter if the client is available.
    
    Args:
        url: InfluxDB URL
        token: Authentication token
        org: Organization name
        bucket: Bucket name
        
    Returns:
        InfluxDBReporter or None if not available
    """
    try:
        return InfluxDBReporter(url, token, org, bucket)
    except ImportError:
        logger.warning("influxdb_client_not_available")
        return None
