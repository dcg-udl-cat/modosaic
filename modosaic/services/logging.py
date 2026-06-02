import atexit
import json
import logging
import logging.config
from importlib import resources
from pathlib import Path

from transformers.utils import logging as hf_logging


class LoggingService:
    """Project logging bootstrapper (JSON config + optional file sink).

    Loads a JSON logging configuration bundled with the package
    (`modosaic.log/log.json`), optionally rewrites the file handler’s output path
    to a user-defined directory, applies the configuration, and starts a
    queue listener if present.

    Notes:
        - The config is read via `importlib.resources.open_text("modosaic.log", "log.json")`.
        - If the config defines a handler named `"file_json"` under `"handlers"`,
          its `"filename"` will be set to `<log_dir>/modosaic.log.jsonl` where
          `log_dir` defaults to `~/.modosaic/logs` (created if missing).
        - If a handler named `"queue_handler"` exists and exposes a `.listener`,
          that listener is started, and automatically stopped at process exit
          via `atexit`.

    Example:
        >>> # Minimal usage: use default ~/.modosaic/logs for file sink (if configured)
        >>> LoggingService.setup_logging()
        >>> # Or direct logs to a custom folder
        >>> LoggingService.setup_logging(Path("./runs/2025-10-02/logs"))
        >>> logging.getLogger(__name__).info("Logging initialized")
    """

    @staticmethod
    def setup_logging(log_dir: Path | None = None) -> None:
        """Configure and start the project-wide logging.

        Steps:
            1) Load JSON config from the package resource `modosaic.log/log.json`.
            2) If a `"file_json"` handler exists, ensure a log directory
               (default `~/.modosaic/logs` or the provided `log_dir`) and point the
               handler’s `"filename"` to `modosaic.log.jsonl` within that folder.
            3) Apply the configuration via `logging.config.dictConfig`.
            4) Start a queue listener (`queue_handler.listener.start()`) when a
               handler named `"queue_handler"` is present; register an `atexit`
               hook to stop it cleanly.

        Args:
            log_dir: Optional directory for file-based logs. If omitted and the
                `"file_json"` handler is present, defaults to `~/.modosaic/logs`.

        Raises:
            FileNotFoundError: If the packaged JSON config cannot be found.
            json.JSONDecodeError: If the JSON config is invalid.
            ValueError: If `dictConfig` receives an invalid configuration.

        """
        hf_logging.set_verbosity_error()
        with resources.open_text("modosaic.log", "log.json") as f_in:
            config = json.load(f_in)

        if "handlers" in config and "file_json" in config["handlers"]:
            # Create logs directory in user's home directory
            log_dir = log_dir or Path.home() / ".modosaic" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            # Update the filename to use the created directory
            config["handlers"]["file_json"]["filename"] = str(
                log_dir / "modosaic.log.jsonl"
            )

        for noisy_logger in ["httpx", "httpcore"]:
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)

        logging.config.dictConfig(config)
        queue_handler = logging.getHandlerByName("queue_handler")
        if queue_handler is not None:
            queue_handler.listener.start()  # type: ignore
            atexit.register(queue_handler.listener.stop)  # type: ignore
