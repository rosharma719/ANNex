class AnnexError(Exception):
    """Raised when the ANNex server returns a 4xx or 5xx response."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message
        super().__init__(f"AnnexError {status_code}: {message}")
