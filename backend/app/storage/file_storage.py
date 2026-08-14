from pathlib import Path

class FileStorage:
    """Store uploaded files in a local directory """
    def __init__(self, base_directory: Path) -> None:
        self._base_directory = base_directory
        self._base_directory.mkdir(parents=True,exist_ok=True)

    def save (self, content: bytes, stored_filename: str) -> Path:
        """Save file content and return its storage path."""   
        file_path = self._base_directory / stored_filename

        with file_path.open("xb") as file:
            file.write(content)

        return file_path

    def delete(self, stored_filename: str) -> None:
        """Delete one stored file."""
        file_path = self._base_directory / stored_filename
        file_path.unlink()