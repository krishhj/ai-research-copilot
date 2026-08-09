from app.storage.file_storage import FileStorage

def test_save_file(tmp_path):
    storage = FileStorage(base_directory=tmp_path)

    saved_path = storage.save(
        content=b"Sample PDF content",
        stored_filename="test-paper.pdf"
    )

    assert saved_path.exists()
    assert saved_path.name == "test-paper.pdf"
    assert saved_path.read_bytes() == b"Sample PDF content"