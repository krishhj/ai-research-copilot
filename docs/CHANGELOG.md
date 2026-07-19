# Changelog

All notable changes to **AI Research Copilot** are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows [Semantic Versioning](https://semver.org/) — while still in `0.x`, things may still shift around as the architecture settles.

---

## [Unreleased]

### Added
- API design document defining the backend HTTP contract (`docs/api_design.md`) — base path, standard success/error response shapes, and planned endpoints

---

## [0.2.0] - 2026-07-15 — Domain Layer Complete

All core domain models are implemented and unit-tested: `Paper`, `Chunk`, `Query`, `Answer`, and `Conversation`.

### Added
- `Chunk` model with unit tests
- `Query` model with unit tests
- `Answer` model with unit tests
- `Conversation` model with unit tests
- `ConversationRoles` enum
- `__init__.py` for the `models` package
- `docs/architecture.md` describing system layers and core workflows
- README updated to reflect the completed domain layer

### Changed
- Renamed `text` field to `chunk_text` on the `Chunk` model for clarity
- Reformatted codebase with `black`
- Modernized type hints (`Optional[X]` → `X | None`)
- Sorted and cleaned up imports with `isort`
- Consolidated and cleaned up core foundation code

### Fixed
- Removed an unused import flagged by `ruff`

---

## [0.1.0] - 2026-07-09 — Project Foundation

Initial backend scaffold: configuration, logging, error handling, and the first domain model.

### Added
- Initial repository setup
- Centralized configuration via Pydantic Settings (`config.py`)
- `.env.example` template
- `requirements.txt`
- Core infrastructure: logger and custom exception handling
- `PaperStatus` enum
- `Paper` model (`PaperMetaData`, `ProcessingMetadata`, `Paper`) with unit tests
- `pytest.ini` for test configuration
- README documenting the v0.1.0 milestone

### Changed
- Refined `.gitignore` to exclude additional generated folders

### Removed
- Unused constants from `app/core/constants.py`

### Fixed
- Spelling mistakes in docstrings/comments
- Incorrectly named file

---

## Notes on Versioning

This repo hasn't had formal GitHub releases/tags cut yet — the version numbers above are **development milestones** pulled from your own commit history (`Add README for v0.1.0`, and the domain-layer-complete checkpoint), not actual git tags. To make the links below work and keep this file self-maintaining going forward, tag matching releases when you're ready:

```powershell
git tag -a v0.1.0 5ce9b2e -m "Project foundation"
git tag -a v0.2.0 <commit-hash-of-domain-layer-README-update> -m "Domain layer complete"
git push origin --tags
```

[Unreleased]: https://github.com/krishhj/ai-research-copilot/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/krishhj/ai-research-copilot/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/krishhj/ai-research-copilot/releases/tag/v0.1.0