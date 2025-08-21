# Changelog

All notable changes to the Tora Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.10] - 2025-08-21

### Changed
- BREAKING: Client now posts logs to `/experiments/{id}/logs/batch` (renamed from `/metrics/batch`). Aligns with backend route/table rename from `metric` to `log`.
- No behavior change to `Tora.metric`/`Tora.result` beyond the endpoint path.

## [0.0.9] - 2025-08-21

### Changed
- BREAKING: `Tora.metric` signature updated to `metric(name, value, step_or_epoch: int)`; `step_or_epoch` is now required and replaces the former optional `step` parameter.
- Any wrapper/utilities that call `Tora.metric` should pass `step_or_epoch` explicitly.

## [0.0.8] - 2025-08-20

### Added
- `tmetric(name, value, step=None)` wrapper in `tora._wrapper` for logging training metrics.

### Changed
- Public API: `tlog` removed from exports in favor of `tmetric`.
- Result logging metadata standardized to `{"type": "result"}`.

### Migration
- Replace calls to `tlog(name, value, step, metadata)` with `tmetric(name, value, step)`.
- For results, use `tresult(name, value)`; result metadata is handled internally.

## [0.0.5] - 2024-01-XX

### Added
- Comprehensive type hints throughout the codebase
- Input validation for all public methods
- Custom exception classes for better error handling
- Buffered metric logging with configurable buffer size
- Context manager support for automatic cleanup
- Global wrapper API for simplified usage
- Comprehensive test suite with >90% coverage
- Pre-commit hooks and development tooling
- Detailed documentation and examples
- Support for metric metadata
- Flush method for immediate metric sending
- Properties for accessing client state

### Changed
- **BREAKING**: Improved error handling with custom exception types
- **BREAKING**: Enhanced validation of inputs (may reject previously accepted invalid inputs)
- **BREAKING**: Updated API response handling to match backend changes
- Improved HTTP client with better error handling and timeout support
- Enhanced logging with structured debug information
- Updated project configuration with modern Python packaging standards

### Fixed
- Fixed workspace creation API payload format
- Fixed experiment loading with proper error handling
- Fixed metric logging validation and error reporting
- Fixed connection management and resource cleanup
- Fixed Unicode handling in HTTP responses

### Security
- Added input sanitization and validation
- Improved error message handling to avoid information leakage

## [0.0.4] - 2023-XX-XX

### Added
- Basic experiment creation and metric logging
- HTTP client implementation
- Configuration management
- Simple wrapper functions

### Fixed
- Basic functionality and API integration

## [0.0.3] - 2023-XX-XX

### Added
- Initial release with core functionality

## [0.0.2] - 2023-XX-XX

### Added
- Early development version

## [0.0.1] - 2023-XX-XX

### Added
- Initial package structure
