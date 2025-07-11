# Changelog

All notable changes to the Tora Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
