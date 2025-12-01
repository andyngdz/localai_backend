# Config API

## Purpose

Configuration endpoint for frontend to fetch application settings without hardcoding.

## Requirements

### Requirement: Config Endpoint

The system SHALL provide GET /config endpoint.

#### Scenario: Fetch config

- **WHEN** GET /config is called
- **THEN** configuration data is returned

### Requirement: Upscaler Options

The system SHALL return upscaler options with metadata including recommendation status.

#### Scenario: Upscaler data

- **WHEN** config is fetched
- **THEN** upscalers array contains value, name, description, suggested_denoise_strength, method, is_recommended

#### Scenario: Recommended upscalers

- **WHEN** config is fetched
- **THEN** RealESRGAN upscalers (x2plus, x4plus, x4plus_anime) have is_recommended=true
- **AND** traditional upscalers (Lanczos, Bicubic, Bilinear, Nearest) have is_recommended=false

#### Scenario: All upscalers included

- **WHEN** config is fetched
- **THEN** Lanczos, Bicubic, Bilinear, Nearest are included

### Requirement: Extensible Schema

The system SHALL use extensible schema for future config items.

#### Scenario: Add new config

- **WHEN** new config type is needed
- **THEN** schema can be extended without breaking changes

## Key Entities

- **UpscalerItem**: value, name, description, suggested_denoise_strength, method, is_recommended
- **ConfigResponse**: upscalers list
