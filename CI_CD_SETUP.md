# CI/CD Setup Documentation

## Overview
This project now includes a comprehensive CI/CD pipeline using GitHub Actions that automatically tests, builds, and deploys the application on every push.

## Workflows

### 1. Main CI/CD Pipeline (`.github/workflows/ci.yml`)
**Triggers**: Push to `main`, `master`, or `develop` branches, and pull requests
**Features**:
- **Multi-Python Testing**: Tests against Python 3.11, 3.12, and 3.13
- **Dependency Caching**: Caches pip dependencies for faster builds
- **System Dependencies**: Installs Chrome and Playwright browsers for testing
- **Code Quality**: Runs flake8 linting and mypy type checking
- **Comprehensive Testing**: Runs the full pytest suite with coverage reporting
- **Security Scanning**: Runs safety and bandit security scans
- **Docker Build**: Builds Docker image on main branch pushes
- **Notifications**: Provides build status notifications

### 2. Release Workflow (`.github/workflows/release.yml`)
**Triggers**: Push of version tags (e.g., `v1.0.0`)
**Features**:
- Runs full test suite before release
- Builds and tags Docker images
- Creates GitHub releases with automated release notes
- Includes Docker usage instructions

### 3. Dependency Management (`.github/dependabot.yml`)
**Features**:
- Weekly automatic dependency updates for Python packages
- Weekly updates for GitHub Actions
- Weekly updates for Docker base images
- Automated PR creation with proper labels and assignees

## Project Templates

### Issue Templates
- **Bug Report**: Structured template for reporting bugs with reproduction steps
- **Feature Request**: Template for proposing new features with priority levels

### Pull Request Template
- Comprehensive checklist for code review
- Type of change categorization
- Testing requirements
- Documentation update reminders

## Docker Support

### Dockerfile
- Multi-stage build optimized for production
- Python 3.13 slim base image
- Playwright browser installation
- Health checks included
- Proper caching for faster builds

### .dockerignore
- Excludes unnecessary files from Docker build context
- Reduces image size and build time

## Environment Variables Required

### For Local Development
```bash
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
ANTHROPIC_API_KEY=your_anthropic_key
DEEPSEEK_API_KEY=your_deepseek_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

### For GitHub Actions (Secrets)
The CI pipeline uses test values for API keys during testing. For production releases, you may want to add these as repository secrets:
- `DOCKER_USERNAME` (optional, for Docker Hub publishing)
- `DOCKER_PASSWORD` (optional, for Docker Hub publishing)

## Badge Integration
The README now includes a CI/CD status badge that shows the current build status.

## Usage

### Running Tests Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest -v

# Run with coverage
pip install coverage
coverage run -m pytest
coverage report -m
```

### Docker Usage
```bash
# Build the image
docker build -t tackle4loss-news-collection .

# Run with environment variables
docker run -e OPENAI_API_KEY=your_key \
           -e SUPABASE_URL=your_url \
           -e SUPABASE_KEY=your_key \
           tackle4loss-news-collection
```

### Creating a Release
1. Create and push a version tag:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```
2. The release workflow will automatically:
   - Run tests
   - Build Docker image
   - Create GitHub release
   - Generate release notes

## Monitoring
- Check the "Actions" tab in GitHub to monitor CI/CD runs
- All workflows include proper error handling and logging
- Failed builds will prevent merging (if branch protection is enabled)

## Next Steps
Consider enabling:
1. **Branch Protection Rules**: Require CI checks to pass before merging
2. **Code Coverage Requirements**: Set minimum coverage thresholds
3. **Deployment to Staging/Production**: Extend workflows for automatic deployment
4. **Slack/Email Notifications**: Add notification integrations for team alerts
