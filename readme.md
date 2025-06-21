 # Tackle4Loss News Collection

[![CI/CD Pipeline](https://github.com/bigsliktobi/Tackle4LossNewsCollection/actions/workflows/ci.yml/badge.svg)](https://github.com/bigsliktobi/Tackle4LossNewsCollection/actions/workflows/ci.yml)
 
 ## Purpose

Tackle4LossNewsCollection is **part 1** of the Tackle4Loss Project that **gathers** extracts, enriches and publicates American Football News an Tackle4Loss.com.

Tackle4Loss News Collection is designed to automatically gather, process, and organize news articles related to specific topics or teams, with a focus on the 'Tackle4Loss' initiative. It leverages Language Models (LLMs) to intelligently process fetched content and stores the curated information in a Supabase database. The project includes a general news collection pipeline, enabling targeted information gathering and analysis.
 
 ## Key Features
 

 *   **Automated News Aggregation**: Gathers articles from a diverse set of online news sources and websites.
 *   **LLM-Powered Content Processing**: Utilizes Language Models (e.g., OpenAI, Gemini) for tasks like summarizing or extracting key information from articles (though current implementation focuses on fetching based on source's provided content).
 *   **Centralized Storage**: Stores processed articles in a structured Supabase database, making them easily accessible for search and analysis.
 *   **Configurable Sources and LLMs**: Allows users to define news sources and choose LLM providers through configuration.
 *   **Automated Workflows**: Includes GitHub Actions for scheduled/automated execution of the collection pipelines.
 *   **Logging**: Comprehensive logging for monitoring and troubleshooting pipeline execution.
 *   **Blacklist Functionality**: Supports a `blacklist.json` file to exclude specific domains or URLs from the news fetching process.

## Setup Instructions

Follow these steps to set up and run the Tackle4Loss News Collection tool:

### 1. Prerequisites
*   Python 3.8 or higher.
*   `pip` (Python package installer).

### 2. Clone the Repository
   ```bash
   git clone https://github.com/BigSlikTobi/Tackle4LossNewsCollection.git 
   cd Tackle4LossNewsCollection
   ```

### 3. Install Dependencies
   Install the required Python libraries using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

### 4. Configure Environment Variables
   This project uses environment variables for configuration. Create a file named `.env` in the root directory of the project and add the following variables:

   ```env
   # Supabase Configuration (Mandatory)
   SUPABASE_URL="your_supabase_project_url"
   SUPABASE_KEY="your_supabase_anon_key"

   # LLM Configuration (Mandatory, choose at least one provider)
   LLM_PROVIDER="openai" # Or "gemini", etc. Defaults to "openai" if not set.
   OPENAI_API_KEY="your_openai_api_key" # Required if LLM_PROVIDER is "openai"
   # GOOGLE_API_KEY="your_google_api_key" # Required if LLM_PROVIDER is "gemini"
   # Add other provider-specific API keys as needed, based on llm_selector.py

   # Logging Configuration (Optional)
   LOG_LEVEL="INFO" # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL. Default: INFO
   # LOG_DIR="logs" # Default: ./logs
   # LOG_FILE="pipeline.log" # Default: pipeline.log (inside LOG_DIR)
   ```

   **Important:**
   *   Replace placeholder values (e.g., `"your_supabase_project_url"`) with your actual credentials and configuration.
   *   The `.env` file is included in `.gitignore` by default to prevent accidental sharing of sensitive keys.
   *   The application loads these variables at startup using `python-dotenv`.

## Running the Pipelines

Once the setup is complete, you can run the news collection pipelines using the following commands from the project's root directory:

### General News Pipeline
This pipeline fetches general news articles based on the default sources configured in `source_manager.py`.
```bash
python pipeline.py
```

### Automated Execution
The project also includes GitHub Actions workflows defined in the `.github/workflows/` directory (e.g., `news_collection.yml`). These workflows can be configured to run the pipeline on a schedule or based on other triggers, automating the news collection process.

## Module Structure

The project is organized into several key modules:

*   **`pipeline.py`**: Main script that orchestrates the general news collection pipeline. It handles initialization, source loading, fetching, processing, and storage of news articles.
*   **`news_fetcher.py`**: Contains the logic for fetching news content from various URLs and sources. It interacts with the LLM for initial processing if required by the source.
*   **`source_manager.py`**: Responsible for managing and providing the configurations for news sources.
*   **`llm_selector.py`**: Manages the selection and initialization of Language Model (LLM) providers (e.g., OpenAI, Gemini). It also handles API key retrieval for the selected LLM.
*   **`db_operations.py`**: Contains functions for interacting with the Supabase database, primarily for storing the fetched and processed articles.
*   **`database_functions.py`**: Provides the Supabase client instance and related utility functions for database connection.
*   **`utils.py`**: A collection of utility functions used across the project (e.g., URL cleaning, slug creation).
*   **`requirements.txt`**: Lists all Python dependencies required for the project.
*   **`logs/`**: The default directory where log files (e.g., `pipeline.log`) are stored.
*   **`.github/workflows/`**: Contains YAML files defining GitHub Actions workflows for continuous integration and automated pipeline execution (e.g., `news_collection.yml`, `team-news-collection.yml`).
*   **`.env` (example)**: A file (typically not committed to version control) that stores environment variables like API keys and database URLs.
*   **`blacklist.json`**: A file used to list domains or URLs that should be excluded from news fetching.

## High-Level Workflow

The news collection pipelines generally follow these steps:

1.  **Initialization**:
    *   Logging is set up according to environment variables (`LOG_LEVEL`, `LOG_DIR`, `LOG_FILE`).
    *   Environment variables are loaded from the `.env` file.
2.  **Configuration Loading**:
    *   Essential environment variables (like `SUPABASE_URL`, `SUPABASE_KEY`, and LLM API keys) are validated.
    *   News sources are loaded via `source_manager.py`.
3.  **Component Setup**:
    *   The appropriate LLM provider is initialized using `llm_selector.py` based on the `LLM_PROVIDER` environment variable and its corresponding API key.
    *   A connection to the Supabase database is established via `database_functions.py` and `db_operations.py`.
4.  **News Fetching**:
    *   `news_fetcher.py` iterates through the active sources.
    *   For each source, it fetches potential news items/articles. This might involve directly accessing RSS feeds, scraping web pages, or using the LLM to identify relevant links on a page.
5.  **Data Formatting**:
    *   The fetched items are standardized into a common format, including fields like `uniqueName`, `source`, `headline`, `href`, `url`, and `publishedAt`.
6.  **Storage**:
    *   The formatted news items are stored in the Supabase database using the `SourceArticles` table.
7.  **Logging and Output**:
    *   Throughout the process, detailed logs are generated.
    *   The pipelines output summaries of their activity, such as the number of articles fetched and stored.
