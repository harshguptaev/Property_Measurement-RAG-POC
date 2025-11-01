ption 1: Export in Terminal
export
export PG_DB_PASSWORD="your_database_password_here"
Option 2: Add to Shell Profile
# Add to ~/.bashrc or ~/.zshrcecho 'export PG_DB_PASSWORD="your_database_password_here"' >> ~/.zshrcsource ~/.zshrc
Option 3: Set in Virtual Environment
# When activating venv_ragsource venv_rag/bin/activateexport PG_DB_PASSWORD="your_database_password_here"
Verification:
# Check if password is setecho $PG_DB_PASSWORD# Test database connectionpython -c "from src.property_rag_status_dao import PropertyRAGStatusDAO; print('✅ Database connected!')"
Note: The password is not stored in code or config files for security - it must be set as an environment variable when running the application.