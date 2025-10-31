# UpskillAir Learning Assistant

A powerful RAG (Retrieval-Augmented Generation) chatbot with **persistent PostgreSQL storage**. Upload documents once and they'll be available forever!

## 🚀 Features

- **Persistent Storage**: Documents are stored in PostgreSQL with pgvector - no need to re-upload files
- **Smart Duplicate Detection**: Automatically skips duplicate documents
- **Multi-format Support**: PDF, TXT, DOCX file processing
- **Vector Search**: Fast semantic search using pgvector embeddings
- **Interactive Chat**: Context-aware responses based on your documents
- **Real-time Telemetry**: Monitor system status and processing logs

## 📋 Prerequisites

1. **Python 3.8+**
2. **PostgreSQL 12+ with pgvector extension**
3. **OpenAI API Key** - Get one from [OpenAI Platform](https://platform.openai.com/api-keys)

## 🗄️ PostgreSQL Setup

### Option 1: Local PostgreSQL Installation

#### Windows
```bash
# Download and install PostgreSQL from:
# https://www.postgresql.org/download/windows/

# After installation, open pgAdmin or psql and run:
CREATE DATABASE upskill_rag;
```

#### Linux (Ubuntu/Debian)
```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb upskill_rag
```

#### macOS
```bash
# Install PostgreSQL via Homebrew
brew install postgresql
brew services start postgresql

# Create database
createdb upskill_rag
```

### Option 2: Docker PostgreSQL (Recommended)

```bash
# Run PostgreSQL with pgvector in Docker
docker run -d \
  --name upskill-postgres \
  -e POSTGRES_DB=upskill_rag \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=yourpassword \
  -p 5432:5432 \
  ankane/pgvector
```

### Option 3: Cloud PostgreSQL

Use any cloud PostgreSQL service that supports pgvector:
- **Supabase** (Free tier available, pgvector included)
- **Neon** (Free tier available)
- **AWS RDS** with pgvector extension
- **Google Cloud SQL**
- **Azure Database for PostgreSQL**

### Install pgvector Extension

Connect to your PostgreSQL database and run:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

The application will also attempt to create this extension automatically on first connection.

## 🛠️ Installation

1. **Clone the repository**
```bash
cd openUpskill
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables (optional)**

You can configure PostgreSQL connection via environment variables:
```bash
# Windows PowerShell
$env:POSTGRES_HOST="localhost"
$env:POSTGRES_PORT="5432"
$env:POSTGRES_DB="upskill_rag"
$env:POSTGRES_USER="postgres"
$env:POSTGRES_PASSWORD="yourpassword"

# Linux/Mac
export POSTGRES_HOST="localhost"
export POSTGRES_PORT="5432"
export POSTGRES_DB="upskill_rag"
export POSTGRES_USER="postgres"
export POSTGRES_PASSWORD="yourpassword"
```

Or use Streamlit secrets (`.streamlit/secrets.toml`):
```toml
POSTGRES_HOST = "localhost"
POSTGRES_PORT = "5432"
POSTGRES_DB = "upskill_rag"
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "yourpassword"
```

## 🚀 Usage

1. **Start the application**
```bash
streamlit run app.py
```

2. **Configure API Key**
   - Enter your OpenAI API key in the sidebar
   - The connection will be tested automatically

3. **Connect to PostgreSQL**
   - Click "Database Settings" in the sidebar to configure connection
   - Click "Test PostgreSQL Connection" to verify
   - The pgvector extension will be initialized automatically

4. **Upload Documents** (One-time!)
   - Click "Upload New Documents" in the sidebar
   - Select PDF, TXT, or DOCX files
   - Click "Process & Store Documents"
   - Documents are permanently stored in PostgreSQL

5. **Query Your Documents**
   - Type questions in the chat
   - Get AI-powered answers based on your documents
   - View source documents for each answer

6. **Manage Documents**
   - View all stored documents in the sidebar
   - Duplicates are automatically detected and skipped
   - Clear all documents with one click if needed

## 🔧 Configuration

### Default PostgreSQL Settings
- **Host**: localhost
- **Port**: 5432
- **Database**: upskill_rag
- **User**: postgres
- **Password**: (empty by default)

### Customize Settings
Update the defaults in `app.py`:
```python
DEFAULT_PG_HOST = "your-host"
DEFAULT_PG_PORT = "5432"
DEFAULT_PG_DATABASE = "your-database"
DEFAULT_PG_USER = "your-user"
DEFAULT_PG_PASSWORD = "your-password"
```

### Document Processing Settings
- **Chunk Size**: 1000 tokens
- **Chunk Overlap**: 200 tokens
- **Max Search Results**: 3 relevant chunks per query

## 📊 How It Works

1. **Document Upload**: Files are processed and split into chunks
2. **Embedding Generation**: OpenAI creates vector embeddings for each chunk
3. **Storage**: Embeddings and metadata are stored in PostgreSQL with pgvector
4. **Retrieval**: User queries are converted to embeddings and matched with stored documents
5. **Response**: GPT generates answers based on relevant document chunks

## 🔍 Database Schema

The application automatically creates these tables:
- `langchain_pg_collection`: Stores collection metadata
- `langchain_pg_embedding`: Stores document chunks and their vector embeddings

## 🐛 Troubleshooting

### PostgreSQL Connection Failed
- Verify PostgreSQL is running: `pg_isready`
- Check credentials and database name
- Ensure firewall allows connections on port 5432

### pgvector Extension Error
```sql
-- Run this in your PostgreSQL database:
CREATE EXTENSION IF NOT EXISTS vector;
```

### OpenAI API Error
- Verify your API key is valid
- Check you have sufficient credits
- Ensure you're using a valid model

### Document Processing Fails
- Check file format (PDF, TXT, DOCX only)
- Verify file is not corrupted
- Check application logs in telemetry panel

## 📝 Notes

- **Persistent Storage**: Documents remain in PostgreSQL even after restarting the app
- **Duplicate Detection**: Files are identified by MD5 hash to prevent duplicates
- **Cost**: OpenAI API charges apply for embeddings and completions
- **Performance**: Large documents take time to process on first upload, but queries are fast afterward

## 🔐 Security

- Never commit API keys to version control
- Use environment variables or secrets management
- Restrict PostgreSQL access with strong passwords
- Consider using connection pooling for production

## 📈 Future Enhancements

- [ ] Multi-user support with document isolation
- [ ] Document versioning
- [ ] Advanced filtering and metadata search
- [ ] Export/import document collections
- [ ] Support for more file formats
- [ ] Batch document processing

## 📄 License

MIT License - feel free to use and modify!

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

**Made with ❤️ using Streamlit, LangChain, and PostgreSQL**
