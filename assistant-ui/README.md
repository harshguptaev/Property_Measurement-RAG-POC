# Property Measurement RAG System - Assistant UI

This is a modern web interface for the Property Measurement RAG system, built with [Assistant UI](https://www.assistant-ui.com/) and Next.js.

## 🎯 Overview

The Assistant UI frontend provides a modern, interactive chat interface for analyzing property documents, roof reports, and measurements. It replaces the previous Gradio interface with a more professional and user-friendly experience.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Assistant UI  │    │   FastAPI       │    │   RAG System    │
│   (Next.js)     │◄──►│   Backend       │◄──►│   (Python)      │
│   Port: 3000    │    │   Port: 8000    │    │   AWS Bedrock   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Components

1. **Frontend (Assistant UI + Next.js)**
   - Modern chat interface with conversation threading
   - Property-specific dashboard and analytics
   - Real-time document statistics
   - Responsive design with Tailwind CSS

2. **Backend API (FastAPI)**
   - RESTful API endpoints for the RAG system
   - Document processing and analysis
   - Health monitoring and statistics
   - CORS support for frontend integration

3. **RAG System (Existing Python Backend)**
   - AWS Bedrock integration
   - Docling document processing
   - Vector store management
   - Multimodal analysis capabilities

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# From the project root directory
./start_assistant_ui.sh
```

This script will:
- Check system requirements
- Install all dependencies
- Start both backend and frontend services
- Provide access URLs and monitoring

### Option 2: Manual Setup

1. **Start the FastAPI Backend**
```bash
# Install Python dependencies
pip install fastapi uvicorn pydantic python-multipart

# Start the API server
python3 api_server.py
# Backend will be available at http://localhost:8000
```

2. **Start the Next.js Frontend**
```bash
# Navigate to frontend directory
cd assistant-ui

# Install Node.js dependencies
npm install

# Start development server
npm run dev
# Frontend will be available at http://localhost:3000
```

## 📱 Usage

### Web Interface

1. Open your browser to `http://localhost:3000`
2. Use the chat interface to ask questions about your property documents
3. View real-time system status in the right sidebar
4. Access conversation history in the left sidebar

### Quick Analysis Examples

- "What is the condition of the roof?"
- "Are there any structural issues?"
- "What repairs are recommended?"
- "Summarize the key findings"
- "Show me damage areas"
- "What are the cost estimates?"

### API Endpoints

The FastAPI backend provides these endpoints:

- `GET /health` - System health and status
- `POST /query` - Query the RAG system
- `GET /documents/stats` - Document statistics
- `GET /documents/search` - Search documents
- `POST /process-documents` - Process new documents

## 🛠️ Configuration

### Environment Variables

Create `.env.local` in the `assistant-ui` directory:

```env
# Backend API URL
RAG_BACKEND_URL=http://localhost:8000

# Optional: OpenAI API key for fallback
OPENAI_API_KEY=your_key_here
```

### Backend Configuration

The FastAPI backend uses the existing `config.yaml` for:
- AWS Bedrock settings
- Vector store configuration
- Model parameters

## 🎨 Features

### Modern Chat Interface
- **Assistant UI Components**: Professional chat UI with threading
- **Real-time Responses**: Streaming responses from the RAG system
- **Conversation History**: Persistent chat threads
- **Mobile Responsive**: Works on desktop and mobile devices

### Property-Specific Features
- **Document Dashboard**: Real-time statistics and system status
- **Quick Analysis Tools**: Pre-configured queries for property analysis
- **Category Organization**: Roof reports, measurements, images, assessments
- **Visual Status Indicators**: System health and connection status

### Developer Experience
- **TypeScript**: Full type safety throughout the frontend
- **Tailwind CSS**: Utility-first styling with consistent design system
- **Hot Reload**: Development server with instant updates
- **API Documentation**: Auto-generated docs at `/docs`

## 📁 Project Structure

```
assistant-ui/
├── src/
│   ├── app/
│   │   ├── api/chat/route.ts     # Chat API endpoint
│   │   ├── page.tsx              # Main application
│   │   └── layout.tsx            # App layout
│   ├── components/
│   │   ├── assistant-ui/         # Assistant UI components
│   │   │   ├── thread.tsx
│   │   │   ├── thread-list.tsx
│   │   │   └── ...
│   │   └── PropertyDashboard.tsx # Custom property dashboard
│   └── lib/
│       └── utils.ts              # Utility functions
├── package.json
└── .env.local                    # Environment configuration

api_server.py                     # FastAPI backend
start_assistant_ui.sh            # Startup script
```

## 🔧 Development

### Adding New Features

1. **Frontend Components**: Add React components in `src/components/`
2. **API Endpoints**: Add new endpoints in `api_server.py`
3. **Styling**: Use Tailwind CSS classes for consistent styling
4. **Types**: Add TypeScript interfaces for type safety

### Customization

- **Theming**: Modify Tailwind configuration in `tailwind.config.js`
- **Components**: Customize Assistant UI components in `src/components/assistant-ui/`
- **API Integration**: Modify backend integration in `src/app/api/chat/route.ts`

## 🐛 Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Kill existing processes
   lsof -ti:3000 | xargs kill -9  # Frontend
   lsof -ti:8000 | xargs kill -9  # Backend
   ```

2. **Dependencies Issues**
   ```bash
   # Reinstall Node.js dependencies
   cd assistant-ui && rm -rf node_modules package-lock.json && npm install
   
   # Reinstall Python dependencies
   pip install -r requirements.txt --force-reinstall
   ```

3. **Backend Connection Issues**
   - Check if FastAPI is running on port 8000
   - Verify environment variables in `.env.local`
   - Check CORS configuration in `api_server.py`

### Logs and Debugging

- **Frontend Logs**: Check browser console and terminal running `npm run dev`
- **Backend Logs**: Check terminal running `python3 api_server.py`
- **API Documentation**: Visit `http://localhost:8000/docs` for interactive API docs

## 🔄 Migration from Gradio

If you're migrating from the Gradio interface:

1. **Data Compatibility**: All existing vector stores and documents work without changes
2. **Configuration**: Same `config.yaml` and environment variables
3. **Features**: All Gradio features are available plus additional enhancements
4. **Performance**: Improved response times and user experience

## 📚 Documentation

- [Assistant UI Documentation](https://www.assistant-ui.com/docs)
- [Next.js Documentation](https://nextjs.org/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project uses the same license as the main Property Measurement RAG system.
