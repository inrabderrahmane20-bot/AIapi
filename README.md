# Traveltto API

Backend API for [Traveltto.com](https://www.traveltto.com) - A comprehensive city information and travel guide service.

## 🚀 Features

- **City Information**: Detailed data for 100+ cities worldwide
- **Wikipedia Integration**: Rich content and summaries
- **Image Galleries**: High-quality images from Wikimedia Commons
- **Geolocation**: Coordinates and interactive maps
- **Search Functionality**: Fast city search
- **Caching**: Performance optimized with disk and memory caching
- **RESTful API**: Clean, documented endpoints

## 📦 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/api/health` | GET | Health check and status |
| `/api/cities` | GET | Get all cities with previews |
| `/api/cities/{city_name}` | GET | Get detailed city information |
| `/api/search?q={query}` | GET | Search cities |
| `/api/regions` | GET | Get all regions |
| `/api/stats` | GET | Get API statistics |
| `/api/cache/clear` | POST | Clear cache |

## 🛠️ Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/traveltto-api.git
cd traveltto-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your configuration