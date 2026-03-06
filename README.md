# arXiv Reader

A mobile-first app to browse astrophysics papers from arXiv in a swipeable card format.

Built for astronomers and astrophysics enthusiasts who want to stay up to date with the latest publications without drowning in PDFs.

## Features

- Swipe through the latest arXiv papers (new submissions only, no replacements)
- Filter by astrophysics subcategory: Galaxies, Cosmology, Planetary, High Energy, Stellar, Instrumentation
- "Today" button to see only papers from the current arXiv listing
- "For You" tab with personalized recommendations based on your likes
- Save papers to your library and export to CSV
- Google sign-in for cross-device sync
- Runs on iOS, Android, and Web

## Tech Stack

**Frontend**
- React Native + Expo (expo-router)
- React Native Gesture Handler + Reanimated
- AsyncStorage for local persistence

**Backend**
- FastAPI (Python)
- MongoDB (Motor async driver)
- arXiv API + HTML scraping for real-time paper feeds
- scikit-learn for paper recommendations (TF-IDF cosine similarity)

## Getting Started

### Prerequisites
- Node.js 18+
- Python 3.10+
- MongoDB running locally

### Backend

```bash
cd app/backend
pip install -r requirements.txt

# Create .env file
echo "MONGO_URL=mongodb://localhost:27017" > .env
echo "DB_NAME=arxiv_reader" >> .env

uvicorn server:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd app/frontend

# Create .env file
echo "EXPO_PUBLIC_BACKEND_URL=http://localhost:8000" > .env

yarn install
yarn web      # Browser
yarn ios      # iOS simulator
yarn android  # Android emulator
```

## How It Works

The backend scrapes arXiv's daily new submissions page (`arxiv.org/list/{category}/new`) to get today's papers, fetches their full metadata via the arXiv API, and serves them paginated. Cross-submissions are included; replacement submissions are excluded. The announcement date (not the submission date) is used as the publication date.

## Support

If you find this useful, consider [supporting the project](https://buymeacoffee.com/Nicolasmonsalves).

## License

MIT
