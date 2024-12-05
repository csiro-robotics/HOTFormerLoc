import './App.css'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navigation from './components/Appbar';
import DefaultPage from './pages/DefaultPage';
import DatasetPage from './pages/Dataset';
import DatasetVisualisationPage from './pages/DatasetVisualisation';
import CitationPage from './pages/Citation';
import DownloadPage from './pages/Download';
import AcknowledgementsPage from './pages/Acknowledgements';

function App() {
  return (
    <Router basename="/hotformerloc">
      <Navigation />
      <Routes>
        <Route path="/" element={<DefaultPage />} />
        <Route path="/dataset" element={<DatasetPage />} />
        <Route path="/dataset-visualisation" element={<DatasetVisualisationPage />} />
        <Route path="/download" element={<DownloadPage />} />
        <Route path="/acknowledgements" element={<AcknowledgementsPage />} />
        <Route path="/citation" element={<CitationPage />}/>
      </Routes>
    </Router>
  )
}
export default App
