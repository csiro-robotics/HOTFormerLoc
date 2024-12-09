import "./App.css";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Appbar from "./components/Appbar/Appbar";
import DefaultPage from "./pages/DefaultPage";
import DatasetPage from "./pages/Dataset";
import CitationPage from "./pages/Citation";
import DownloadPage from "./pages/Download";
import AcknowledgementsPage from "./pages/Acknowledgements";

function App() {
  return (
    <Router basename="/hotformerloc">
      <Appbar siteName="HOTFormerLoc" />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<DefaultPage/>} />
          <Route path="/dataset" element={<DatasetPage/>} />
          <Route path="/download" element={<DownloadPage/>} />
          <Route path="/acknowledgements" element={<AcknowledgementsPage/>} />
          <Route path="/citation" element={<CitationPage/>} />
        </Routes>
      </main>
    </Router>
  );
}

export default App;
