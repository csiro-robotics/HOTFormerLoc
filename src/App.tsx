import "./App.css";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navigation from "./components/Appbar";
import DefaultPage from "./pages/DefaultPage";
import DatasetPage from "./pages/Dataset";
import DatasetVisualisationPage from "./pages/DatasetVisualisation";
import CitationPage from "./pages/Citation";
import DownloadPage from "./pages/Download";
import AcknowledgementsPage from "./pages/Acknowledgements";

function App() {
  return (
    <Router basename="/hotformerloc">
      <Navigation />
      <main className="main-content">
        <Routes>
          <Route path="/" Component={DefaultPage} />
          <Route path="/dataset" Component={DatasetPage} />
          <Route
            path="/dataset-visualisation"
            Component={DatasetVisualisationPage}
          />
          <Route path="/download" Component={DownloadPage} />
          <Route path="/acknowledgements" Component={AcknowledgementsPage} />
          <Route path="/citation" Component={CitationPage} />
        </Routes>
      </main>
    </Router>
  );
}

export default App;
