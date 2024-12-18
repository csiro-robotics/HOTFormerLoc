import "./App.css";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Appbar from "./components/Appbar/Appbar";
import DefaultPage from "./pages/DefaultPage";
import PaperPage from "./pages/Paper";
import DatasetPage from "./pages/Dataset";
import DownloadPage from "./pages/Download";


function App() {
  return (
    <Router basename="/hotformerloc">
      <Appbar siteName="HOTFormerLoc" />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<DefaultPage/>} />
          <Route path="/paper" element={<PaperPage/>} />
          <Route path="/dataset" element={<DatasetPage/>} />
          <Route path="/download" element={<DownloadPage/>} />
        </Routes>
      </main>
    </Router>
  );
}

export default App;
